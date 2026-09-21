from __future__ import annotations

import hashlib
import json
import logging
import stat
import uuid

import pytest

from cuda_db.config import RuntimeSettings
from cuda_db.security.keys import (
    KeyFileError,
    generate_key,
    load_key_file,
    main as keys_main,
    revoke_key,
)
from cuda_db.server.admission import AdmissionController, PredictionAdmissionMiddleware
from cuda_db.server.security import ApiKeyAuthenticator, AuthenticationMiddleware

from .conftest import TEST_INPUT_ELEMS, TEST_OUTPUT_ELEMS


def _settings(store, **overrides):
    values = {
        "input_elems": TEST_INPUT_ELEMS,
        "output_elems": TEST_OUTPUT_ELEMS,
        "require_auth": True,
        "api_keys_file": str(store),
    }
    values.update(overrides)
    return RuntimeSettings(**values)


def _authorization(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def test_generated_store_is_hashed_and_mode_0600(tmp_path):
    store = tmp_path / "keys.json"

    token = generate_key("pilot-a", store)
    document = json.loads(store.read_text())

    assert token.startswith("pilot-a.")
    assert token not in store.read_text()
    assert document == {
        "schema_version": 1,
        "keys": [
            {
                "id": "pilot-a",
                "sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
            }
        ],
    }
    assert stat.S_IMODE(store.stat().st_mode) == 0o600
    with pytest.raises(KeyFileError, match="already exists"):
        generate_key("pilot-a", store)


def test_key_file_validation_rejects_duplicate_ids_fields_and_uppercase_digest(tmp_path):
    store = tmp_path / "keys.json"
    digest = "a" * 64
    invalid_documents = [
        '{"schema_version":1,"schema_version":1,"keys":[]}',
        json.dumps(
            {
                "schema_version": 1,
                "keys": [
                    {"id": "same", "sha256": digest},
                    {"id": "same", "sha256": digest},
                ],
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "keys": [{"id": "pilot", "sha256": "A" * 64}],
            }
        ),
    ]

    for document in invalid_documents:
        store.write_text(document)
        with pytest.raises(KeyFileError):
            load_key_file(store)

    store.write_bytes(
        json.dumps({"schema_version": 1, "keys": []}).encode("utf-16")
    )
    with pytest.raises(KeyFileError, match="UTF-8"):
        load_key_file(store)


def test_key_cli_prints_plaintext_only_for_generate(tmp_path, capsys):
    store = tmp_path / "keys.json"

    assert keys_main(["generate", "--key-id", "pilot", "--store", str(store)]) == 0
    generated = capsys.readouterr()
    token = generated.out.strip()
    assert token.startswith("pilot.")
    assert generated.err == ""
    assert token not in store.read_text()

    assert keys_main(["revoke", "--key-id", "pilot", "--store", str(store)]) == 0
    revoked = capsys.readouterr()
    assert revoked.out == ""
    assert revoked.err == ""


async def test_authentication_is_before_admission_and_body_receive(tmp_path):
    store = tmp_path / "keys.json"
    generate_key("pilot", store)
    authenticator = ApiKeyAuthenticator(required=True)
    authenticator.load(str(store))
    controller = AdmissionController(1)
    held = controller.try_acquire()
    receive_calls = 0
    sent = []

    async def downstream(scope, receive, send):  # pragma: no cover
        raise AssertionError("unauthenticated request reached the application")

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b"secret", "more_body": False}

    async def send(message):
        sent.append(message)

    admission = PredictionAdmissionMiddleware(
        downstream, controller, max_request_bytes=16, request_timeout_ms=1_000
    )
    authentication = AuthenticationMiddleware(admission, authenticator)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/predict/",
        "headers": [],
    }

    await authentication(scope, receive, send)

    assert sent[0]["status"] == 401
    assert (b"www-authenticate", b"Bearer") in sent[0]["headers"]
    assert receive_calls == 0
    assert controller.inflight == 1
    held.finish_http()


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"authorization": "Basic abc"},
        {"authorization": "Bearer bad id.secret"},
        {"authorization": "Bearer pilot.bad.secret"},
    ],
)
async def test_missing_and_malformed_credentials_are_generic(
    client_for, tmp_path, headers
):
    store = tmp_path / "keys.json"
    generate_key("pilot", store)
    async with client_for(_settings(store)) as client:
        response = await client.post(
            "/predict", json={"input": [1.0] * TEST_INPUT_ELEMS}, headers=headers
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "authentication required"}
    assert response.headers["www-authenticate"] == "Bearer"
    uuid.UUID(response.headers["x-request-id"])


async def test_duplicate_authorization_headers_are_rejected(client_for, tmp_path):
    store = tmp_path / "keys.json"
    token = generate_key("pilot", store)
    async with client_for(_settings(store)) as client:
        response = await client.post(
            "/predict",
            json={"input": [1.0] * TEST_INPUT_ELEMS},
            headers=[
                ("authorization", f"Bearer {token}"),
                ("authorization", f"Bearer {token}"),
            ],
        )
    assert response.status_code == 401


async def test_query_credentials_are_not_accepted(client_for, tmp_path):
    store = tmp_path / "keys.json"
    token = generate_key("pilot", store)
    async with client_for(_settings(store)) as client:
        response = await client.post(
            "/predict",
            params={"api_key": token},
            json={"input": [1.0] * TEST_INPUT_ELEMS},
        )
    assert response.status_code == 401


async def test_key_changes_apply_only_after_restart(client_for, tmp_path):
    store = tmp_path / "keys.json"
    old_token = generate_key("old", store)
    settings = _settings(store)

    async with client_for(settings) as client:
        assert (
            await client.post(
                "/predict",
                json={"input": [1.0] * TEST_INPUT_ELEMS},
                headers=_authorization(old_token),
            )
        ).status_code == 200
        new_token = generate_key("new", store)
        assert (
            await client.post(
                "/predict",
                json={"input": [1.0] * TEST_INPUT_ELEMS},
                headers=_authorization(new_token),
            )
        ).status_code == 401

    async with client_for(settings) as client:
        assert (
            await client.post(
                "/predict",
                json={"input": [1.0] * TEST_INPUT_ELEMS},
                headers=_authorization(new_token),
            )
        ).status_code == 200
        revoke_key("old", store)
        assert (
            await client.post(
                "/predict",
                json={"input": [1.0] * TEST_INPUT_ELEMS},
                headers=_authorization(old_token),
            )
        ).status_code == 200

    async with client_for(settings) as client:
        assert (
            await client.post(
                "/predict",
                json={"input": [1.0] * TEST_INPUT_ELEMS},
                headers=_authorization(old_token),
            )
        ).status_code == 401


async def test_deployment_mode_forces_auth_and_disables_documentation(
    client_for, tmp_path
):
    store = tmp_path / "keys.json"
    token = generate_key("pilot", store)
    settings = _settings(store, require_auth=False, deployment_mode=True)
    async with client_for(settings) as client:
        assert (await client.get("/docs")).status_code == 404
        assert (await client.get("/redoc")).status_code == 404
        assert (await client.get("/openapi.json")).status_code == 404
        assert (
            await client.post(
                "/predict", json={"input": [1.0] * TEST_INPUT_ELEMS}
            )
        ).status_code == 401
        assert (
            await client.post(
                "/predict",
                json={"input": [1.0] * TEST_INPUT_ELEMS},
                headers=_authorization(token),
            )
        ).status_code == 200


async def test_bad_key_store_fails_before_native_runtime(monkeypatch, tmp_path):
    import cuda_db.server.app as app_module

    store = tmp_path / "keys.json"
    store.write_text("not json")
    runtime_constructed = False

    class RuntimeMustNotStart:
        def __init__(self, config):
            nonlocal runtime_constructed
            runtime_constructed = True

    monkeypatch.setattr(app_module, "InferenceRuntime", RuntimeMustNotStart)
    app = app_module.create_app(_settings(store))

    with pytest.raises(KeyFileError):
        async with app.router.lifespan_context(app):
            pass
    assert not runtime_constructed


def test_required_auth_needs_a_store_path():
    with pytest.raises(ValueError, match="api_keys_file"):
        RuntimeSettings(require_auth=True)


def test_auth_environment_settings(monkeypatch):
    monkeypatch.setenv("CUDA_DB_REQUIRE_AUTH", "true")
    monkeypatch.setenv("CUDA_DB_API_KEYS_FILE", "/keys.json")
    monkeypatch.setenv("CUDA_DB_DEPLOYMENT_MODE", "false")
    settings = RuntimeSettings.from_env()
    assert settings.require_auth
    assert settings.api_keys_file == "/keys.json"
    assert not settings.deployment_mode


async def test_request_log_is_structured_and_does_not_log_query_token(
    client_for, tmp_path, caplog
):
    store = tmp_path / "keys.json"
    token = generate_key("pilot", store)
    caplog.set_level(logging.INFO, logger="cuda_db")
    async with client_for(_settings(store)) as client:
        response = await client.post(
            "/predict?credential=must-not-appear",
            json={"input": [1.0] * TEST_INPUT_ELEMS},
            headers={**_authorization(token), "x-request-id": "client-controlled"},
        )

    assert response.status_code == 200
    uuid.UUID(response.headers["x-request-id"])
    assert response.headers["x-request-id"] != "client-controlled"
    records = [r for r in caplog.records if r.name == "cuda_db.request"]
    assert records
    event = json.loads(records[-1].message)
    assert event["endpoint"] == "/predict"
    assert event["status"] == 200
    assert event["key_id"] == "pilot"
    assert "must-not-appear" not in caplog.text
    assert token not in caplog.text
