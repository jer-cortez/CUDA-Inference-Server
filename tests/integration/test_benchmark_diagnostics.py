async def test_private_diagnostics_report_actual_settings_without_secret_paths(client, settings):
    response = await client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    config = data["effective_config"]
    assert config["max_wait_ms"] == settings.max_wait_ms
    assert config["input_elems"] == settings.input_elems
    assert config["max_inflight_requests"] == settings.max_inflight_requests
    assert "api_keys_file" not in config
    assert "model_path" not in config
    assert data["process_id"]
    assert (await client.get("/healthz")).json()["process_id"] == data["process_id"]


async def test_diagnostics_identify_new_runtime(client_for, settings):
    async with client_for(settings) as first:
        first_id = (await first.get("/healthz")).json()["process_id"]
    async with client_for(settings) as second:
        assert (await second.get("/healthz")).json()["process_id"] != first_id
