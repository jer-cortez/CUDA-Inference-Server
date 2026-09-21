# API-key authentication

cuda-db can require a bearer token for both prediction endpoints. Authentication
runs before admission control and before the request body is read, so an invalid
client cannot occupy an inference slot or make the server buffer its payload.

## Create a key store

Generate the first key with the bundled administration command:

```console
python -m cuda_db.security.keys generate --key-id pilot-a --store /run/secrets/cuda-db-keys.json
```

The command prints a token such as `pilot-a.<secret>` exactly once. Capture it in
your secret manager; it cannot be recovered from the file. The store contains
only the key id and SHA-256 digest of the complete token and is atomically written
with mode `0600`.

Send the token in the standard header:

```text
Authorization: Bearer pilot-a.<secret>
```

Tokens in query parameters are never accepted. Missing, duplicated, malformed,
unknown, and revoked credentials all receive the same `401` response.

## Server configuration

The relevant environment variables are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CUDA_DB_REQUIRE_AUTH` | `false` | Require a valid key on prediction requests. |
| `CUDA_DB_API_KEYS_FILE` | empty | Absolute or working-directory-relative key-store path. |
| `CUDA_DB_DEPLOYMENT_MODE` | `false` | Force authentication and disable `/docs`, `/redoc`, and `/openapi.json`. |

When authentication is required, the file must exist, contain at least one key,
and match this strict schema:

```json
{
  "schema_version": 1,
  "keys": [
    {
      "id": "pilot-a",
      "sha256": "64 lowercase hexadecimal characters"
    }
  ]
}
```

Invalid configuration aborts startup before the inference runtime is created.
Changes intentionally take effect only after a restart, giving each process an
immutable view of its credentials.

Deployment mode does not imply `CUDA_DB_REQUIRE_GPU`; CPU-only environments can
therefore run security checks while still receiving deployment-mode protections.
Health and readiness endpoints remain unauthenticated for an internal
orchestrator. Restrict them at the network or reverse-proxy boundary.

## Rotation and revocation

Add the replacement key, distribute its one-time token, then restart the server:

```console
python -m cuda_db.security.keys generate --key-id pilot-b --store /run/secrets/cuda-db-keys.json
```

After clients have moved, remove the old id and restart again:

```console
python -m cuda_db.security.keys revoke --key-id pilot-a --store /run/secrets/cuda-db-keys.json
```

The command refuses to overwrite an existing id. Revoking the final key leaves a
valid management file with an empty list, but a server requiring authentication
will refuse to start from it.

## Logging and request identity

Every HTTP response includes a server-generated `X-Request-ID`; an incoming
header with that name is ignored. Structured request logs contain only the
request id, method, normalized endpoint, status, duration, direct peer address,
and authenticated key id. Query strings, headers, request and response bodies,
tokens, tensors, and arbitrary unknown paths are not logged. The application
does not trust forwarded client-address headers.
