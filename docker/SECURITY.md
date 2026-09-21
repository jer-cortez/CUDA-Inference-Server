# Authenticated HTTPS deployment

`docker-compose.deploy.yml` is the standalone pilot deployment. It does not
extend or merge the local GPU Compose file, so the inference container never
inherits the development `127.0.0.1:8000` host publication. Caddy is the only
public service and publishes TCP/443 only. HTTP/3 is disabled so Caddy does not
open a UDP listener; HTTP/1.1 and HTTP/2 remain enabled.

This is a single-host deployment boundary, not cloud provisioning. It does not
create DNS, a firewall, a load balancer, a registry, an account, or an AWS
resource. Complete the GPU checks in [README.md](README.md) on the target host
before treating it as production-ready.

## Security boundary

- Caddy accepts exactly `POST /predict` and `POST /predict/raw`. Every other
  method or path, including `/healthz`, `/readyz`, `/docs`, `/redoc`, and
  `/openapi.json`, receives a generic 404 without reaching the application.
- The inference container has no host port and is attached only to an internal
  Docker bridge. Caddy joins that bridge and a separate edge bridge.
- Both prediction routes require `Authorization: Bearer <id.secret>`. The key
  store contains only SHA-256 digests. Authentication runs before the
  application reads the request body.
- Caddy limits bodies to 4 MiB without pre-buffering them, allows at most 30
  seconds to read an upload, caps request headers at 16 KiB, and waits up to 35
  seconds for application response headers. The application independently
  enforces a 4 MiB / 30 second limit.
- Full-duplex HTTP handling lets Caddy return the application's authentication
  rejection before an HTTP/1 client finishes sending its body. This preserves
  the auth-before-body boundary instead of letting Go drain an unauthenticated
  upload before writing the 401 response.
- Uvicorn proxy-header trust and raw access logs are disabled. Caddy strips
  `Forwarded`, `X-Forwarded-*`, and `X-Real-IP`, has no access logger, and
  suppresses request-associated proxy error records. Application audit records
  contain bounded fields, never authorization values, query strings, or input
  bodies. Do not enable Caddy debug/access logs or Uvicorn access logs without a
  fresh disclosure review.
- Caddy's admin listener and persisted dynamic config are disabled. Certificate
  state and private keys remain in the `caddy_data` Docker volume; protect that
  volume as secret material.

The body and timeout limits reduce accidental resource exhaustion, but they are
not rate limiting or denial-of-service protection. Apply network-level source
controls or a carefully configured upstream rate limiter if the pilot threat
model requires them. Do not trust upstream client-address headers in the
application.

## Host and TLS prerequisites

In addition to the NVIDIA/Docker requirements in [README.md](README.md), the
host needs:

1. A public DNS A/AAAA record for one stable hostname resolving directly to the
   host (remove an unusable AAAA record rather than leaving it stale).
2. Inbound and outbound TCP/443 permitted. Leave TCP/80 closed; it is not used.
3. No other process bound to TCP/443.
4. A contact email for the ACME account.

The production Caddyfile disables automatic HTTP redirects and ACME HTTP-01,
leaving TLS-ALPN-01 on port 443. A CDN or TLS-terminating proxy in front of this
host can prevent that challenge from working. In that topology, provision DNS
validation deliberately with a reviewed Caddy DNS-provider build; the stock
image used here does not bundle provider plugins.

Caddy is pinned to the official multi-platform image
`caddy:2.11.4-alpine@sha256:de23def33b17fb5d1290b0f6c2add1d70780e52341896c00a4c8a2a2fe9d355e`.
The tag and OCI index digest were verified against the official registry on
2026-09-21. Re-verify both the release and digest before upgrading.

## Create the API-key store

The on-disk schema is strict:

```json
{
  "schema_version": 1,
  "keys": [
    {"id": "pilot-a", "sha256": "<64 lowercase hexadecimal digest characters>"}
  ]
}
```

The placeholder above is intentionally not a usable record. Do not hand-author
digests or commit a real store. First build the serving image; building needs no
GPU, although the later production run does:

```sh
docker build --platform linux/amd64 -f docker/Dockerfile \
  -t cuda-db:gpu-local .
sudo install -d -o 10001 -g 10001 -m 0700 /srv/cuda-db/secrets
docker run --rm --network none --read-only --user 10001:10001 \
  --cap-drop ALL --security-opt no-new-privileges \
  --entrypoint python \
  --mount type=bind,source=/srv/cuda-db/secrets,target=/keys \
  cuda-db:gpu-local -m cuda_db.security.keys generate \
  --key-id pilot-a --store /keys/api_keys.json
sudo chmod 0400 /srv/cuda-db/secrets/api_keys.json
```

Move the one-time stdout value directly into the intended secret manager. The
token has the form `pilot-a.secret` and is printed only by `generate`.
The store is atomically written mode 0600 by the CLI; mode 0400 is sufficient
once administration is complete. UID/GID 10001 is the non-root application
identity and must be able to traverse the directory and read the file. Compose
file-backed secrets can be implemented as bind mounts and may ignore requested
secret UID/GID/mode fields, so host ownership is required rather than relying
on Compose metadata.

Run key-administration operations serially. The CLI also takes a sibling lock
for each read/modify/replace, but that does not make simultaneous operator
workflows or key distribution safe.

## Start and verify

Set absolute paths on the target host, then validate the resolved configuration
before starting it:

```sh
export CUDA_DB_HOSTNAME=api.example.com
export CUDA_DB_TLS_EMAIL=operations@example.com
export CUDA_DB_API_KEYS_HOST_FILE=/srv/cuda-db/secrets/api_keys.json
export CUDA_DB_MODELS_DIR=/srv/cuda-db/models

docker compose -f docker/docker-compose.deploy.yml config --quiet
docker compose -f docker/docker-compose.deploy.yml up -d
docker compose -f docker/docker-compose.deploy.yml ps
docker compose -f docker/docker-compose.deploy.yml logs --tail 100 inference caddy
```

Do not put a bearer token in a URL. Read it from the secret manager into the
client environment and send it only in the header:

```sh
curl --fail-with-body \
  -H "Authorization: Bearer $CUDA_DB_CLIENT_TOKEN" \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @request.f32 \
  "https://$CUDA_DB_HOSTNAME/predict/raw"
```

Verify that external health and documentation paths return 404. Inspect health
only from inside the private service:

```sh
docker compose -f docker/docker-compose.deploy.yml exec -T inference \
  python /opt/cuda-db/healthcheck.py
docker compose -f docker/docker-compose.deploy.yml port inference 8000
```

The final command must print nothing. `docker compose ps` should show only
`0.0.0.0:443->443/tcp` (and possibly the IPv6 equivalent) on Caddy, with no
published inference port and no port 80.

## Rotate or revoke a key

The application loads keys once at startup. The CLI atomically replaces the
store inode, so restart is not enough to guarantee that a file-backed Compose
secret follows the new inode: force-recreate the inference service after every
change.

Use an overlap rotation:

1. Generate `pilot-b` into the same store and place its printed token in the
   secret manager.
2. Restore host ownership/read permissions if the administration identity was
   different.
3. Force-recreate inference so it loads the new store.
4. Test `pilot-b`, then move all callers from `pilot-a` to `pilot-b`.
5. Revoke `pilot-a` and force-recreate inference again.

```sh
docker run --rm --network none --read-only --user 10001:10001 \
  --cap-drop ALL --security-opt no-new-privileges \
  --entrypoint python \
  --mount type=bind,source=/srv/cuda-db/secrets,target=/keys \
  cuda-db:gpu-local -m cuda_db.security.keys generate \
  --key-id pilot-b --store /keys/api_keys.json
sudo chmod 0400 /srv/cuda-db/secrets/api_keys.json
docker compose -f docker/docker-compose.deploy.yml up -d --no-deps \
  --force-recreate inference

# After callers have moved to pilot-b:
docker run --rm --network none --read-only --user 10001:10001 \
  --cap-drop ALL --security-opt no-new-privileges \
  --entrypoint python \
  --mount type=bind,source=/srv/cuda-db/secrets,target=/keys \
  cuda-db:gpu-local -m cuda_db.security.keys revoke \
  --key-id pilot-a --store /keys/api_keys.json
sudo chmod 0400 /srv/cuda-db/secrets/api_keys.json
docker compose -f docker/docker-compose.deploy.yml up -d --no-deps \
  --force-recreate inference
```

Never revoke the last working key before a replacement has been loaded and
tested. A store with no keys fails application startup closed.

## Local trusted-CA validation

`docker-compose.https-test.yml` uses the real application with the CPU stub and
the same Caddy routing/security policy. It publishes only loopback port 8443,
uses Caddy's local CA, and makes no external certificate request. The test
harness extracts the generated root certificate and validates TLS normally; it
does not use `curl -k` or disable certificate verification.

Run the harness from the repository root:

```sh
python3 docker/test_https.py
```

It uses a temporary key store outside the Docker build context and checks valid
and invalid credentials, hidden endpoints, the body limit, spoofed forwarding
headers, absence of a host inference port, a generic upstream-failure response,
and CA continuity across a Caddy restart. The named local CA volume is kept only
for the duration selected by the harness; never reuse its credentials in the
public deployment.

## Recovery and updates

Restarting Caddy retains certificates because `/data` is a named volume. Do not
run `docker compose down --volumes` during routine recovery; deleting that
volume destroys the ACME account and certificate state and can trigger issuance
rate limits. Backups containing `/data` must be encrypted and access-controlled.

Review Caddy and base-image security releases regularly. For an update, verify
the exact official tag/digest, update both Compose files together, validate both
Caddyfiles with the pinned image, run the local HTTPS harness, and then roll the
pilot. GPU inference correctness and stop/restart behavior still require the
separate GPU-host validation; a successful CPU HTTPS harness does not establish
CUDA correctness.
