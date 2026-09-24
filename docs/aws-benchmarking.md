# Milestone 4: AWS GPU benchmarking

The deliverable is a measured deployment recommendation between `g4dn.xlarge`
and `g6f.large` in provisional region `us-east-1`. Local tooling is preparatory;
it does not establish GPU correctness, memory fit, AWS recovery or capacity.
Permanent infrastructure, public launch, autoscaling and continuous monitoring
belong to Milestone 5.

## Before paid execution

Confirm the AWS account through an approved profile/SSO session, region,
instance offerings, available quota, compatible AMI/driver, current Linux
On-Demand prices, maximum runtime and approved total budget. Run one GPU
candidate at a time. Include existing instance usage when checking the
[G/VT vCPU quota](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html).
The T4 baseline and fractional L4 candidate have different CPUs and host RAM;
these are whole-instance comparisons. Check the
[instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html).
G6f requires its supported GRID setup; AWS's
[launch announcement](https://aws.amazon.com/about-aws/whats-new/2025/07/amazon-ec2-g6f-instances-fractional-gpus/)
specifies GRID 18.4 or later. Verify current compatibility before choosing an AMI.

Record rates and their retrieval time from
[EC2 pricing](https://aws.amazon.com/ec2/pricing/on-demand/),
[VPC pricing](https://aws.amazon.com/vpc/pricing/), and the applicable storage,
registry and transfer price pages. No illustrative price in this repository
is a live quote.

Before starting instances, tag every experiment resource with an experiment
ID, owner and expiry; establish and verify a scheduled stop using explicit
instance IDs. Include the load generator. Verify the schedule still applies
after the lifecycle stop/start test. Billing alerts are supplementary and do
not enforce a spending limit. Stopped instances can retain billable storage
and address resources.

Screening is 15 runs per candidate (3 batch sizes × 5 concurrency levels).
At 30 seconds warmup plus 120 seconds measurement this is 37.5 minutes per
candidate, before startup, changes, validation or overload tests. Three
five-minute repetitions add 15 measured minutes per shortlisted configuration;
the winner also needs a 15-minute soak. Budget these stages and supporting
resources before launch. Eliminate correctness-failing configurations early.

## Deployment and private diagnostics

Build the pinned image and model artifacts using [the GPU runbook](../docker/README.md),
then use [the authenticated HTTPS deployment](../docker/SECURITY.md). Preserve
the image digest, Git revision (and whether dirty), model/reference hashes,
AMI, driver, instance type, region/AZ, timestamps, and client location.
Keep model artifacts on persistent EBS, not ephemeral instance storage.

Only prediction requests pass through Caddy. `/healthz` and `/readyz` stay
private. The benchmark can use a private diagnostics URL when legitimately
reachable, or a command file containing an argv JSON array. A concrete command
file on the load generator is:

```json
["ssh", "gpu-host", "python3", "/srv/cuda-db/docker/private_diagnostics.py"]
```

Configure `gpu-host` in SSH configuration with host-key verification and
agent/profile access. The client appends `/readyz` or `/healthz`; the helper
uses `docker compose exec` to query loopback inside the inference container.
It publishes no port. Use an absolute `--compose-file` argument in the array
if not using the default deployment Compose file. The remote Compose
environment must have the deployment variables configured. Command stdout
must contain only a JSON response; nonzero exit means a failed probe.
Command files are trusted executable configuration and must be reviewed.

Diagnostics include an allowlisted `effective_config` and a per-startup
`process_id`. A process/configuration change invalidates a measurement.
Scheduler counter snapshots bracket measured traffic after warmup; maxima and
cumulative averages must not be represented as per-run deltas. Do not run
other inference traffic while collecting benchmark counters.

Credentials come from `CUDA_DB_BENCHMARK_TOKEN` or a protected token file using
`--token-file` / `CUDA_DB_BENCHMARK_TOKEN_FILE`. Do not put tokens in CLI arguments,
URLs, provenance files or version control. `--ca-file` /
`CUDA_DB_BENCHMARK_CA_FILE` adds a trusted CA while retaining hostname and
certificate verification. A test CA is suitable for the private pilot;
provision a certificate for the hostname actually used by the load generator.
Never use an insecure TLS bypass. The test-CA setup in `docker/test_https.py`
is an ephemeral local test harness, not an AWS deployment recipe.

## Correctness and recovery gates

For each batch size 1, 4 and 8, restart the configured service, collect
effective settings, then run the reference smoke check. Verify model and
reference checksums, eight distinct input/output mappings, finite 1000-class
outputs, matching top-1 and maximum absolute error ≤ 0.01. Do not relax the
tolerance. Run the four installed-wheel model tests from the GPU runbook.
Retain GPU activity evidence during repeated inference; provider selection
alone does not prove execution of all model operators on GPU.

Use disposable containers to test missing models, checksum mismatch and no
GPU access. Each must fail startup. Keep their outputs and exit statuses.
An OOM or unsupported G6f runtime is a candidate failure; do not change model,
precision or input shape silently to make it fit.

Test separately and retain machine-readable outcomes:

1. Container restart while idle.
2. Graceful container stop with inference requests in flight.
3. Container start, readiness and reference validation.
4. EC2 stop/start, service recovery and reference validation.

Record readiness/startup time, stop/drain time, container exit status, OOM or
forced-kill evidence, and every client outcome. HTTP submission does not prove
server admission; failures with unknown admission must stay unknown. Interrupted
requests must not be described as lossless recovery. Bound test command timeouts
and inspect the host if Docker times out rather than assuming it stopped.

The EC2 test is an operator-controlled step after cloud approval. Record instance
ID and addresses before stop, wait for `stopped`, start, wait for instance/system
checks, resolve the new address, and reconnect with SSH host-key verification.
Update test DNS if needed. Confirm persistent model checksums, protected token
store permissions and Caddy certificate/CA continuity. Verify Docker starts at
boot and the configured restart policy actually recovers the service. Then
rerun readiness and numerical reference validation. Container restart alone is
not evidence of EC2 recovery. Preserve the EC2 timestamps and API/console state
transition records alongside the container evidence.

## Performance protocol

Use a separate same-region load generator with adequate CPU/network capacity.
Keep its instance type, location and software consistent between candidates;
record AZ and cross-AZ transfer implications. Capture load-generator CPU/RAM
and scheduling delay as well as inference-host telemetry. A laptop run is an
optional user-experience measurement, not the capacity comparison.

Keep image, model, precision, request format, one Uvicorn worker, admission
limit and executor settings constant. Restart between batching configurations
and inspect effective settings. Screen:

| Maximum batch | Wait | Concurrency |
|---|---|---|
| 1 | 0 ms | 1, 2, 4, 8, 16 |
| 4 | 5 ms | 1, 2, 4, 8, 16 |
| 8 | 5 ms | 1, 2, 4, 8, 16 |

Add an explicit overload test above the actual admission limit. Use binary
requests for the main comparison and a smaller JSON comparison to measure
serialization overhead. Precompute payloads outside the timed request path;
record that client-side input encoding is excluded. Observed batch size one
is valid, especially under sparse arrivals.

Use at least 30 seconds warmup; extend until timings, memory and clocks have
stabilized. Screen for two minutes per run. For shortlisted configurations,
run three five-minute repetitions at the workload of interest, rotate candidate
or run order where practical, and report counts and variation. Run a 15-minute
soak for the proposed winner. Low-volume p99 estimates have little tail support;
retain successful sample counts and label this limitation.

Run scheduled arrivals at 1, 5, 10 and 25 requests/second, then short periods
above sustainable capacity. Include idle-to-active transitions: record the
first responses after a defined idle interval without warmup, separately from
steady-state runs. Bound outstanding client work. The schedule must continue
independently of response completion; report offered, attempted, successful,
rejected and missed arrivals plus scheduling delay and unfinished requests.
Keep actual-send-to-response and scheduled-arrival-to-response latency distinct.
Failures and missed arrivals do not disappear into successful-response latency
percentiles. A slow load generator invalidates server-capacity conclusions.

Resource sampling is periodic: it can miss memory peaks. Preserve OOM flags,
container exits and application errors alongside samples. Unsupported fractional
GPU telemetry is unavailable, not zero. Keep sample timestamps for joining with
run start/end times, and preserve per-GPU identity.

### Commands after host approval and correctness validation

Install the benchmark dependencies in the load generator's Python environment
(`httpx`, `numpy`; plotting additionally needs `matplotlib`). The remote-only
runner does not require a CUDA toolkit or the native extension on the client.
Create the diagnostics command file above and a protected token file. Use the
actual hostname, CA and file paths in place of these examples:

```sh
python docker/smoke_test.py --models models \
  --url https://inference.example.test \
  --diagnostics-command-file diagnostics.json \
  --token-file /secure/benchmark-token --ca-file /secure/test-ca.crt

python benchmarks/load_test.py --url https://inference.example.test \
  --mode dynamic --diagnostics-command-file diagnostics.json \
  --token-file /secure/benchmark-token --ca-file /secure/test-ca.crt \
  --concurrency 1,2,4,8,16 \
  --closed-loop-warmup-s 30 --closed-loop-duration-s 120 \
  --instance-type g4dn.xlarge --region us-east-1 \
  --client-location us-east-1-load-generator --output g4dn-b8-screen

python benchmarks/load_test.py --url https://inference.example.test \
  --mode dynamic --diagnostics-command-file diagnostics.json \
  --token-file /secure/benchmark-token --ca-file /secure/test-ca.crt \
  --request-rate 1,5,10,25 --warmup-s 30 --duration-s 300 \
  --max-outstanding 128 --instance-type g4dn.xlarge --region us-east-1 \
  --client-location us-east-1-load-generator --output g4dn-b8-rates-repeat1
```

Add `--image-digest`, `--ami-id`, `--availability-zone`, `--driver-version`,
and `--server-git-revision` from the recorded deployment inventory. Missing
values remain unknown. The server's effective configuration is authoritative:
`--mode` labels an external run and does not reconfigure the remote service.
Apply each batch/wait configuration on the GPU host, restart and validate it
before invoking the client. Use `--mode serial` for the actual batch-1/wait-0
configuration. Repeat with unique output names (the benchmark overwrites the
same stem) and use `--endpoint json` for the smaller JSON comparison.
Results are written under `benchmarks/results/` and ignored by Git.

Run sampling on both machines with unique output paths:

```sh
python benchmarks/resource_sampler.py --interval 1 --duration 900 \
  --output benchmarks/results/gpu-host-resources.jsonl
```

Run the container lifecycle check **on the GPU host**, where Docker controls
the intended Compose project. Use a local diagnostics command file such as
`["python3", "/srv/cuda-db/docker/private_diagnostics.py"]`:

```sh
python docker/lifecycle_test.py --models models \
  --url https://inference.example.test \
  --diagnostics-command-file local-diagnostics.json \
  --token-file /secure/benchmark-token --ca-file /secure/test-ca.crt \
  --inflight-requests 16 --output benchmarks/results/lifecycle-b8.json
```

This command restarts/stops/starts the inference service. Run it separately
from performance measurements. An inconclusive overlap check needs another
attempt and evidence review; it does not prove graceful in-flight recovery.
The separate EC2 stop/start test remains the operator step described above.

## Comparison report and cost ledger

Run `python benchmarks/report.py inventory.json --output comparison.md` to
produce Markdown and companion JSON. Paths below are relative to the inventory.
Create the inventory from actual results and dated rates; this example omits
prices deliberately:

```json
{
  "experiments": [
    {
      "candidate": "g4dn.xlarge",
      "results": "g4dn-b8.json",
      "instance_hourly_usd": null,
      "price_checked_at": null,
      "price_source": null,
      "correctness": {"status": "pending", "evidence": null},
      "recovery": {"status": "pending", "evidence": null}
    }
  ],
  "targets": {},
  "resources": [],
  "spend": [],
  "spend_complete": false
}
```

After agreeing workload targets, `targets` accepts `required_rps`, `p95_ms`
and/or `p99_ms`, `max_error_fraction`, and `headroom_fraction` (e.g. 0.2 means
20% throughput above required rate). Evidence files must exist and statuses
must explicitly be `passed` to make a configuration eligible. The tool uses
operator-supplied evidence statuses; review their contents before approval.
Keep separate experiment entries for configurations with different gate results.
The automated match is provisional and must be reviewed against all repetitions,
soak/resource evidence and scheduled-arrival latency. Without targets it produces
a capacity envelope, not an invented SLA.

Each spend entry has `category`, `quantity`, `unit`, `unit_price_usd`, and
optionally `source`/`checked_at`. Include GPU runtime, load-generator runtime,
EBS, registry storage/requests, public IPv4, and applicable transfer charges.
Mark whether the ledger is complete; retain billing reconciliation separately.
The compute-only comparison is:

`hourly instance price × 1000 / (successful requests/second × 3600)`.

Operating cost depends on the intended traffic and availability schedule.
At sparse traffic, include idle instance hours; do not substitute saturated
throughput into a low-utilization operating-cost estimate. Report fixed
availability costs, ancillary charges and actual experiment spend separately.
Choose the cheapest reliable configuration meeting the agreed target with
measured headroom. A different batch size may win at each arrival rate.
Optionally list resource JSONL paths in `resources`; the companion report JSON
includes per-metric sampled maxima/means and available sample counts. These
summaries cover each entire supplied file, not inferred per-run intervals.

## Export and cleanup

Copy raw result JSON/CSV, resource JSONL, validation/lifecycle evidence, image
and model identities, metadata, price snapshots and the report to the approved
destination before cleanup; verify checksums of exported files. Exclude tokens,
key stores, private keys and certificate-state volumes from ordinary result
archives.

Inventory exact experiment-tagged instances, EBS volumes/snapshots, registry
images, addresses, test DNS records, security groups and stop schedules. Remove
only approved temporary resources by explicit ID. Preserve intentionally
retained model or certificate state and report its owner, purpose and cost.
Stopping compute is not full cleanup. Finish with an explicit list of removed
and retained billable resources. The local scripts do not provision, stop or
delete AWS resources automatically.
