# magma-bench 2.0.0

Evaluate agents on interactive robotic tasks using the shared agent HTTP protocol
v2. Bench executes and evaluates environment-visible decisions; the remote runtime
owns memory, model calls and internal continuations. No agent package is required
in the benchmark environment.

## Run

Start a compatible agent server, then run:

```bash
magma-bench run \
  --benchmark-root /path/to/magma-benchmark-files \
  --agent-address http://localhost:8000 \
  --agent-name experiment-1
```

The server must expose `/health`, `/v1/info` and `/v1/responses`. Bench validates
protocol version `2.0` and requests one candidate per episode observation. The
execution name defaults to the server's `agent_id`.

The configuration address remains `magma_agent_address`. Runtime options can be
provided as `benchmark.extra_keys` in the configuration and overridden with:

```bash
magma-bench run --benchmark-root /path/to/magma-benchmark-files \
  --extra-keys '{"inference_mode": true}'
```

Deterministic decoding is enabled by default. Use `--no-deterministic-decoding`
to disable it. A conflicting `extra_keys.inference_mode` is rejected.

User instructions are sent as `user`; environment feedback is sent as `env`,
including its complete JSON content. Initial history is placed in `memory.history`.
Subsequent memory is replaced by the runtime's returned memory without inspection.

## Annotated episode videos

```bash
pip install -e '.[video]'
magma-bench run --benchmark-root /path/to/magma-benchmark-files \
  --videos --video-fps 20 --video-hold-seconds 1
```

Each executed episode produces an annotated MP4 in its scenario `videos/`
directory. Enabling videos while resuming a run records only replayed episodes.

## Per-episode model logs

Use `--logs` or `--model-logs` with any compatible runtime. Each exchange produces:

- A Markdown file for that episode only, rendering every ordered
  `internal_step` with its component, complete prompt and raw model output.
- A response error section when the model runtime or parser rejects the output.

When the episode ends, `final.md` records its terminal status and the failure
details already available to the benchmark, such as judge reasoning, stage
verification scores and actual execution logs. Internal steps do not count as
additional benchmark decisions.

Logs are written under
`scenarios/.partial/<scenario>/model_logs/<skeleton>/<semantic>/<condition>/`
and move with the scenario directory when it completes.

## Results and migration

Version 2.0.0 uses result schema `2.0`. Runtime identity, version, protocol and
options must match when resuming a run with `--results-path`. Old result schemas
and agent adapters are unsupported; start a new result directory. Compiled
benchmark artifacts keep their existing format.

The CLI accepts an optional `--agent-name` instead of a positional agent family,
`--agent-address` instead of the old server-address aliases, and a JSON
`--extra-keys` object instead of arbitrary adapter arguments.

`magma-bench --help`, `magma-bench run --help` and `magma-bench --version` show the
available commands and package version.
