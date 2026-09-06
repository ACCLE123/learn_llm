# Qwen3 Retail Tool Agent

This is a separate project for building a local `Qwen3-4B` tool-calling agent
for the text-based Retail domain in `tau2-bench` (the current repository that
hosts the τ² benchmark).

The project intentionally starts with a small, observable agent loop before
adding reinforcement learning:

```text
user or tool message
  -> agent state (policy + tool schema + history)
  -> Qwen backend
  -> assistant text or typed tool call
  -> tau2 environment
  -> next tool/user message
```

## Scope

### Phase 0/1 — current work

- Keep a complete typed conversation and tool-call state.
- Inject the domain policy and available tool schemas into the model context.
- Use a pluggable backend: the included backend targets a locally loaded Qwen
  model, while tests use a deterministic fake backend.
- Validate tool names and arguments before an environment receives a call.
- Stop safely at a configured turn limit and expose each step for tracing.

### Later phases

1. Implement the τ² `HalfDuplexAgent` adapter and validate it on `mock`.
2. Run a frozen Qwen3-4B baseline on a small Retail development subset.
3. Add trajectory logging, evaluation reports, and failure classification.
4. Treat a complete Retail interaction as one rollout and extend
   `mini_grpo` with trajectory GRPO.

## Design boundaries

The model may choose a response or a tool call. It does **not** directly own
business authority: the environment/workflow owns schema validation,
permissions, policy enforcement, execution, and audit logs. The agent core
therefore avoids guessing tool results or repairing malformed calls into a
different semantic action.

## Local setup

The source tree has no mandatory third-party dependency for its unit tests:

```bash
cd tau2_retail_agent
python3 -m unittest discover -s tests -v
```

Run the τ² integration tests and the real mock-environment smoke test with the
dedicated environment:

```bash
conda activate tau2-agent
cd /Users/yangqi/Code2/GRPO/tau2_retail_agent
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/run_mock_smoke.py
```

The smoke test uses a deterministic backend, but invokes the official τ² mock
tools and feeds their real results back into the agent. It is the prerequisite
for replacing that backend with `TransformersQwenBackend`.

Run the real Qwen smoke test after the deterministic check passes:

```bash
# Download once in a normal terminal; it can take a while on a slow network.
hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

# Then run with the local checkpoint.
PYTHONPATH=src python scripts/run_qwen_mock.py --model models/Qwen3-4B
```

The first run downloads approximately 8.04 GB of model weights. The backend
uses CUDA when available and otherwise moves to MPS when PyTorch exposes it;
CPU remains a functional but slow fallback.

## Retail development baseline

The first Retail baseline uses the official train split and always selects the
first five task IDs in order. Inspect that fixed selection without loading Qwen
or calling a remote service:

```bash
PYTHONPATH=src python scripts/run_retail_baseline.py --dry-run
```

Retail uses τ²'s LLM user simulator, so an actual run requires credentials for
the LiteLLM provider behind the explicitly supplied user model. With DeepSeek:

```bash
export DEEPSEEK_API_KEY="your_api_key"
PYTHONPATH=src python scripts/run_retail_baseline.py \
  --model models/Qwen3-4B \
  --user-llm deepseek/deepseek-chat
```

Each task's sanitized τ² simulation, raw Qwen completions, reward, duration,
and tool-result errors are written to a timestamped directory below
`artifacts/retail_baseline/`.

Qwen thinking is disabled by default so tool calls arrive promptly. Add
`--enable-thinking` only when deliberately comparing a reasoning-enabled run.
Each local Qwen decision has a 90-second decoding budget by default; override
it with `--max-generation-seconds` when profiling slower hardware.

## τ² runtime

τ² keeps its task data next to its source tree, so the local checkout lives at
`vendor/tau2-bench/` and is intentionally ignored by this repository. The
current integration was verified against commit
`672227c6b6676edc20d57ea53b7000262aae77b9`.

To reproduce the runtime on a fresh machine:

```bash
conda create -n tau2-agent python=3.12 -y
conda activate tau2-agent
git clone https://github.com/sierra-research/tau2-bench.git vendor/tau2-bench
git -C vendor/tau2-bench checkout 672227c6b6676edc20d57ea53b7000262aae77b9
python -m pip install -e vendor/tau2-bench gymnasium torch 'transformers>=4.51.0'
```

For a real τ² run, use the benchmark's supported setup (`uv sync` from its
repository) and install the optional model dependencies in the same Python
environment. The exact Qwen checkpoint, decoding parameters, τ² version, and
user-simulator configuration must be frozen before recording a baseline.

The repository's dedicated runtime is the `tau2-agent` Conda environment. It
uses Python 3.12 because τ² v1 requires Python `>=3.12,<3.14`; the existing
`transformers` environment remains untouched.

## Current acceptance criteria

- `AgentCore` records policy, tools, user messages, assistant messages, and
  tool results in order.
- Unknown tools and invalid tool arguments fail before reaching a business
  environment.
- A turn budget prevents endless model/tool loops.
- The core can be tested without downloading a model or the benchmark.
