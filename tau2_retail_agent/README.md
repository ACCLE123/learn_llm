# Qwen3 Retail Tool Agent

This is a separate project for building a local `Qwen3-1.7B` tool-calling agent
for the text-based Retail domain in `tau2-bench` (the current repository that
hosts the τ² benchmark).

The project intentionally starts with a small, observable agent loop before
adding reinforcement learning:

```text
user or tool message
  -> Observe: compact tool-attributed facts
  -> Plan: private typed decision (read/propose) plus schema retrieval
  -> Act: Qwen chooses text or a typed tool call
  -> Verify: require a read after a successful write
  -> Recover: inspect facts after an error before another write
  -> τ² environment -> next tool/user message
```

## Scope

### Phase 0/1 — current work

- Keep a complete typed conversation and tool-call state.
- Inject the domain policy and only the schema-retrieved candidate tools into
  the model context.
- Use a pluggable backend: the included backend targets a locally loaded Qwen
  model, while tests use a deterministic fake backend.
- Validate tool names and arguments before an environment receives a call.
- Classify τ² tools as read/write/think/generic from their declared metadata.
- Require explicit confirmation before write actions, consume that confirmation
  after one write decision, and use a read-only verify phase after success.
- Require write arguments to be grounded in user input or successful tool
  observations; errors enter a read-only recovery phase.
- Use a compact, private structured plan during discovery so a small model can
  decide whether it must gather evidence or can safely formulate a proposal.
- Stop safely at a configured turn limit and expose each step for tracing.

### Later phases

1. Implement the τ² `HalfDuplexAgent` adapter and validate it on `mock`.
2. Run a frozen Qwen3-1.7B baseline on a small Retail development subset.
3. Add trajectory logging, evaluation reports, and failure classification.
4. Treat a complete Retail interaction as one rollout and extend
   `mini_grpo` with trajectory GRPO.

## Design boundaries

The model may choose a response or a tool call. It does **not** directly own
business authority: the environment/workflow owns schema validation,
permissions, policy enforcement, execution, and audit logs. The agent core
therefore avoids guessing tool results or repairing malformed calls into a
different semantic action.

## Agent variants and experiment protocol

`baselines/pure_qwen_v0.json` freezes the historical all-tools Qwen3-4B result:
the first five official Retail train tasks scored `0/5`, with the exact model,
runtime, evaluator, and decoding configuration recorded. Do not overwrite that
file. The primary experiment model is Qwen3-1.7B; retain Qwen3-4B only for a
later migration check of the best 1.7B agent workflow.

`baselines/qwen3_1.7b_paov_v0.json` freezes the first diverse eight-task
Qwen3-1.7B development result. Its reported τ² success rate was `1/8`, but
its conservative behavioral-success rate was `0/8` because the reward-success
trajectory still contained tool errors. This distinction is retained in all
new reports.

`baselines/qwen3_1.7b_paov_v1.json` freezes the same eight tasks after the
evidence and recovery controls were added. It retains the same `1/8` reported
reward and `0/8` behavioral success, but lowers environment tool-result errors
from 29 to 16. This is a safety-control comparison, not evidence of better
task completion.

The default agent is generic `PAOV-v2`, not a task-specific script:

1. In normal discovery, Qwen first returns a private JSON plan containing
   `mode: read|propose` and `missing_facts`. Invalid JSON fails closed to
   `read`; no plan text is shown to the user.
2. Tool schemas and τ² metadata are converted into read/write/think/generic
   specs.
3. `read` exposes every read-only schema; `propose` uses schema retrieval.
   After a successful read, evidence collection also exposes every read-only
   schema to avoid lexical Top-K omissions.
4. Writes require user confirmation and arguments whose values appear in the
   user request or a successful observation.
5. A write consumes its confirmation. Successful writes enter Verify; failed
   reads or writes enter Recover, where only read tools are exposed.
6. Verify and Recover fail closed when no read tool exists; they never fall
   back to a state-changing tool.

The workflow never contains task IDs, product names, order formats, or desired
benchmark actions. This makes it suitable for comparisons across Retail task
types and later trajectory GRPO.

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
hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B

# Then run with the local checkpoint.
PYTHONPATH=src python scripts/run_qwen_mock.py --model models/Qwen3-1.7B
```

The backend uses CUDA when available and otherwise moves to MPS when PyTorch
exposes it; CPU remains a functional but slow fallback. Qwen3-1.7B is the
default because it enables faster local iteration. Keep decoding limits fixed
when comparing agent variants.

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
  --model models/Qwen3-1.7B \
  --user-llm deepseek/deepseek-chat \
  --judge-llm deepseek/deepseek-chat \
  --candidate-tools 4
```

`--user-llm` simulates the customer. `--judge-llm` evaluates tasks that contain
natural-language assertions; τ² otherwise defaults that independent evaluator
to OpenAI.

Each task's sanitized τ² simulation, raw Qwen completions, reward, duration,
tool-result errors, workflow plans, observations, and a `failure_report.json`
are written to a timestamped directory below `artifacts/retail_baseline/`.

PAOV-v2 structured planning is enabled by default. To run the exact PAOV-v1
ablation for a controlled comparison, add `--disable-structured-planning`.
Each artifact records private `workflow.structured_plans` in addition to
action plans and observations; planner output is never shown to the user.

Analyze any completed run across all of its task types:

```bash
PYTHONPATH=src python scripts/analyze_retail_failures.py \
  artifacts/retail_baseline/<timestamp>
```

The report has non-exclusive labels such as generation stop, no tool action,
tool execution error, environment-goal failure, and communication failure. It
also distinguishes `benchmark_reward_success` from conservative
`behavioral_success` so a reward with tool errors is not treated as a clean
agent completion.

For workflow development, avoid repeatedly tuning only the leading tasks. This
selects a reproducible train subset that maximizes coverage of distinct τ²
evaluation action names:

```bash
PYTHONPATH=src python scripts/run_retail_baseline.py \
  --split train --selection diverse --limit 8 --dry-run
```

Pass the displayed IDs back with `--task-ids id1,id2,...` to freeze a specific
mixed task set for an experiment.

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
