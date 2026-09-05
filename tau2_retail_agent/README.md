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

For a real τ² run, use the benchmark's supported setup (`uv sync` from its
repository) and install the optional model dependencies in the same Python
environment. The exact Qwen checkpoint, decoding parameters, τ² version, and
user-simulator configuration must be frozen before recording a baseline.

## Current acceptance criteria

- `AgentCore` records policy, tools, user messages, assistant messages, and
  tool results in order.
- Unknown tools and invalid tool arguments fail before reaching a business
  environment.
- A turn budget prevents endless model/tool loops.
- The core can be tested without downloading a model or the benchmark.
