# Tool-Agent GRPO Research

This repository studies a local Qwen tool agent on the text-based Retail domain
of τ²-bench, and later trains it with trajectory-level GRPO.

## Structure

```text
.
├── tau2_retail_agent/  # Qwen3-4B tool agent, τ² integration, evaluation, traces
└── mini_grpo/          # Small, readable GRPO implementation to extend for trajectories
```

## Development order

1. Run the agent core tests in `tau2_retail_agent/`.
2. Integrate the agent with τ² `mock`, then establish a frozen Retail baseline.
3. Record trajectories, metrics, and failure categories.
4. Extend `mini_grpo/` to optimize complete Retail trajectories using GRPO.

The previous arithmetic SFT/GRPO experiment, including its data, checkpoints,
and single-turn reward code, has been deliberately removed. `mini_grpo/` is
retained as the learning implementation that will be adapted after the agent
baseline is stable.
