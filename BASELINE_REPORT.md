# Qwen2.5-0.5B GRPO 算术实验报告

## 实验目的

记录 `Qwen/Qwen2.5-0.5B-Instruct` 在固定算术测试集上的训练前表现、GRPO 训练过程与最终结果。

## 数据划分

| 划分 | 文件 | 数量 | 用途 |
|---|---|---:|---|
| 训练集 | `data/arithmetic/train.jsonl` | 5,000 | GRPO 采样、奖励和参数更新 |
| 验证集 | `data/arithmetic/validation.jsonl` | 500 | 调整训练配置与奖励函数 |
| 测试集 | `data/arithmetic/test.jsonl` | 500 | 固定的训练前/训练后最终对比 |

三个划分按运算及操作数去重；对乘法也排除了交换操作数的等价题，例如训练集的 `5 × 350` 不会在验证集或测试集中以 `350 × 5` 出现。

## 评测设置

| 项目 | 值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 数据 | `data/arithmetic/test.jsonl` |
| 任务 | 1–3 位正整数乘法 |
| 提示词要求 | 仅输出 `<answer>整数</answer>` |
| 解码 | 贪心解码（`do_sample=false`） |
| 最大生成 token | 32 |
| 硬件后端 | Apple MPS |

正确性规则：从模型输出中提取 `<answer>...</answer>` 内的整数，与测试集的标准答案精确比较。未匹配该格式的输出记为格式不合规，且不能计为正确。

## 训练前 Baseline

| 指标 | 结果 |
|---|---:|
| 测试题数 | 500 |
| 正确题数 | 95 |
| 测试准确率 | **19.0%** |
| 格式合规题数 | 468 |
| 格式合规率 | **93.6%** |

原始逐题输出保存在 `outputs/test_baseline_results.jsonl`，机器可读汇总保存在 `outputs/test_baseline_summary.json`。

## GRPO 训练配置

| 项目 | 值 |
|---|---:|
| 训练数据 | `data/arithmetic/train.jsonl`（5,000 条） |
| 训练步数 | 1,250 |
| 每题候选数 | 4 |
| LoRA | rank 8，alpha 16 |
| 学习率 | `2e-5`，warmup 50 步 |
| KL 系数 | 0.04 |
| 最大生成长度 | 32 tokens |
| Checkpoint | 每 250 步保存 |
| 运行时间 | 约 46.5 分钟（Apple MPS） |

奖励规则：模型输出中恰好存在一个 `<answer>整数</answer>` 标签，且整数与标准答案一致时奖励为 1；其他情况奖励为 0。

训练过程稳定，没有出现 NaN 或发散。平均 KL 约为 0.055。约 31.8% 的生成组具有非零组内奖励方差，能提供 GRPO 的相对学习信号；其余较难题目经常出现候选答案全错的情况。

## Checkpoint 选择

所有 checkpoint 仅在 validation 集上评测，按准确率选择最终模型：

| Checkpoint | Validation 准确率 | 格式合规率 |
|---|---:|---:|
| **250（选中）** | **38.8%** | 98.8% |
| 750 | 38.2% | 99.0% |
| 1250 | 37.4% | 99.2% |
| 1000 | 35.6% | 98.8% |
| 500 | 34.6% | 99.0% |

完整排名见 `outputs/grpo_formal_v1/checkpoint_validation.json`。validation 准确率在 250 步达到峰值，之后轻微回落，因此最终采用 `checkpoint-250`。

## 最终测试结果

冻结 test 集只在选定 checkpoint 后评测一次。

| 指标 | 训练前 Baseline | GRPO checkpoint-250 | 变化 |
|---|---:|---:|---:|
| Test 准确率 | 19.0%（95/500） | **34.8%（174/500）** | **+15.8 个百分点** |
| 格式合规率 | 93.6% | **98.2%** | +4.6 个百分点 |

GRPO 将测试准确率从 19.0% 提升至 34.8%，说明该奖励设计与训练流程对该自动判分算术任务有效。选中 checkpoint 的 validation 准确率为 38.8%，test 准确率为 34.8%，存在合理的泛化落差，但没有出现只在 validation 集提升的现象。

最终结果见 `outputs/grpo_formal_v1/final_test_evaluation.json`；选中的 LoRA adapter 位于 `outputs/grpo_formal_v1/checkpoint-250/`。

## 实验协议与后续工作

1. GRPO 训练阶段只使用训练集；超参数、奖励函数和训练轮数的选择只参考验证集。
2. 测试集保持冻结，不用于后续调参或数据难度选择；本实验已完成其唯一一次最终评测。
3. 本轮训练在 250 步的 validation 表现最好；后续实验可降低生成结束后的无效 token，或调整训练时长与每题候选数，以提高有效奖励组比例。
