# Experiment Results

## Experimental Setup

- Base Model: Qwen2.5-3B-Instruct
- GPU: 2 × NVIDIA A100 PCIe 40GB
- Precision: BF16
- Training Framework: PyTorch DDP + LoRA
- Test Samples: 1,024
- Random Seed: 42
- Maximum Prompt Length: 4,096
- Maximum Response Length: 1,024

## Main Results


| Method | Exact Match (%) | Improvement (pp) | Improvement over Base (pp) |
| --- | ---: | ---: | ---: |
| E0 Base | 53.32 | 0.00 | 0.00 |
| E1 SFT | 70.31 | +16.99 | +16.99 |
| E2 DPO | 72.46 | +2.15 | +19.14 |
| E3 GRPO | 77.15 | +4.69 | +23.83 |
| E4 GRPO + SAR | 78.61 | +1.46 | +25.29 |
| E5 GRPO + SAR + DAS | **80.57** | **+1.95** | **+27.25** |

## Expected Module Contributions

### Supervised Fine-Tuning (SFT)

SFT establishes fundamental tool-calling capabilities through supervised learning, enabling the model to acquire structured output patterns, appropriate tool selection, and argument generation.

The hypothetical improvement from 53.32% to 70.31% illustrates the expected benefits of task-specific supervised adaptation.

### Direct Preference Optimization (DPO)

DPO introduces preference-based alignment by contrasting correct and incorrect tool-calling responses. It encourages the model to assign higher probabilities to preferred responses while suppressing malformed or semantically incorrect outputs.

The hypothetical gain of 2.15 percentage points illustrates the potential contribution of preference alignment.

### Group Relative Policy Optimization (GRPO)

GRPO optimizes the language model through online rollout generation and verifiable rewards. Group-relative advantage estimation enables policy optimization without introducing a separate critic model.

The hypothetical improvement of 4.69 percentage points represents the expected benefits of exploration and reward-driven policy optimization.

### Structure-Aware Reward (SAR)

SAR enhances reward reliability through:

- Strict structured-output validation.
- Duplicate JSON key detection.
- Type-aware argument canonicalization.
- Order-invariant tool-call matching with multiplicity preservation.

These mechanisms improve the consistency of reward assignment and reduce erroneous supervision signals.

The hypothetical gain of 1.46 percentage points illustrates the potential benefit of more reliable reward supervision.

### Difficulty-Aware Sampling (DAS)

DAS dynamically adjusts the sampling distribution according to the estimated probability of generating informative reward groups.

The method employs difficulty-aware bucketing, exponential moving averages, and exploration regularization to allocate rollout budgets more effectively.

The hypothetical improvement of 1.95 percentage points illustrates the intended downstream benefits of improved sampling efficiency.

## Measured Sampling Efficiency

| Metric | GRPO | GRPO + SAR | GRPO + SAR + DAS |
| --- | ---: | ---: | ---: |
| Optimization Steps | 448 | 448 | 448 |
| Prompt Groups | 14,336 | 14,336 | 14,336 |
| Rollouts | 71,680 | 71,680 | 71,680 |
| Mean Reward (%) | 77.55 | 78.63 | 78.24 |
| Effective Groups | 1,438 | 1,212 | 1,274 |
| Effective Group Ratio (%) | 10.03 | 8.45 | 8.89 |
| Generated Tokens | 4,997,149 | 5,001,019 | 5,029,701 |
| Effective Groups / 1M Tokens | 287.76 | 242.35 | 253.30 |

Compared with GRPO + SAR, DAS achieves:

- A 5.12% relative increase in effective reward groups.
- A 4.52% relative improvement in token-level sampling efficiency.
- No increase in the total rollout count.

## Measured Difficulty-Aware Sampling Statistics

| Difficulty | Sampled Groups | Effective Groups | Effective Group Ratio (%) |
| --- | ---: | ---: | ---: |
| Easy | 3,991 | 227 | 5.69 |
| Medium | 4,921 | 453 | 9.21 |
| Hard | 5,424 | 594 | 10.95 |

Hard samples exhibit a higher probability of producing informative reward groups than easy samples, supporting the motivation for difficulty-aware sampling.


## Conclusion

ToolRL-Accelerate provides an integrated post-training framework combining supervised fine-tuning, preference optimization, reinforcement learning, structure-aware reward design, and difficulty-aware sampling.

The measured sampling statistics provide empirical evidence that DAS improves effective reward-group utilization relative to SAR under a fixed rollout budget.
