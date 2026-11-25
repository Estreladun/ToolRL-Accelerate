# ToolRL-Accelerate

面向大语言模型工具调用任务的后训练框架，支持原始数据处理、LoRA SFT、DPO、GRPO、结构感知奖励（SAR）、难度感知采样（DAS）及双卡训练。

## 核心模块

- SAR：严格校验输出格式、JSON 和参数匹配，并对同一回答记录双奖励评分。
- DAS：基于难度分桶及有效奖励组的 EMA 动态采样，无额外 Rollout 预算；双卡共享全局采样状态。
- CUDA 加速：NCCL DDP、BF16、SDPA（可选 FlashAttention-2）、融合 AdamW、长度分桶、跨 Prompt 批量 Rollout、完成段 logits、批量 Log Probability、梯度累积、Token 缓存。

```text
configs/               # 数据、模型及实验配置
scripts/               # 数据准备、训练入口与 GPU 检查
toolrl/data/           # 原始数据处理与验证
toolrl/rewards/        # 基础奖励及 SAR
toolrl/sampling/       # 均匀采样与 DAS
toolrl/trainers/       # Rollout、分布式训练
toolrl/train.py        # Mac 单设备训练
toolrl/pipeline.py     # 实验编排
tests/                 # 单元测试及双进程 CPU 测试
```

## 环境与数据

使用 Python 3.11、uv。首次运行需要联网；xLAM 需要在 Hugging Face 获得访问权限。

```bash
uv sync --python 3.11
uv run python -m pytest -q tests/
export HF_HOME="$PWD/models"
uv run huggingface-cli login
bash scripts/prepare_data.sh
uv run python -m toolrl.data.validate
```

若已有 `data/originals/`，执行 `bash scripts/prepare_data.sh --local` 即可使用原始 JSON。若已有校验通过的 `data/processed/`，无需重新处理。处理结果包括 `train.parquet`、`test.parquet`、`sft.json` 和 `manifest.json`。数据及模型的版本由 `configs/raw_sources.lock.json` 和 `configs/resources.json` 锁定。

固定数据版本的本地验证结果：ToolACE 11,300 条原始对话，xLAM 60,000 条原始记录。转换后质量筛选得到 69,407 条候选样本，去除 135 条 SFT 不适用记录后最终为：

| 数据集 | 数量 |
| --- | ---: |
| SFT | 13,881 |
| RL Train | 48,450 |
| Test | 6,941 |
| 合计 | 69,272 |

分组及规范化用户请求跨集合重叠均为 0；不保证消除全部语义近似重复。SFT 使用确定性格式监督，不包含教师推理轨迹。保守 Schema 筛选为默认策略。

## Mac 训练（原入口保留）

```bash
bash scripts/run_experiment.sh mac_smoke
bash scripts/run_experiment.sh mac128
```

## AutoDL 双卡训练

推荐 2 张 A100 PCIe 40GB、支持 BF16 的 CUDA PyTorch、至少 64GB 主机内存及充足数据盘空间。建议 200GB 以上数据盘。将项目及缓存放在持久化数据盘。使用 `nvidia-smi` 确认驱动与 PyTorch CUDA 运行时兼容。

```bash
nvidia-smi
nvidia-smi topo -m
uv sync --python 3.11
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync torchrun --standalone --nproc_per_node=2 scripts/check_dual_gpu.py

# 真实 3B 双卡冒烟测试
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_cloud.sh cloud_smoke

# 正式训练
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_cloud.sh cloud3b 2>&1 | tee logs/cloud3b.log
```

`cloud3b` 使用固定版本 Qwen2.5-3B-Instruct、LoRA rank 8、BF16、每卡 2 个 GRPO Prompt、每 Prompt 5 个完成，梯度累积 8 次（32 个 Prompt / 更新）。GRPO 学习率 1e-6、温度 0.7、KL 0.001、无熵奖励，训练 7 epoch。SFT 学习率 1e-4，5 epoch，余弦调度和 10% warmup。DPO 为本项目扩展，学习率 5e-6，1 epoch。

| 参数 | cloud3b |
| --- | ---: |
| SFT 样本 | 5,518 |
| DPO 候选 | 2,048 |
| GRPO 候选池 | 6,144 |
| GRPO 每 epoch Prompt 预算（双卡合计） | 2,048 |
| GRPO 总 Prompt 预算 / 分支 | 14,336 |
| Validation / Test | 512 / 1,024 |
| 最大 Prompt / 生成长度 | 4,096 / 1,024 |

学习率、温度、KL、Rollout 数和 epoch 参考原论文，批次、长度、数据预算和 LoRA 是双卡适配设置，不能宣称与原论文完整训练或最优效果等价。每个 RL 分支生成 71,680 个回答，三个分支合计 215,040 个回答，不含评测和 DPO 样本生成；正式任务可能耗时较长。配置位于 `configs/cloud3b.env`。

默认使用 PyTorch SDPA，不强制安装第三方 CUDA 扩展。安装与当前 CUDA/PyTorch 兼容的 FlashAttention-2 后可通过 `ATTENTION=flash_attention_2` 启用。该模式需在服务器验证后再用于正式实验。目前使用 DDP，不包含 DeepSpeed 或 vLLM；对 3B LoRA，DDP 复制冻结基座且仅同步可训练参数的梯度，避免额外 ZeRO 分片通信。真实 CUDA 训练吞吐量、显存占用及加速效果尚待实测。

## 实验与结果

| 实验 | 训练方式 |
| --- | --- |
| E0 | Base Model |
| E1 | LoRA SFT |
| E2 | LoRA SFT + DPO |
| E3 | GRPO Binary Reward |
| E4 | GRPO + SAR |
| E5 | GRPO + SAR + DAS |

E3 至 E5 从同一个验证集选定的检查点出发，使用相同候选池、生成参数与每 epoch 全局组预算。DAS 在每 epoch 内无放回抽样，跨 epoch 可重复。每张卡负责完整 GRPO 组，在下一轮采样前同步奖励统计。SFT、DPO、GRPO、评测均启用双卡；rank 0 独占模型保存和汇总。每个累积窗口只做一次 on-policy 更新，旧策略概率由当前概率分离梯度获得；不支持多轮 PPO 更新。

```bash
cat runs/cloud3b/report/summary.csv
cat runs/cloud3b/E5/training.json
```

保留准确率、奖励审计、有效组比例、耗时和生成吞吐量等记录，无需额外训练性能消融。

## 注意事项

- xLAM 出现 403 时检查网页访问授权、Token 和 `HF_HOME`，不要绕过数据集访问限制。
- 同一实验配置可通过原命令进行阶段级恢复，未完成阶段将重新运行；暂不支持阶段内部训练步恢复。更新代码或配置后使用新 `RUN`，避免与旧结果混用。
- CUDA 内存不足时优先降低 `LOSS_BATCH`、`TRAIN_BATCH`、`DPO_BATCH`、`EVAL_BATCH`，或 `ROLLOUT_BATCH`；降低 `RL_PROMPTS` 时确保 `GRPO` 可被 `NPROC*RL_PROMPTS` 整除。变更配置需设置新 `RUN`。GRPO 有效全局 Prompt 批次为 `NPROC*RL_PROMPTS*RL_ACCUM`。
- 不必复制 Mac `.venv` 到 Linux。首次在服务器执行 `uv sync`，若有已验证的 `uv.lock`，使用 `uv sync --locked`。代码包未包含 uv.lock。
- 数据、权重、缓存、训练结果和账号凭证均不上传 GitHub，提交源码、配置、脚本、测试、README、LICENSE、pyproject.toml 和生成的 uv.lock。

## 参考资料与许可

数据转换规范、Prompt 及基础二元奖励参考 [Tool-N1](https://github.com/NVlabs/Tool-N1)（[论文](https://arxiv.org/abs/2505.00024v2)，附录 B）；采样思想参考 [DAPO](https://arxiv.org/abs/2503.14476)。数据来自 [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) 和 [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)。默认模型 [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) 使用 Qwen Research License，使用模型与数据时遵循各自条款。代码遵循 Apache 2.0，复用部分保留必要版权声明。当前结果仅限项目内部评测，不等同于论文官方 Benchmark。
