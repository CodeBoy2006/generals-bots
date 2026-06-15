# Expander 对抗训练流程与策略

本文记录 8x8 generated 地图上训练 PPO 策略超过 Expander 的可复用流程。它不是一次性 devlog，而是后续继续训练、复现实验和判断模型强度时的操作指南。

当前已验证的 checkpoint：

```text
/tmp/generals-ppo-8x8-expander-gpu-v5.eqx
```

该 checkpoint 在 sampled policy 模式下，对 randomized Expander 的独立 2048 局评估超过 90% 总胜率。`.eqx` 是实验产物，应保存在 `/tmp` 或实验目录，不提交进 Git。

## 目标与判定标准

训练目标要用独立评估确认，不能只看训练过程中的 rollout 胜率。

推荐验收标准：

- 地图：8x8 generated
- mountain density：0.12-0.22
- cities：4-8
- minimum general distance：5
- max steps：500
- opponent：`expander`
- policy mode：`sample`
- 每个评估至少 2048 局
- 同一 checkpoint 至少测两个独立 seed
- 同一 seed 分别测 `--policy-player 0` 和 `--policy-player 1`
- 目标胜率按总局数计算，draw 不是 win

不要只报告 decisive win rate。decisive win rate 可以辅助分析，但如果 draw 很多，总胜率仍然不足。

## 环境准备

安装开发依赖和 CUDA extra：

```bash
uv sync --extra dev --extra cuda13
```

确认 JAX 使用 GPU：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

期望输出包含：

```text
gpu
[CudaDevice(id=0)]
```

如果 GPU 不可用，可以用 CPU 做 smoke test，但不适合长时间策略训练。

## 训练路线

最终有效路线分为三段：

1. 行为克隆 warm start：让策略先学会 Expander 风格的基础扩张动作。
2. PPO probe：从 BC checkpoint 对 Expander 进行短 PPO，确认强化学习方向能提升胜率。
3. GPU fine-tune：从较强 PPO checkpoint 继续训练，使用大 reset pool、低学习率和多 epoch/minibatch 更新。

这条路线的关键判断是：BC 只提供起点，真正超过 Expander 来自后续 PPO-vs-Expander fine-tune。

## 阶段一：行为克隆 warm start

从 soft Expander teacher 训练一个可用起点：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --map-generator generated \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 1000 \
  --lr 0.0007 \
  --model-path /tmp/generals-bc-8x8-soft.eqx \
  --seed 46
```

经验结果：

- 浅层 BC 会很弱，对 Expander 可能只有个位数胜率。
- 1000-2000 iterations 后，通常能得到足够好的 PPO 起点。
- BC 模型可能对 Random 很强，但这不代表它强于 Expander。

如果训练中断，可以续训：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 1000 \
  --lr 0.0007 \
  --init-model-path /tmp/generals-bc-8x8-soft.eqx \
  --model-path /tmp/generals-bc-8x8-soft-v2.eqx
```

## 阶段二：PPO probe

先用较小训练量确认 PPO-vs-Expander 会提升，而不是破坏 BC 策略：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 256 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --pool-size 4096 \
  --num-steps 64 \
  --num-iterations 300 \
  --num-epochs 4 \
  --minibatch-size 2048 \
  --lr 0.00005 \
  --truncation 500 \
  --opponent expander \
  --init-model-path /tmp/generals-bc-8x8-soft.eqx \
  --model-path /tmp/generals-ppo-8x8-expander-probe.eqx \
  --seed 9101
```

训练日志中的 episode 胜率只用于观察趋势。保存 checkpoint 后必须独立评估。

## 阶段三：GPU fine-tune

接近 90% 时，继续使用更大的 reset pool 和更低学习率，减少固定地图池过拟合：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 512 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --pool-size 16384 \
  --num-steps 64 \
  --num-iterations 700 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --truncation 500 \
  --opponent expander \
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v4.eqx \
  --model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --seed 9104
```

有效的 fine-tune 特征：

- rollout 内胜率大多在 88-93% 区间波动。
- draw rate 下降，平均终局时间缩短。
- 独立评估中的 loss 数继续下降。

如果 rollout 胜率长期停在 85-89%，继续堆同一 PPO 配方收益会降低，应考虑调整奖励、对手课程或引入更强 teacher。

## 阶段四：checkpoint 与 current-policy 自博弈

当前训练入口支持两种 self-play：

- frozen checkpoint self-play：learner 从 `--init-model-path` 加载并继续更新，非 learner 玩家由 `--opponent-policy-path` 指定的冻结 checkpoint 控制。
- current-policy self-play：传 `--self-play-opponent` 后，非 learner 玩家在每轮 rollout 中使用当前正在更新的同一个 policy。

frozen opponent 更适合作为稳定 best-response 训练入口；current-policy opponent 更接近同步自博弈，但 PPO 更新仍只使用 `--learner-player` 指定座位的数据。

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 512 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --pool-size 16384 \
  --num-steps 64 \
  --num-iterations 300 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --truncation 500 \
  --init-model-path /tmp/generals-ppo-current.eqx \
  --opponent-policy-path /tmp/generals-ppo-best-frozen.eqx \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 1.0 \
  --model-path /tmp/generals-ppo-selfplay-next.eqx \
  --seed 9201
```

current-policy self-play 命令：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 256 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --pool-size 8192 \
  --num-steps 64 \
  --num-iterations 160 \
  --num-epochs 2 \
  --minibatch-size 4096 \
  --lr 0.000002 \
  --truncation 500 \
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --self-play-opponent \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 1.0 \
  --model-path /tmp/generals-ppo-8x8-current-selfplay.eqx \
  --seed 26010
```

新增参数：

- `--self-play-opponent`：让非 learner 玩家使用当前 learner policy；不能和 `--opponent-policy-path` 同时使用。
- `--learner-player 0|1`：选择 learner 控制环境中的哪个玩家槽位。用它可以分别训练先手/后手视角，避免只优化 player 0。
- `--terminal-reward-scale N`：在 decisive terminal transition 上给 learner 胜局 `+N`、败局 `-N`。默认 `0.0`，保持旧 composite reward 行为。

使用建议：

- 先固定一个 frozen opponent，确认 learner 视角、终局奖励和评估基线都稳定，再尝试 current-policy opponent。
- 每次 self-play 后都要重新测 Expander、其它 heuristic、历史 best checkpoint 和镜像座位。
- 如果新模型打赢历史模型但对 Expander 或 mixed heuristic 退化，不应替换 best checkpoint。
- 后续可以把多个历史 checkpoint 做成 league opponent，避免只针对一个 frozen/current opponent 过拟合。

### 当前 v5 自博弈结果

以 `/tmp/generals-ppo-8x8-expander-gpu-v5.eqx` 为 current checkpoint，sample-vs-sample 自身基线在 2048 局独立评估中接近 50% decisive：

```text
v5 as player 0 vs v5 sample:
  wins/losses/draws = 948/893/207
  win rate = 46.29%
  decisive win rate = 51.49%

v5 as player 1 vs v5 sample:
  wins/losses/draws = 911/919/218
  win rate = 44.48%
  decisive win rate = 49.78%
```

第一轮 frozen self-play（v5 warm start, opponent=v5 sample, learner=player 0, 700 iterations, `lr=5e-6`）只得到小幅提升：

```text
/tmp/generals-ppo-8x8-selfplay-v1.eqx as player 0 vs v5 sample:
  wins/losses/draws = 1003/828/217
  win rate = 48.97%
  decisive win rate = 54.78%

as player 1 vs v5 sample:
  wins/losses/draws = 913/913/222
  win rate = 44.58%
  decisive win rate = 50.00%
```

提高终局奖励、增大学习率、长 rollout 或切换 v5 greedy 对手，均未出现接近 80% 的趋势。当前结论：在现有 42k 参数网络和 PPO objective 下，直接 frozen self-play 更适合做小幅 fine-tune，不足以快速学出压倒性 best response。

新增 current-policy self-play 能力后，用 v5 warm start 做 160 iterations 短试验也没有产生提升：

```text
/tmp/generals-ppo-8x8-current-selfplay-v1.eqx, player 0, seed 26020:
  candidate wins/losses/draws = 435/495/94
  same-seed v5 baseline       = 448/477/99

player 1, seed 26021:
  candidate wins/losses/draws = 438/486/100
  same-seed v5 baseline       = 449/477/98
```

结论：current-policy self-play 已经是可用训练模式，但这组参数没有学出对 v5 的 best response。若继续 self-play 路线，应优先尝试 checkpoint league、历史池采样、对手建模或更强 value target，而不是只把同一个 PPO policy 同步对打更久。

## 阶段五：胜者轨迹辅助克隆

`examples/_experimental/ppo/outcome_clone.py` 是一个 outcome-conditioned auxiliary trainer。它完整 rollout policy-vs-policy 对局，然后把最终胜者视角的动作作为监督样本训练同一个 `PolicyValueNetwork`。

基础命令：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/outcome_clone.py 256 \
  --num-steps 500 \
  --num-iterations 100 \
  --num-epochs 1 \
  --minibatch-size 8192 \
  --lr 0.00001 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --init-model-path /tmp/generals-ppo-current.eqx \
  --opponent-policy-path /tmp/generals-ppo-best-frozen.eqx \
  --policy-mode sample \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --winner-source both \
  --negative-weight 0.0 \
  --model-path /tmp/generals-ppo-outcome-clone.eqx \
  --seed 9701
```

关键参数：

- `--winner-source both`：胜者来自任一玩家；这最像从 self-play winner trajectories 蒸馏。
- `--winner-source learner`：只克隆 learner 赢局；选择压力更强，但当前实验中容易退化。
- `--negative-weight`：可选对比项，降低最终败者实际动作的概率。当前实验中 `0.2` 会快速压低与 v5 的相似度，但没有提升胜率，需谨慎使用。

当前实证结果：

```text
both winner cloning, 80 iterations:
  /tmp/generals-ppo-8x8-outcome-v4-p0.eqx as player 0 vs v5 sample
  wins/losses/draws = 1001/839/208
  win rate = 48.88%
  decisive win rate = 54.40%

learner-only winner cloning, 200 iterations:
  /tmp/generals-ppo-8x8-outcome-v5-learner-p0.eqx as player 0 vs v5 sample
  wins/losses/draws = 863/934/251
  win rate = 42.14%
  decisive win rate = 48.02%
```

结论：胜者轨迹克隆提供了可复用的长时序辅助训练能力，但单独使用仍没有让 v5-vs-v5 从约 50% 拉到 80%。下一步更可能需要 league/self-play population、显式 opponent modeling、搜索 teacher，或扩大网络容量，而不是继续微调同一个小网络的最后几层。

## 阶段六：rollout-search 强辅助策略

`examples/_experimental/ppo/search_policy.py` 将 checkpoint 当作 policy prior，对当前局面的 top-k 候选动作做短 rollout 评分，再选择期望更高的动作。它不产生新的 `.eqx` checkpoint，但可以作为强评估策略和后续蒸馏 teacher。

推荐的强辅助配置：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/search_policy.py /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --num-games 512 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500 \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --opponent-policy-mode sample \
  --search-player 0 \
  --top-k 4 \
  --rollout-steps 16 \
  --rollouts-per-action 4 \
  --seed 19192
```

当前验证结果：

```text
rollout-search as player 0 vs v5 sample, seed 19192:
  wins/losses/draws = 454/46/12
  win rate = 88.67%
  decisive win rate = 90.80%

rollout-search as player 1 vs v5 sample, seed 19193:
  wins/losses/draws = 449/47/16
  win rate = 87.70%
  decisive win rate = 90.52%
```

蒸馏尝试：

```text
/tmp/generals-ppo-8x8-rollout-search-distill-v1.eqx as player 0 vs v5 sample:
  wins/losses/draws = 912/918/218
  win rate = 44.53%
  decisive win rate = 49.84%
```

结论：rollout-search 已经让“v5 + 强辅助推理”稳定超过当前 v5 checkpoint 的 80% 总胜率，但目前还没有成功把该行为压缩回现有 42k 参数 checkpoint。继续训练纯 checkpoint 时，应把 search policy 作为 teacher，同时考虑更大网络、更多输入通道、DAgger 数据混合或训练时保留 search distillation 的 KL/temperature 控制。

### 保守 rollout-search 蒸馏

`examples/_experimental/ppo/conservative_search_distill.py` 是当前推荐的 search-teacher 训练入口。它与直接交叉熵蒸馏不同：

- 固定 `--base-model-path` 作为 rollout-search teacher 和 KL anchor。
- 学生从 `--init-model-path` warm start；省略时默认从 base checkpoint 开始。
- 每个学生状态只对 base policy top-k 候选动作做短 rollout 评分。
- 只有当 search 最优动作不是 base 的 top-prior 动作，且分数差超过 `--min-margin` 时，才加入动作监督。
- `--target-mode hard` 的总 loss 为 `kl_weight * KL(base || student) + improve_weight * weighted CE(search_action)`。
- `--target-mode soft` 会把 top-k search 分数转为候选动作上的软目标，避免把大量小 margin 候选强制压成单标签。
- `--policy-input full-state` 会让学生使用 privileged 完整状态编码；此模式不等同于标准 fogged observation policy，评估时也必须传 `evaluate_policy.py --policy-input full-state`。
- `--policy-input augmented-full-state` 会保留原 9 个 fogged observation 通道，并追加 9 个 privileged full-state 通道；默认输入通道数为 18。

基础命令：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/conservative_search_distill.py 128 \
  --base-model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --target-mode soft \
  --num-steps 64 \
  --num-iterations 80 \
  --minibatch-size 8192 \
  --min-margin 1 \
  --margin-scale 4 \
  --improve-weight 0.02 \
  --kl-weight 1.0 \
  --lr 0.000001 \
  --model-path /tmp/generals-ppo-8x8-conservative-search.eqx \
  --seed 23020
```

本轮实测结论：

```text
/tmp/generals-ppo-8x8-conservative-search-v2.eqx, player 0, seed 23120:
  candidate wins/losses/draws = 444/472/108
  same-seed v5 baseline       = 445/484/95

/tmp/generals-ppo-8x8-conservative-search-v2.eqx, player 1, seed 23121:
  candidate wins/losses/draws = 436/491/97
  same-seed v5 baseline       = 422/499/103
```

expanded-64 学生加更强 KL 也没有产生显著提升：

```text
/tmp/generals-ppo-8x8-expanded64-conservative-search-v1.eqx, player 0, seed 23130:
  candidate wins/losses/draws = 442/485/97
  same-seed v5 baseline       = 439/490/95

player 1, seed 23131:
  candidate wins/losses/draws = 459/471/94
  same-seed v5 baseline       = 468/462/94
```

因此，保守蒸馏能力已经可复用，但当前结果仍只是“接近保持 v5”，没有把 rollout-search 的 80%+ 胜率压缩进纯 `.eqx` checkpoint。下一步更有希望的方向是训练显式 Q/value-improvement head、在网络输入中加入 rollout/search 特征，或把 search 保留为评测/实战时的规划模块，而不是继续只做动作分类蒸馏。

#### soft target 与 full-state 探测

对 16,337 个 active 样本的 top-k search 分数做探测时，search 最优动作有 60.8% 不等于 base top-prior 动作，但大多数 margin 很小：

```text
margin vs base, all samples:
  p50 = 0.048
  p75 = 0.201
  p95 = 1.004
  p99 = 254.649

switched action fraction = 60.8%
```

这解释了为什么硬 argmax 蒸馏容易退化：大量标签来自近似并列候选，单标签 CE 会放大 rollout 噪声。soft target 蒸馏避免了这个问题，但默认 observation 输入仍没有产生显著提升：

```text
/tmp/generals-ppo-8x8-soft-search-v1.eqx, player 0, seed 24120:
  candidate wins/losses/draws = 478/458/88
  same-seed v5 baseline       = 475/450/99

player 1, seed 24121:
  candidate wins/losses/draws = 465/452/107
  same-seed v5 baseline       = 479/445/100
```

进一步检查发现，`search_policy.py` 的 rollout 评分推进完整 `GameState`，而普通 policy checkpoint 只接收 fogged `Observation`。直接把 v5 checkpoint 接到替换式 full-state 9 通道编码上会更弱：

```text
full-state v5 wrapper vs v5 sample, player 0:
  wins/losses/draws = 418/512/94
  win rate = 40.82%
```

修正 KL 后的 full-state soft-search 训练仍未提升：

```text
/tmp/generals-ppo-8x8-fullstate-soft-search-v2.eqx, player 0, seed 24430:
  candidate wins/losses/draws = 392/542/90
  same-seed v5 baseline       = 485/440/99

player 1, seed 24431:
  candidate wins/losses/draws = 410/528/86
  same-seed v5 baseline       = 482/446/96
```

当前结论：不能简单把原 9 个 observation 通道替换成 full-state 语义。下一步如果继续 privileged checkpoint 路线，应扩展输入通道，并把 v5 原始 9 通道 conv1 权重原样复制，额外 full-state/search 特征通道从 0 初始化，这样才能保留 v5 基线行为再学习隐藏信息增益。

#### augmented-full-state 输入

当前实现加入了 18 通道 augmented 输入：

```text
channels 0-8:   标准 fogged observation，与 v5 完全一致
channels 9-17:  privileged full-state 编码
```

从 9 通道 v5 checkpoint warm start 到 18 通道学生时，`load_or_create_network(..., input_channels=18, init_input_channels=9)` 会复制原始 conv1 的前 9 个输入通道权重，并把新增通道权重置 0。这样在额外通道全 0 时，logits/value 与原 checkpoint 保持一致。

soft target augmented 蒸馏基本保持 v5，但没有明显提升：

```text
/tmp/generals-ppo-8x8-augmented-soft-search-v1.eqx, player 0, seed 25110:
  candidate wins/losses/draws = 444/467/113
  same-seed v5 baseline       = 445/468/111

player 1, seed 25111:
  candidate wins/losses/draws = 455/480/89
  same-seed v5 baseline       = 460/468/96
```

hard high-margin augmented 蒸馏出现小幅波动性改善，但仍远离 80%：

```text
/tmp/generals-ppo-8x8-augmented-hard-search-v1.eqx, player 0, seed 25220:
  candidate wins/losses/draws = 443/484/97
  same-seed v5 baseline       = 436/468/120

player 1, seed 25221:
  candidate wins/losses/draws = 472/475/77
  same-seed v5 baseline       = 454/495/75
```

继续加大 improve 权重并降低 KL 的 v2 退化明显：

```text
/tmp/generals-ppo-8x8-augmented-hard-search-v2.eqx, player 0, seed 25320:
  candidate wins/losses/draws = 398/531/95
  same-seed v5 baseline       = 484/440/100

player 1, seed 25321:
  candidate wins/losses/draws = 381/548/95
  same-seed v5 baseline       = 409/519/96
```

结论：18 通道 augmented 输入解决了“替换通道语义破坏 v5”的问题，是后续 privileged/search-feature 学习的正确接口；但当前 search-action 蒸馏目标仍不足以把 80%+ rollout-search 行为压缩进纯 checkpoint。

### 容量扩展实验

训练和评估入口现在支持非默认网络容量：

```bash
--channels 64,64,64,32
--opponent-channels 32,32,32,16
```

`--channels` 描述候选 checkpoint 的四层卷积通道；`--opponent-channels` 描述冻结对手 checkpoint。默认 v5 使用 `(32, 32, 32, 16)`。

一次临时实验将 v5 权重嵌入到 `(64, 64, 64, 32)` 的更宽网络中，初始评估仍接近 v5-vs-v5 基线：

```text
/tmp/generals-ppo-8x8-v5-expanded-64.eqx as player 0 vs v5 sample:
  wins/losses/draws = 468/450/106
  win rate = 45.70%
  decisive win rate = 50.98%
```

随后用该 expanded-64 checkpoint 做 DAgger 式 rollout-search 蒸馏，学生状态由学生自身产生、标签来自固定 v5 rollout-search teacher；结果退化：

```text
/tmp/generals-ppo-8x8-expanded64-dagger-search-v1.eqx as player 0 vs v5 sample:
  wins/losses/draws = 371/1512/165
  win rate = 18.12%
  decisive win rate = 19.70%
```

这说明“简单扩宽 + search-label 交叉熵”仍不足以压缩 search teacher。后续若继续走纯 checkpoint 路线，应优先尝试混合目标：保持 v5 行为的 KL/BC 权重更强、只在高置信 search 改进样本上更新、或使用价值/优势回归而不是强制动作分类。

## 评估命令

评估 player 0：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --num-games 2048 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500 \
  --opponent expander \
  --policy-mode sample \
  --policy-player 0 \
  --seed 8501
```

镜像评估 player 1：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --num-games 2048 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500 \
  --opponent expander \
  --policy-mode sample \
  --policy-player 1 \
  --seed 8501
```

最终 v5 结果：

```text
seed 8501, policy_player=0:
  wins/losses/draws = 1854/150/44
  win rate = 90.53%

seed 8501, policy_player=1:
  wins/losses/draws = 1846/168/34
  win rate = 90.14%

seed 8503, policy_player=0:
  wins/losses/draws = 1859/155/34
  win rate = 90.77%

seed 8503, policy_player=1:
  wins/losses/draws = 1856/160/32
  win rate = 90.62%
```

同 checkpoint 对 Random 的 sanity check：

```text
seed 8504, policy_player=0:
  wins/losses/draws = 2039/2/7
  win rate = 99.56%
```

评估 candidate checkpoint 对 frozen checkpoint：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py /tmp/generals-ppo-selfplay-next.eqx \
  --opponent-policy-path /tmp/generals-ppo-best-frozen.eqx \
  --opponent-policy-mode sample \
  --num-games 2048 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500 \
  --policy-mode sample \
  --policy-player 0 \
  --seed 8601
```

## 策略层面的经验

Expander 的优势是快速占地、局部贪心扩张和稳定吞噬中立格。直接行为克隆会把这些优点学到，但也会继承它的单一路线。

PPO fine-tune 后有效提升主要来自：

- 学会在局部扩张和进攻之间更早切换。
- 减少无意义 draw，让强势局更快转化成终局。
- 在对手贪扩张时更频繁地抓住突破窗口。
- 通过 sampled policy 保留一定行动多样性，避免 greedy 在少数局面中重复选错固定动作。

当前结论应精确表述为：8x8 generated 地图、500 step、sampled policy 对 randomized Expander 超过 90% 总胜率。不要把这个结果直接外推到更大地图、真实 generals.io 人类对战或 greedy policy。

## 常见问题

### 为什么 sample 超过 90%，greedy 没有？

网络动作空间中很多局面存在多个近似可行动作。sample 模式保留探索性和随机化，能避免在少数战术局面中固定走到坏分支。当前 v5 的 greedy 总胜率仍低于 90%，所以报告时必须注明 `--policy-mode sample`。

### 为什么要镜像 `--policy-player`？

地图生成和双方出生位置可能带来微弱偏差。只测 player 0 可能高估或低估模型强度。`evaluate_policy.py --policy-player 1` 会让模型坐到另一侧，并按模型视角统计 wins/losses。

### 为什么训练内胜率不能作为验收？

训练内 episode 来自当前 reset pool 和当前 on-policy 采样，且样本量通常较小。最终验收必须用独立 seed、独立地图批次和足够局数。

### 继续提升到 greedy 90% 怎么做？

可以尝试：

- 降低 entropy 对 sample 分布的依赖，增加 greedy 友好的蒸馏阶段。
- 用 v5 自采样数据做 DAgger 式 teacher 修正。
- 增加 anti-Expander 专门 reward，例如更强的终局速度奖励和 loss 风险惩罚。
- 引入 checkpoint league，避免只针对一个 Expander 分布过拟合。
