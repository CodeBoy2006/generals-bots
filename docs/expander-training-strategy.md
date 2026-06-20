# Expander 对抗训练流程与策略

本文记录 8x8 generated 地图上训练 PPO 策略超过 Expander 的可复用流程。它不是一次性 devlog，而是后续继续训练、复现实验和判断模型强度时的操作指南。

当前已验证的 checkpoint：

```text
/tmp/generals-ppo-8x8-expander-gpu-v5.eqx
```

该 checkpoint 在 sampled policy 模式下，对 randomized Expander 的独立 2048 局评估超过 90% 总胜率。`.eqx` 是实验产物，应保存在 `/tmp` 或实验目录，不提交进 Git。

当前新增目标是训练一个 adaptive checkpoint，在 8x8、12x12 和 16x16 generated 地图上都对 Expander 超过 90% 总胜率。它比现有 v5 结果更严格：同一个模型文件必须覆盖三个有效棋盘尺寸，并且每个尺寸都要测 player 0 和 player 1。

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

adaptive 多尺寸目标的验收标准：

- checkpoint：同一个 `AdaptivePolicyValueNetwork` `.eqx` 文件
- 有效尺寸：8x8、12x12、16x16
- padding：`--pad-to 16`
- 地图：generated
- opponent：`expander`
- policy mode：`sample`
- max steps：建议 750，避免 12x12/16x16 因截断过早变成 draw
- 每个尺寸和座位至少 2048 局
- 达标条件：六个 size-seat pair 的总胜率都超过 90%

使用 `evaluate_adaptive_policy.py --require-win-rate 0.90` 可以把该门槛变成 CI/脚本可读的非零退出条件。

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
- `--opponent-policy-pool a.eqx,b.eqx`：让非 learner 玩家从多个同架构 frozen checkpoint 中采样对手；不能和 `--opponent-policy-path` 或 `--self-play-opponent` 同时使用。每个 training iteration 会为每个环境采样一个 opponent index，并在该 iteration 的 rollout steps 内保持不变。
- `--opponent-policy-pool-modes sample,greedy`：指定 opponent pool 中每个 checkpoint 的执行模式，省略时全部使用 `sample`。
- `--learner-player 0|1`：选择 learner 控制环境中的哪个玩家槽位。用它可以分别训练先手/后手视角，避免只优化 player 0。
- `--terminal-reward-scale N`：在 decisive terminal transition 上给 learner 胜局 `+N`、败局 `-N`。默认 `0.0`，保持旧 composite reward 行为。
- `--checkpoint-dir DIR`、`--checkpoint-every N`、`--keep-checkpoints K`：周期保存训练中间 checkpoint，并可只保留最新 K 个，用于后续 league 评估和选模。

使用建议：

- 先固定一个 frozen opponent，确认 learner 视角、终局奖励和评估基线都稳定，再尝试 current-policy opponent。
- 每次 self-play 后都要重新测 Expander、其它 heuristic、历史 best checkpoint 和镜像座位。
- 如果新模型打赢历史模型但对 Expander 或 mixed heuristic 退化，不应替换 best checkpoint。
- 后续可以把多个历史 checkpoint 做成 league opponent，避免只针对一个 frozen/current opponent 过拟合。

### Checkpoint league best-response

当目标变成“对所有 heuristic 和 v5 都超过 80%”时，单一 frozen v5 对手不够可靠。推荐把历史 checkpoint 组成 ordinary policy opponent pool，并周期保存中间模型：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
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
  --lr 0.000001 \
  --truncation 500 \
  --init-model-path generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-pool generals-ppo-8x8-expander-gpu-v2.eqx,generals-ppo-8x8-expander-gpu-v3.eqx,generals-ppo-8x8-expander-gpu-v4.eqx,generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-pool-modes sample,sample,sample,sample \
  --learner-player 0 \
  --terminal-reward-scale 1.0 \
  --checkpoint-dir /tmp/generals-league-p0 \
  --checkpoint-every 50 \
  --keep-checkpoints 8 \
  --model-path /tmp/generals-ppo-8x8-league-p0-v1.eqx \
  --seed 30200
```

每个候选训练完后，用 league evaluator 统一验收：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_league.py /tmp/generals-ppo-8x8-league-p0-v1.eqx \
  --checkpoint-opponent v5=generals-ppo-8x8-expander-gpu-v5.eqx:sample \
  --num-games 1024 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --max-steps 500 \
  --policy-mode sample \
  --json-output /tmp/generals-ppo-8x8-league-p0-v1-league.json \
  --seed 30300
```

`evaluate_league.py` 默认评估所有 `HEURISTIC_NAMES` 的两个 seat；`--checkpoint-opponent` 用来加入 v5 或其它 frozen checkpoint。报告中的 `league_score` 是所有 required opponent-seat pair 的最低总胜率，因此它比平均胜率更适合作为 promotion gate。最终目标只有在每个 heuristic seat 和 v5 两个 seat 都超过 `80%` 时才算完成。

第一轮普通 checkpoint-pool PPO 从 v5 warm start，对 v2-v5 sample pool 训练 player 0。训练在 iter 120 手动中止，但 `--checkpoint-every 50` 保留了 iter 50/100：

```text
/tmp/generals-league-p0-v1/generals-ppo-8x8-league-p0-v1-iter-000050.eqx
/tmp/generals-league-p0-v1/generals-ppo-8x8-league-p0-v1-iter-000100.eqx
```

同 seed 512 局 league 评估显示，heuristic gate 全部保住，但 v5 gate 只有噪声级变化：

```text
v5 baseline league, seed 30300:
  heuristic required pairs passed = 12/12
  v5 player 0 = 237/224/51, win rate 46.29%
  v5 player 1 = 225/228/59, win rate 43.95%
  league_score = 43.95%

league p0-v1 iter 50:
  heuristic required pairs passed = 12/12
  v5 player 0 = 224/231/57, win rate 43.75%
  v5 player 1 = 230/233/49, win rate 44.92%
  league_score = 43.75%

league p0-v1 iter 100:
  heuristic required pairs passed = 12/12
  v5 player 0 = 226/238/48, win rate 44.14%
  v5 player 1 = 231/235/46, win rate 45.12%
  league_score = 44.14%
```

结论：ordinary checkpoint-pool PPO 能力已经可用，但第一轮没有产生接近 80% 的 best-response 信号。继续做纯 checkpoint 时，应优先引入 search teacher 或更强的 policy/value 改进目标，而不是只扩大同一 PPO 配方。

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

GUI 中使用同一个强辅助策略：

```bash
SEARCH_POLICY=1 ./play-v5.command
```

watch 单边搜索：

```bash
MODEL_0_SEARCH_POLICY=1 ./watch-v5.command
```

watch 双边搜索：

```bash
MODEL_0_SEARCH_POLICY=1 MODEL_1_SEARCH_POLICY=1 ./watch-v5.command
```

GUI 搜索默认使用 `top_k=4, rollout_steps=16, rollouts_per_action=4`，与下面的评测配置一致。若窗口交互太慢，可通过 `SEARCH_TOP_K`、`SEARCH_ROLLOUT_STEPS` 和 `SEARCH_ROLLOUTS_PER_ACTION` 临时降低预算。当前 GUI search agent 只支持 9 通道 observation checkpoint，因此适用于 v5 这类标准 checkpoint；18 通道 augmented/full-state checkpoint 仍应使用普通 PPO GUI 或另行实现对应搜索 prior。

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

2026-06-16 用相同 search 配置重新验证 v5-vs-v5，512 局两席都超过 80%：

```text
rollout-search as player 0 vs v5 sample, seed 30510:
  wins/losses/draws = 462/32/18
  win rate = 90.23%
  decisive win rate = 93.52%

rollout-search as player 1 vs v5 sample, seed 30511:
  wins/losses/draws = 454/43/15
  win rate = 88.67%
  decisive win rate = 91.35%
```

同日新增 `evaluate_league.py --search-policy` 后，先用 128 局/row 快速确认所有 heuristic 两席都超过 80%，随后扩大到 512 局/row 做强证据评估：

```text
v5 + rollout-search vs heuristic league, 512 games/row, seed 30530:
  expander:           p0 495/17/0,  p1 496/13/3
  city-rush:          p0 510/0/2,   p1 512/0/0
  general-hunter:     p0 507/2/3,   p1 508/1/3
  defensive-expander: p0 509/2/1,   p1 501/6/5
  balanced:           p0 507/1/4,   p1 509/1/2
  mixed:              p0 508/2/2,   p1 511/1/0
  required pairs = 12/12 passed
  heuristic league_score = 96.68%
```

当前严格表述：`v5 + rollout-search` 作为强辅助推理策略，已经在当前证据下超过所有 heuristic 和 v5 的 80% 胜率门槛；但这不是纯 `.eqx` checkpoint。若目标限定为纯模型文件，仍需继续把 search 行为蒸馏或训练进 checkpoint。

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

#### augmented PPO best-response 训练

`train.py` 现在也支持 `--policy-input`、`--input-channels`、`--init-input-channels`、`--opponent-policy-input` 和 `--opponent-input-channels`。这让 PPO rollout 本身可以使用 18 通道 augmented 输入，而不是只在蒸馏脚本中使用 privileged 特征。

从 v5 直接扩展到 18 通道并对 frozen v5 sample 训练 player 0：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
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
  --num-iterations 500 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --truncation 500 \
  --policy-input augmented-full-state \
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 2.0 \
  --model-path /tmp/generals-ppo-8x8-augmented-ppo-br-p0-v1.eqx \
  --seed 26110
```

结果有小幅 player 0 提升，但远离 80%，且 player 1 没有改善：

```text
/tmp/generals-ppo-8x8-augmented-ppo-br-p0-v1.eqx, player 0, seed 26120:
  candidate wins/losses/draws = 489/441/94
  same-seed v5 baseline       = 455/453/116

player 1, seed 26121:
  candidate wins/losses/draws = 471/453/100
  same-seed v5 baseline       = 476/456/92
```

从 p0 候选继续训练 player 1 时必须显式传 `--init-input-channels 18`，否则 18 通道 checkpoint 会被误按 9 通道 warm start 读取：

```bash
uv run python examples/_experimental/ppo/train.py 512 \
  --policy-input augmented-full-state \
  --input-channels 18 \
  --init-input-channels 18 \
  --init-model-path /tmp/generals-ppo-8x8-augmented-ppo-br-p0-v1.eqx \
  --opponent-policy-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-mode sample \
  --learner-player 1 \
  --terminal-reward-scale 2.0 \
  --model-path /tmp/generals-ppo-8x8-augmented-ppo-br-alt-v1.eqx
```

交替 seat 训练保住了 player 0 的小幅提升，但 player 1 变差：

```text
/tmp/generals-ppo-8x8-augmented-ppo-br-alt-v1.eqx, player 0, seed 26140:
  candidate wins/losses/draws = 479/462/83
  same-seed v5 baseline       = 432/471/121

player 1, seed 26141:
  candidate wins/losses/draws = 467/476/81
  same-seed v5 baseline       = 493/451/80
```

从 v5 直接训练 player 1 也没有成功：

```text
/tmp/generals-ppo-8x8-augmented-ppo-br-p1-v1.eqx, player 1, seed 26160:
  candidate wins/losses/draws = 446/492/86
  same-seed v5 baseline       = 468/470/86

player 0, seed 26161:
  candidate wins/losses/draws = 465/465/94
  same-seed v5 baseline       = 449/491/84
```

把 18 通道 augmented PPO 与 expanded-64 容量结合也退化：

```text
/tmp/generals-ppo-8x8-expanded64-augmented-ppo-br-p0-v1.eqx, player 0, seed 26220:
  candidate wins/losses/draws = 466/467/91
  same-seed v5 baseline       = 493/442/89

player 1, seed 26221:
  candidate wins/losses/draws = 463/472/89
  same-seed v5 baseline       = 495/433/96
```

把 terminal reward 从 `2.0` 提高到 `20.0` 会直接破坏策略：

```text
/tmp/generals-ppo-8x8-augmented-ppo-terminal20-p0-v1.eqx, player 0, seed 26320, 512 games:
  candidate wins/losses/draws = 111/374/27
  win rate = 21.68%
```

新增 general-target shaping 后，`train.py` 可以用完整 `GameState` 奖励强兵向敌方 general 靠近。该奖励是势能差：

```text
general_target_reward = scale * (potential_after - potential_before)
```

其中 potential 来自我方满足 `--general-target-min-army` 的 owned cells 到敌方 general 的最近曼哈顿距离。默认 scale 为 `0.0`，不改变旧训练。

从 v5 开始训练攻击性候选：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
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
  --num-iterations 500 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --truncation 500 \
  --policy-input augmented-full-state \
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 2.0 \
  --general-target-reward-scale 0.05 \
  --general-target-min-army 2 \
  --model-path /tmp/generals-ppo-8x8-general-target-p0-v1.eqx \
  --seed 26510
```

同 seed 评估结果：

```text
/tmp/generals-ppo-8x8-general-target-p0-v1.eqx, player 0, seed 26520:
  candidate wins/losses/draws = 476/473/75
  same-seed v5 baseline       = 488/450/86
  prior best augmented p0     = 473/458/93

player 1, seed 26521:
  candidate wins/losses/draws = 434/487/103
  same-seed v5 baseline       = 441/478/105
  prior best augmented p0     = 441/483/100
```

结论：general-target shaping 让 player 0 的 draw rate 从 prior best 的 `9.08%` 降到 `7.32%`，平均终局时间从 `289.8` 降到 `285.9`，说明策略更偏进攻；但总胜率没有超过 v5，也没有超过 prior best。它可以作为攻击性调节旋钮继续研究，但不能替代当前最佳胜率候选。

新增 path-assignment shaping 后，`train.py` 可以在 reward 计算内部缓存 shortest-path 距离场，而不修改 `GameState` 结构：

- enemy general distance map：所有 passable cell 到敌方 general 的最短路距离。
- non-owned city distance map：所有 passable cell 到中立/敌方城市的最短路距离。
- frontier distance map：所有 passable cell 到最近非己方 passable cell 的最短路距离。

每个满足 `--path-assignment-min-army` 的己方强兵格都会在这三类目标中选择加权势能最高的一类，作为该兵团当前的分配目标：

```text
path_assignment_reward = scale * (assigned_potential_after - assigned_potential_before)
```

这比 Manhattan general-target 更适合绕山运兵：如果最短路需要先远离敌方 general 才能绕到缺口，path-assignment 会给正奖励；原 Manhattan 势能会给负奖励。默认 scale 为 `0.0`，不改变旧训练。

从 v5 开始训练 full target-assignment 候选：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
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
  --num-iterations 500 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --truncation 500 \
  --policy-input augmented-full-state \
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 2.0 \
  --path-assignment-reward-scale 0.2 \
  --path-assignment-min-army 2 \
  --path-assignment-general-weight 1.0 \
  --path-assignment-city-weight 0.8 \
  --path-assignment-frontier-weight 0.25 \
  --model-path /tmp/generals-ppo-8x8-path-assignment-p0-v1.eqx \
  --seed 26610
```

该 full 版本降低 draw rate，但 player 0 强度下降，说明 frontier 目标会把运兵奖励拉向局部扩张：

```text
/tmp/generals-ppo-8x8-path-assignment-p0-v1.eqx, player 0, seed 26620:
  candidate wins/losses/draws = 464/484/76
  same-seed v5 baseline       = 491/437/96

player 1, seed 26621:
  candidate wins/losses/draws = 451/488/85
  same-seed v5 baseline       = 439/464/121
```

随后训练更保守的 general+city 版本，关闭 frontier 目标并降低 scale：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
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
  --num-iterations 500 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --truncation 500 \
  --policy-input augmented-full-state \
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 2.0 \
  --path-assignment-reward-scale 0.12 \
  --path-assignment-min-army 2 \
  --path-assignment-general-weight 1.0 \
  --path-assignment-city-weight 0.8 \
  --path-assignment-frontier-weight 0.0 \
  --model-path /tmp/generals-ppo-8x8-path-assignment-p0-v2.eqx \
  --seed 26630
```

同 seed 评估结果：

```text
/tmp/generals-ppo-8x8-path-assignment-p0-v2.eqx, player 0, seed 26620:
  candidate wins/losses/draws = 458/465/101
  same-seed v5 baseline       = 491/437/96

player 1, seed 26621:
  candidate wins/losses/draws = 462/473/89
  same-seed v5 baseline       = 439/464/121
```

最后用同一 general+city 配置训练 learner-player 1：

```text
/tmp/generals-ppo-8x8-path-assignment-p1-v1.eqx, player 1, seed 26621:
  candidate wins/losses/draws = 432/490/102
  same-seed v5 baseline       = 439/464/121

player 0, seed 26620:
  candidate wins/losses/draws = 437/481/106
  same-seed v5 baseline       = 491/437/96
```

结论：path-assignment shaping 能表达“沿真实最短路运兵”和“为不同兵团选择 general/city/frontier 目标”，也能降低部分 draw rate；但直接作为 PPO reward 时仍会引入错误局部目标，尤其是 frontier 权重。本轮最佳可观察信号是 p0-v2 作为 player 1 的总胜率从同 seed v5 baseline 的 `42.87%` 到 `45.12%`，但它没有在 player 0 或 decisive win rate 上形成稳定优势，不能作为新 best checkpoint。

新增 residual GRU 记忆 PPO 后，实验入口为：

- `examples/_experimental/ppo/recurrent_network.py`：`RecurrentPolicyValueNetwork`，在 CNN base 之后叠加 GRU hidden state 和 residual policy/value delta。delta heads 零初始化，因此初始 logits/value 等于 base CNN。
- `examples/_experimental/ppo/train_recurrent.py`：维护每个环境的 hidden state，episode reset 时清零；支持 frozen checkpoint opponent 或 heuristic/Expander opponent。
- `examples/_experimental/ppo/evaluate_recurrent_policy.py`：评估 recurrent checkpoint，评估时同样携带 hidden state。

预期 v5 warm-start 命令如下：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train_recurrent.py 512 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --pool-size 16384 \
  --num-steps 64 \
  --num-iterations 500 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000002 \
  --truncation 500 \
  --hidden-size 64 \
  --policy-input observation \
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 2.0 \
  --freeze-base \
  --model-path /tmp/generals-recurrent-ppo-8x8-v5-p0.eqx \
  --seed 26710
```

本轮执行时当前环境缺少 `/tmp/generals-ppo-8x8-expander-gpu-v5.eqx`，也没有 v4/BC 历史 checkpoint，因此不能直接完成 v5 warm-start。为验证 RNN 训练链路，先重新生成一个短训 Expander-soft BC warm-start：

```text
/tmp/generals-bc-8x8-rnn-warm.eqx, player 0 vs Expander, seed 26750:
  wins/losses/draws = 148/815/61
  win rate = 14.45%

player 1 vs Expander, seed 26751:
  wins/losses/draws = 169/803/52
  win rate = 16.50%
```

然后训练三类 recurrent 候选：

```text
/tmp/generals-recurrent-ppo-8x8-expander-fresh-v1.eqx
  fresh recurrent PPO vs Expander, no warm-start
  player 0: 0/419/605
  player 1: 0/440/584

/tmp/generals-recurrent-ppo-8x8-bc-expander-p0-v1.eqx
  BC warm-start, unfrozen base, 300 PPO iterations
  player 0: 0/869/155

/tmp/generals-recurrent-ppo-8x8-bc-expander-p0-short.eqx
  BC warm-start, unfrozen base, 30 PPO iterations
  player 0: 66/908/50

/tmp/generals-recurrent-ppo-8x8-bc-expander-p0-freeze-v1.eqx
  BC warm-start, --freeze-base, 100 PPO iterations
  player 0: 147/819/58
  player 1: 148/809/67
```

结论：RNN/GRU 机制已经可训练、可评估，并且 `--freeze-base` 能保护 warm-start base 不被 PPO 迅速破坏。没有强 v5 起点时，fresh recurrent PPO 学不到 Expander 胜局；弱 BC 起点上，unfrozen PPO 会退化，frozen-base RNN 基本保持 BC 强度但没有明显提升。

v5 checkpoint 放回仓库根目录后，执行了两组 frozen-base residual GRU 训练。共同设置：

- base/init/opponent：`generals-ppo-8x8-expander-gpu-v5.eqx`
- opponent mode：`sample`
- hidden size：64
- envs：512
- steps：64
- iterations：500
- epochs/minibatch：4 / 4096
- learning rate：`2e-6`
- terminal reward scale：`2.0`
- CNN base：`--freeze-base`

训练命令使用仓库根目录的 v5 文件：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train_recurrent.py 512 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --pool-size 16384 \
  --num-steps 64 \
  --num-iterations 500 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000002 \
  --truncation 500 \
  --hidden-size 64 \
  --freeze-base \
  --policy-input observation \
  --init-model-path generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-path generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 2.0 \
  --model-path /tmp/generals-recurrent-ppo-8x8-v5-freeze-p0-v1.eqx \
  --seed 26810
```

把 `--learner-player` 改成 `1` 并把输出路径改为 `/tmp/generals-recurrent-ppo-8x8-v5-freeze-p1-v1.eqx` 可复现第二组。

对 frozen v5 sample 的同 seed 评估如下：

```text
/tmp/generals-recurrent-ppo-8x8-v5-freeze-p0-v1.eqx, player 0, seed 26820:
  candidate wins/losses/draws = 480/442/102
  same-seed v5 baseline       = 459/463/102

player 1, seed 26821:
  candidate wins/losses/draws = 443/466/115
  same-seed v5 baseline       = 455/469/100

/tmp/generals-recurrent-ppo-8x8-v5-freeze-p1-v1.eqx, player 1, seed 26821:
  candidate wins/losses/draws = 447/463/114
  same-seed v5 baseline       = 455/469/100

player 0, seed 26820:
  candidate wins/losses/draws = 466/443/115
  same-seed v5 baseline       = 459/463/102
```

对 Expander heuristic 的独立 1024 局评估：

```text
/tmp/generals-recurrent-ppo-8x8-v5-freeze-p0-v1.eqx vs Expander, player 0, seed 26840:
  candidate wins/losses/draws = 927/80/17
  same-seed v5 baseline       = 922/89/13

player 1, seed 26841:
  candidate wins/losses/draws = 935/77/12
  same-seed v5 baseline       = 917/87/20
```

当前 RNN 结论：`/tmp/generals-recurrent-ppo-8x8-v5-freeze-p0-v1.eqx` 是这批 recurrent 训练里最好的候选。它对 v5 sample 的 player 0 有明确小幅提升，两个席位汇总为 `923/908/217`，优于同 seed v5 baseline 的 `914/932/202`；对 Expander 两个席位汇总为 `1862/157/29`，也高于同 seed v5 baseline 的 `1839/176/33`。但提升仍是小幅 residual memory gain，不是 80%+ best-response 级别的突破；继续训练时应保留 v5 与该 RNN checkpoint 双基线，下一步再尝试更低学习率、周期性评估保存、或部分解冻 CNN 后半层。

因此，当前 PPO best-response 结论是：

- 18 通道输入能被 PPO 训练链路正常使用。
- 从 v5 warm start 后，普通终局奖励只产生 2-5 个百分点级别的 seat-dependent 波动。
- 更强 terminal reward 会加速策略崩坏，而不是学出 best response。
- general-target shaping 会降低 draw rate/终局时间，但本次没有提升总胜率。
- path-assignment shaping 能减少路径盲区，但目标权重必须谨慎；frontier 目标容易把奖励拉向局部扩张。
- residual GRU memory 已可用；从 v5 冻结底座 warm start 时能带来小幅提升，但当前收益仍不足以替代 rollout-search 或 checkpoint league。
- expanded-64 容量没有改善 PPO 吸收 hidden-state 信息的能力。

#### 高 margin search 蒸馏中止记录

最后一次尝试改用高 margin search 标签，目标是只学习 search 评分明显优于 base top-prior 的样本，减少低 margin 噪声：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/conservative_search_distill.py 128 \
  --base-model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --policy-input augmented-full-state \
  --target-mode hard \
  --num-steps 64 \
  --num-iterations 500 \
  --minibatch-size 8192 \
  --min-margin 25 \
  --margin-scale 100 \
  --max-improve-weight 1.0 \
  --improve-weight 0.2 \
  --kl-weight 1.0 \
  --lr 0.000005 \
  --model-path /tmp/generals-ppo-8x8-augmented-hard-search-highmargin-v1.eqx \
  --seed 26410
```

该运行按用户指令在 iter 470 附近终止，脚本没有正常保存 checkpoint，因此没有可评估模型。训练日志显示：

```text
selected samples: usually 3.0%-5.3% of 8192
mean selected margin: roughly 310-360
KL near interrupt: about 0.03
```

这说明高 margin 样本确实存在，且不会立即把学生推离 v5；但本次中止前没有产生 checkpoint。若恢复该方向，应先给 `conservative_search_distill.py` 增加定期 checkpoint 保存，避免长实验被中断时丢失中间模型。

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

## 阶段七：adaptive 8/12/16 多尺寸训练

固定尺寸 v5 只解决 8x8。要推进“8x8、12x12、16x16 都超过 Expander 90%”的新目标，当前新增了一条单 checkpoint adaptive 训练路径：

- `AdaptivePolicyValueNetwork` 固定使用 `pad_to=16` 的输入画布。
- 输入通道在标准 fogged observation 外追加 active-cell、padding、坐标和尺寸比例信息。
- 动作空间固定为 `8 * pad_to * pad_to + 1`，最后一个 logit 是全局 pass。
- value head 只在 active cells 上池化，避免 padding 区域污染不同尺寸的价值估计。
- reset pool 默认按 `--grid-sizes` 做尺寸均衡采样，也可用 `--grid-size-weights` 对困难尺寸过采样；generated 地图按有效尺寸自动设置默认 minimum general distance。

推荐先训练 adaptive Expander-soft warm start：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 12288 \
  --num-steps 32 \
  --num-iterations 2000 \
  --lr 0.0007 \
  --channels 64,64,64,32 \
  --checkpoint-dir /tmp/generals-adaptive-bc-checkpoints \
  --checkpoint-every 100 \
  --keep-checkpoints 10 \
  --model-path /tmp/generals-adaptive-bc-8-12-16.eqx \
  --seed 47000
```

再从 BC checkpoint 对 Expander 做 PPO：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1.5,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 16384 \
  --num-steps 64 \
  --num-iterations 700 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --opponent expander \
  --learner-player mixed \
  --terminal-reward-scale 1.0 \
  --truncation-reward-scale 0.5 \
  --init-model-path /tmp/generals-adaptive-bc-8-12-16.eqx \
  --checkpoint-dir /tmp/generals-adaptive-ppo-checkpoints \
  --checkpoint-every 50 \
  --keep-checkpoints 10 \
  --model-path /tmp/generals-adaptive-ppo-8-12-16.eqx \
  --seed 47100
```

每个候选 checkpoint 都必须使用 size-seat 矩阵评估：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_adaptive_policy.py /tmp/generals-adaptive-ppo-8-12-16.eqx \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --num-games 2048 \
  --max-steps 750 \
  --opponent expander \
  --policy-mode sample \
  --map-generator generated \
  --json-output /tmp/generals-adaptive-ppo-8-12-16-eval.json \
  --require-win-rate 0.90 \
  --seed 47200
```

当前状态：adaptive 训练、BC、PPO 和评估基础设施已可运行并有 CPU smoke coverage。CUDA PPO 已把当前 best checkpoint 推到 70.31% 的六行最小胜率，后续 search-distill 候选把 512-row minimum 小幅推到 71.29%，但还没有任何 checkpoint 证明六个 size-seat pair 都超过 90%。后续训练应继续用 `--checkpoint-every` 保存 PPO 候选，并优先评估中间 checkpoint 的 `min_win_rate`，避免只看训练 rollout 胜率。

### CPU medium baseline

2026-06-16 在 CPU-only JAX 环境跑了一组中等 smoke，目标是验证 checkpoint 保存、训练链路和评估矩阵，不是冲击最终胜率。

BC 设置：

```text
model: /tmp/generals-adaptive-bc-medium.eqx
num_envs=64, num_steps=16, num_iterations=80, pool_size=192
grid_sizes=8,12,16, pad_to=16, map_generator=generated
checkpoint_every=20, keep_checkpoints=4
final train log: loss=3.1039, accuracy=18.5%
```

BC 的 32 games/row、300 step 评估：

```text
8x8 p0:  1/21/10, win rate 3.12%
8x8 p1:  2/18/12, win rate 6.25%
12x12 p0: 0/9/23, win rate 0.00%
12x12 p1: 1/13/18, win rate 3.12%
16x16 p0: 2/2/28, win rate 6.25%
16x16 p1: 1/1/30, win rate 3.12%
min_win_rate = 0.00%
```

随后从该 BC checkpoint 跑短 PPO：

```text
model: /tmp/generals-adaptive-ppo-medium.eqx
num_envs=64, num_steps=16, num_iterations=40, num_epochs=2, minibatch_size=512
opponent=expander, learner_player=0, terminal_reward_scale=1.0
final train log: iter 40 loss=0.0189, rollout wins=0
```

PPO 的 32 games/row、300 step 评估：

```text
8x8 p0:  1/23/8, win rate 3.12%
8x8 p1:  0/25/7, win rate 0.00%
12x12 p0: 1/7/24, win rate 3.12%
12x12 p1: 1/4/27, win rate 3.12%
16x16 p0: 0/1/31, win rate 0.00%
16x16 p1: 0/1/31, win rate 0.00%
min_win_rate = 0.00%
```

随后用 hard `--teacher expander` 跑了对照 BC。128 env、384 pool 的 run 在 iter 100 后手动中止，但保留了 `/tmp/generals-adaptive-bc-hard-medium-checkpoints/generals-adaptive-bc-hard-medium-iter-000100.eqx`。训练日志到 iter 100 时约为 `loss=3.2799, accuracy=15.7%`。同样 32 games/row、300 step 评估：

```text
8x8 p0:  2/24/6, win rate 6.25%
8x8 p1:  0/26/6, win rate 0.00%
12x12 p0: 1/6/25, win rate 3.12%
12x12 p1: 0/9/23, win rate 0.00%
16x16 p0: 0/3/29, win rate 0.00%
16x16 p1: 0/3/29, win rate 0.00%
min_win_rate = 0.00%
```

另一个对照是从同一个 soft BC checkpoint 训练 `learner_player=1`，其余短 PPO 参数与 player 0 run 对齐。训练日志中出现少量 player 1 rollout win，但评估仍远弱：

```text
model: /tmp/generals-adaptive-ppo-medium-p1.eqx
8x8 p0:  2/24/6, win rate 6.25%
8x8 p1:  3/25/4, win rate 9.38%
12x12 p0: 2/12/18, win rate 6.25%
12x12 p1: 0/12/20, win rate 0.00%
16x16 p0: 0/3/29, win rate 0.00%
16x16 p1: 0/4/28, win rate 0.00%
min_win_rate = 0.00%
```

结论：短 CPU 训练量远远不足以产生可用 adaptive checkpoint；它只证明基础设施能跑、checkpoint 能保留、评估能输出完整矩阵。hard Expander teacher 在这个训练量下没有明显优于 soft target，且仍有多个 size-seat pair 为 0%。单独训练 player 1 能带来一点局部变化，但不能解决 12x12/16x16 的弱项。下一轮有意义的实验应使用 CUDA JAX 跑完整 BC 配方，或先设计并实现双座位/交替座位 adaptive PPO 训练，避免只针对一个 `learner_player` 更新。

### GPU adaptive run v1

2026-06-16 启用 `uv run --extra dev --extra cuda13` 后，JAX 可使用 `CudaDevice(id=0)`。先跑 adaptive BC warm start：

```text
model: /tmp/generals-adaptive-bc-gpu-v1.eqx
effective training: 512 envs, 32 steps, about 1000 total BC iterations
pool_size=12288, grid_sizes=8,12,16, pad_to=16
final BC log: loss around 2.54, accuracy around 23%
```

该 BC checkpoint 的 256 games/row、750 step 评估：

```text
8x8 p0: 35.55%
8x8 p1: 36.72%
12x12 p0: 35.94%
12x12 p1: 33.98%
16x16 p0: 21.09%
16x16 p1: 21.09%
min_win_rate = 21.09%
```

随后从 BC 进行一系列 PPO probe：

```text
p0 all-size PPO -> /tmp/generals-adaptive-ppo-gpu-p0-v1.eqx
  256 games/row min_win_rate = 28.52%

p0 -> p1 all-size PPO -> /tmp/generals-adaptive-ppo-gpu-p0p1-v1.eqx
  256 games/row min_win_rate = 56.25%

p0 -> p1 -> p0 all-size PPO, iter-100 early stop
  /tmp/generals-adaptive-ppo-gpu-alt6-p0-v1-checkpoints/generals-adaptive-ppo-gpu-alt6-p0-v1-iter-000100.eqx
  256 games/row min_win_rate = 63.28%

16x16-only p1 then 16x16-only p0 curriculum
  /tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx
  512 games/row min_win_rate = 70.31%
```

当前 best adaptive checkpoint 是 `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx`。512 games/row、750 step、sample policy 对 Expander 的矩阵：

```text
8x8 p0:  375/136/1, win rate 73.24%
8x8 p1:  380/130/2, win rate 74.22%
12x12 p0: 409/78/25, win rate 79.88%
12x12 p1: 408/84/20, win rate 79.69%
16x16 p0: 370/48/94, win rate 72.27%
16x16 p1: 360/59/93, win rate 70.31%
min_win_rate = 70.31%
```

Negative follow-ups:

- Continuing all-size p1 from the 16-only best (`/tmp/generals-adaptive-ppo-gpu-alt8-p1-v1.eqx`) reduced the 256-row `min_win_rate` to 66.41%.
- 8x8-only p0 training from the 16-only best (`/tmp/generals-adaptive-ppo-gpu-8p0-v1.eqx`) reduced the 256-row `min_win_rate` to 64.06%, mostly by hurting 16x16.
- 8x16-focused p1 training (`/tmp/generals-adaptive-ppo-gpu-8x16p1-v1.eqx`) reached only 68.16% over 512 games/row; p0 symmetry (`/tmp/generals-adaptive-ppo-gpu-8x16p0-v1.eqx`) reached only 67.97% over 256 games/row.
- A second 16x16-only p1 continuation (`/tmp/generals-adaptive-ppo-gpu-16p1-v2.eqx`) looked promising at 256 games/row but fell to 68.75% over 512 games/row.
- Raising `--terminal-reward-scale` to `2.0` for all-size p1 (`/tmp/generals-adaptive-ppo-gpu-term2-p1-v1.eqx`) reached only 69.92% over 256 games/row.

结论：GPU 训练把 adaptive checkpoint 从 CPU baseline 的 0% 推到 70% min win rate，证明 adaptive architecture 和 alternating/curriculum PPO 方向有效；但距离六行都超过 90% 仍有明显差距。现有单座位续训、8x16 课程和单纯提高终局奖励已经进入平台期。下一轮优先方向应是引入新的训练信号来降低 16x16 draw rate 与提升 8x8 decisive strength，而不是继续盲目 low-lr fine-tune。可尝试：显式 draw/timeout 惩罚、按尺寸加权采样、真正的双座位同批训练，或把 rollout-search/target-assignment 信号接入 adaptive trainer。

### Adaptive trainer v2 controls

2026-06-16 新增 trainer-v2 控制项：

- `--grid-size-weights 8:1.5,12:1,16:2`：在 adaptive reset pool 中按权重分配有效尺寸，避免困难的 16x16 样本不足。
- `--learner-player alternate`：按 training iteration 在 player 0 和 player 1 之间交替 learner seat，降低单座位 fine-tune 造成的遗忘。
- `--learner-player mixed`：把总 `num_envs` 拆成 player 0 和 player 1 两半，分别收集 learner 轨迹后在同一个 PPO batch 中拼接更新，避免 iteration 级交替带来的座位跷跷板。
- `--truncation-reward-scale 0.5`：对达到 truncation 且非 decisive terminal 的 transition 给 learner 负奖励，直接压低 16x16 高 draw 率。

下一条建议从当前 best checkpoint 继续：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/train_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1.5,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 8192 \
  --num-steps 64 \
  --num-iterations 300 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --opponent expander \
  --learner-player mixed \
  --terminal-reward-scale 1.0 \
  --truncation-reward-scale 0.5 \
  --init-model-path /tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx \
  --checkpoint-dir /tmp/generals-adaptive-ppo-gpu-v2-checkpoints \
  --checkpoint-every 50 \
  --keep-checkpoints 6 \
  --model-path /tmp/generals-adaptive-ppo-gpu-v2.eqx \
  --seed 62016
```

评估顺序：先对 final 和保留的 checkpoint 做 256 games/row triage；若 `min_win_rate` 高于 70.31%，再升到 512 games/row 或 2048 games/row。只有六个 size-seat pair 的总胜率都超过 90%，才可替换当前 best。

trainer-v2 首轮结果没有超过当前 best：

```text
/tmp/generals-adaptive-ppo-gpu-v2.eqx
  config: weights 8:1.5,12:1,16:2, learner_player=alternate, truncation_reward_scale=0.5
  256 games/row min_win_rate = 67.97%
  16x16 p0 = 174/25/57, win rate 67.97%
  16x16 p1 = 184/22/50, win rate 71.88%

/tmp/generals-adaptive-ppo-gpu-v2-checkpoints/generals-adaptive-ppo-gpu-v2-iter-000100.eqx
  512 games/row min_win_rate = 67.19%
```

去掉截断惩罚的隔离对照也没有超过当前 best：

```text
/tmp/generals-adaptive-ppo-gpu-v2-notrunc.eqx
  config: weights 8:1.5,12:1,16:2, learner_player=alternate, truncation_reward_scale=0.0
  256 games/row min_win_rate = 68.36%
```

结论：weighted pool + alternating seat 可以把部分 12x12 行拉高，但会牺牲 8x8 或 16x16；`truncation_reward_scale=0.5` 不是当前瓶颈的直接解。下一步应尝试更大 adaptive 网络容量，例如 `--channels 64,64,64,32`，先用 adaptive BC warm start，再用 PPO 细调和 size-seat matrix 评估。

### Adaptive capacity probes

直接从大容量 BC 重新起步没有超过当前 best：

```text
/tmp/generals-adaptive-bc-wide-v1.eqx
  config: channels=64,64,64,32, weights 8:1,12:1,16:2
  128 games/row min_win_rate = 16.41%
  8x8 rows around 42-44%, 12x12 rows around 41-42%, 16x16 rows around 16-18% with >60% draw

/tmp/generals-adaptive-ppo-wide-v1-smallbatch.eqx
  config: 128 env, 16 steps, learner_player=alternate
  128 games/row min_win_rate = 18.75%

/tmp/generals-adaptive-ppo-wide-p0-v1.eqx
  config: 128 env, 32 steps, learner_player=0
  128 games/row min_win_rate = 31.25%
```

结论：wide-from-scratch/BC 会丢掉当前 best 的 70% 策略质量；真正有意义的容量路线应该从当前 best 小模型扩容，而不是从弱 BC 重新学。`load_or_create_adaptive_network(..., init_channels=...)` 现在支持把小 adaptive checkpoint 按通道前缀复制到更宽网络，额外卷积特征保留随机初始化，但它们通往旧输出/head 的连接初始为零，因此初始 logits/value 与源 checkpoint 一致且新通道仍可学习。

下一条建议从当前 best 做零填充扩容：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/train_adaptive.py 128 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 400 \
  --num-epochs 4 \
  --minibatch-size 2048 \
  --lr 0.000005 \
  --channels 64,64,64,32 \
  --init-channels 32,32,32,16 \
  --opponent expander \
  --learner-player alternate \
  --terminal-reward-scale 1.0 \
  --init-model-path /tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx \
  --checkpoint-dir /tmp/generals-adaptive-ppo-expanded-v1-checkpoints \
  --checkpoint-every 50 \
  --keep-checkpoints 8 \
  --model-path /tmp/generals-adaptive-ppo-expanded-v1.eqx \
  --seed 63300
```

扩容 PPO 实验结果：

```text
/tmp/generals-adaptive-ppo-expanded-v1.eqx
  config: output-preserving expansion 32,32,32,16 -> 64,64,64,32, all-size alternate
  256 games/row min_win_rate = 68.75%
  /tmp/generals-adaptive-ppo-expanded-v1-checkpoints/...-iter-000100.eqx
    512 games/row min_win_rate = 67.58%

/tmp/generals-adaptive-ppo-expanded-v2.eqx
  config: trainable extra conv channels, all-size alternate
  256 games/row min_win_rate = 68.36%
  best 128-row retained checkpoints only reached 70.31%, not enough to justify 512-row promotion

/tmp/generals-adaptive-ppo-expanded-16p1-v1.eqx
  config: trainable expansion, 16x16-only, learner_player=1
  256 games/row min_win_rate = 68.75%
```

结论：扩容 warm start 本身可用，且能保持源 checkpoint 初始行为；但当前 PPO objective 仍会在尺寸/座位之间迁移强度，未突破 70.31% best。下一轮不应继续只改 `lr`、weights 或 seat schedule。更有希望的方向是把 rollout-search、target-assignment 或更强 teacher 信号接入 adaptive 训练，先用监督/蒸馏提高 tactical finish rate，再做 PPO fine-tune。

### Adaptive search distillation

2026-06-16 新增 `examples/_experimental/ppo/adaptive_search_distill.py`，把固定尺寸 conservative search distillation 搬到 adaptive action space：冻结 `/tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx` 作为 search prior、KL anchor 和 rollout/opponent policy，用 top-k 短 rollout 分数做 soft target。

首轮窄模型 p1 distillation 配置：

```text
model: /tmp/generals-adaptive-search-distill-p1-v1.eqx
base: /tmp/generals-adaptive-ppo-gpu-16p0-v1.eqx
channels: 32,32,32,16
grid_size_weights: 8:1,12:1,16:2
target_mode: soft
learner_player: 1
num_envs=256, num_steps=8, num_iterations=40
top_k=4, rollout_steps=8, rollouts_per_action=2
lr=1e-6, kl_weight=1.0, improve_weight=0.05
```

Retained checkpoint triage at 256 games/row:

```text
iter 10: min_win_rate = 70.70% (8p0 72.66, 8p1 73.44, 12p0 83.59, 12p1 85.16, 16p0 70.70, 16p1 74.22)
iter 20: min_win_rate = 67.58% (16p1 bottleneck)
iter 30: min_win_rate = 63.28% (16p0 bottleneck)
iter 40: min_win_rate = 70.70% (8p0 72.66, 8p1 75.00, 12p0 82.81, 12p1 82.42, 16p0 72.27, 16p1 70.70)
```

iter 40 promoted to 512 games/row:

```text
/tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
8x8 p0: 367/142/3, win rate 71.68%
8x8 p1: 382/130/0, win rate 74.61%
12x12 p0: 424/73/15, win rate 82.81%
12x12 p1: 427/73/12, win rate 83.40%
16x16 p0: 367/48/97, win rate 71.68%
16x16 p1: 365/60/87, win rate 71.29%
min_win_rate = 71.29%
```

结论：adaptive search distillation 首轮小幅超过当前 70.31% best，但没有接近 90%。该 checkpoint 可作为新的低门槛候选，但不能宣布目标完成，也不值得直接升到 2048 games/row。下一轮应尝试更强 search budget 或按两个 learner seat 分开蒸馏后再做低学习率 PPO fine-tune，重点观察 8x8 与 16x16 行是否同时提升。

后续三个 follow-up 都没有超过 p1 r8 iter40：

```text
p1 -> p0 sequential search distill, same budget:
  /tmp/generals-adaptive-search-distill-p1p0-v1-ckpts/...-iter-000010.eqx: 256-row min = 68.36%
  /tmp/generals-adaptive-search-distill-p1p0-v1-ckpts/...-iter-000020.eqx: 256-row min = 67.19%
  /tmp/generals-adaptive-search-distill-p1p0-v1-ckpts/...-iter-000030.eqx: 256-row min = 65.62%
  /tmp/generals-adaptive-search-distill-p1p0-v1-ckpts/...-iter-000040.eqx: 256-row min = 65.23%

p1 search distill with rollout_steps=16:
  /tmp/generals-adaptive-search-distill-p1-v2-r16-ckpts/...-iter-000010.eqx: 256-row min = 69.92%
  /tmp/generals-adaptive-search-distill-p1-v2-r16-ckpts/...-iter-000020.eqx: 256-row min = 67.58%
  /tmp/generals-adaptive-search-distill-p1-v2-r16-ckpts/...-iter-000030.eqx: 256-row min = 67.97%

low-lr alternate PPO follow-up from p1 r8 iter40:
  /tmp/generals-adaptive-search-distill-p1-v1-ppo-alt-v1-ckpts/...-iter-000030.eqx: 256-row min = 67.97%
  /tmp/generals-adaptive-search-distill-p1-v1-ppo-alt-v1-ckpts/...-iter-000060.eqx: 256-row min = 69.53%
  /tmp/generals-adaptive-search-distill-p1-v1-ppo-alt-v1-ckpts/...-iter-000090.eqx: 256-row min = 68.36%
  /tmp/generals-adaptive-search-distill-p1-v1-ppo-alt-v1-ckpts/...-iter-000120.eqx: 256-row min = 68.75%
```

结论更新：简单连续换座位 distillation、单纯加长 rollout search budget、以及常规 alternate PPO follow-up 都会重新引入 size/seat tradeoff。当前最好的 adaptive 候选仍是 p1 r8 iter40 的 512-row `71.29%` minimum。下一步更应该改训练目标本身，例如只训练高置信 search 改进样本、加入 draw/finish auxiliary target，或做双座位同批 KL/CE，避免单座位更新把另一个座位压下去。

随后新增 `--soft-weight-mode active|improvement`，让 soft target 可以只对 margin-selected search improvements 赋权，而不是对所有 active 样本赋权。后续又新增 `--soft-improvement-extra-weight`，用于在 active soft CE 之外叠加高 margin improvement CE；默认 `0.0` 保持旧行为。两组 high-confidence p1 probe 没有超过 active-soft p1 r8 iter40：

```text
improvement mode, min_margin=1, margin_scale=4:
  /tmp/generals-adaptive-search-distill-p1-improve-v1-ckpts/...-iter-000010.eqx: 256-row min = 64.45%
  /tmp/generals-adaptive-search-distill-p1-improve-v1-ckpts/...-iter-000020.eqx: 256-row min = 69.14%
  /tmp/generals-adaptive-search-distill-p1-improve-v1-ckpts/...-iter-000030.eqx: 256-row min = 67.58%
  /tmp/generals-adaptive-search-distill-p1-improve-v1-ckpts/...-iter-000040.eqx: 256-row min = 71.09%

improvement mode, min_margin=0.2, margin_scale=1:
  /tmp/generals-adaptive-search-distill-p1-improve-v2-m02-ckpts/...-iter-000010.eqx: 256-row min = 69.92%
  /tmp/generals-adaptive-search-distill-p1-improve-v2-m02-ckpts/...-iter-000020.eqx: 256-row min = 66.80%
  /tmp/generals-adaptive-search-distill-p1-improve-v2-m02-ckpts/...-iter-000030.eqx: 256-row min = 69.14%
```

结论更新：纯 improvement-only weighting 选择样本太少，且会放大不稳定 seat/size 迁移；它不是 active-soft 目标的直接替代。下一轮可用 `--soft-weight-mode active --soft-improvement-extra-weight N` 跑混合目标，保留 all-active soft/KL 稳定项，同时给 high-margin improvement 样本增加额外 loss；如果这仍不行，再转向 finish/draw/Q-value 辅助目标。

混合目标 probe 结果同样没有超过 p1 r8 iter40：

```text
active soft + improvement extra, extra_weight=0.02, min_margin=0.2:
  /tmp/generals-adaptive-search-distill-p1-mixed-v1-ckpts/...-iter-000010.eqx: 256-row min = 71.88%
    promoted to 512-row: min = 68.95% (8p0 73.83, 8p1 69.53, 12p0 81.45, 12p1 83.98, 16p0 71.88, 16p1 68.95)
  /tmp/generals-adaptive-search-distill-p1-mixed-v1-ckpts/...-iter-000020.eqx: 256-row min = 65.62%
  /tmp/generals-adaptive-search-distill-p1-mixed-v1-ckpts/...-iter-000030.eqx: 256-row min = 63.67%
  /tmp/generals-adaptive-search-distill-p1-mixed-v1-ckpts/...-iter-000040.eqx: 256-row min = 70.70%

active soft + improvement extra, extra_weight=0.005, min_margin=0.2:
  /tmp/generals-adaptive-search-distill-p1-mixed-v2-x005-ckpts/...-iter-000010.eqx: 256-row min = 68.75%
  /tmp/generals-adaptive-search-distill-p1-mixed-v2-x005-ckpts/...-iter-000020.eqx: 256-row min = 71.09%
  /tmp/generals-adaptive-search-distill-p1-mixed-v2-x005-ckpts/...-iter-000030.eqx: 256-row min = 67.58%
```

结论更新：额外 improvement CE 没有解决 16x16 draw/finish bottleneck，还会使 8p1 或 16p1 在更大样本下掉队。继续在同一 search-CE family 内调权重的价值很低；下一步应改成 outcome/finish 辅助信号，或直接让 adaptive rollout-search evaluator 证明 search teacher 在 8/12/16 上是否有足够上限。

### Adaptive PPO v3-noarch controls

2026-06-16 新增 `train_adaptive.py` 的 v3-noarch 训练控制项：

- `--reward-mode terminal`：关闭 dense `composite_reward_fn`，只保留 decisive terminal win/loss reward，避免继续强化局部 material/path 代理目标。
- `--gamma` 和 `--gae-lambda`：允许从旧的 `0.99/0.95` 切到更长时序的 `1.0/0.9`。
- `--top-advantage-fraction`：每个 PPO batch 只用最高 advantage 分位的 transition 更新 policy/entropy，value loss 仍使用完整 batch。
- `--ema-decay` 和 `--eval-ema`：维护参数 EMA；开启 `--eval-ema` 时 periodic checkpoint 和 final model 保存 EMA 参数，便于直接用现有 evaluator 比较 EMA。

下一条 GPU continuation 从当前最强的 71.29%/512-row search-distill candidate 启动，先看 256 games/row triage，再决定是否 promotion 到 512-row：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/train_adaptive.py 512 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 16384 \
  --num-steps 256 \
  --num-iterations 80 \
  --num-epochs 1 \
  --minibatch-size 4096 \
  --lr 0.000003 \
  --opponent expander \
  --learner-player mixed \
  --reward-mode terminal \
  --terminal-reward-scale 1.0 \
  --gamma 1.0 \
  --gae-lambda 0.9 \
  --top-advantage-fraction 0.25 \
  --ema-decay 0.999 \
  --eval-ema \
  --init-model-path /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx \
  --checkpoint-dir /tmp/generals-adaptive-ppo-v3-noarch-ckpts \
  --checkpoint-every 10 \
  --keep-checkpoints 8 \
  --model-path /tmp/generals-adaptive-ppo-v3-noarch.eqx \
  --seed 66016
```

如果 v3-noarch 仍不能把 512-row minimum 明显推过 75%，下一步不应再调 learning rate 或 search CE 权重，而应实现 HL-Gauss by-size value head 或 memory-stack adaptive network。

GPU smoke result:

```text
model: /tmp/generals-adaptive-ppo-v3-noarch-gpu-smoke.eqx
base: /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
config: 128 envs, num_steps=64, num_iterations=5, reward_mode=terminal, gamma=1.0, gae_lambda=0.9,
        top_advantage_fraction=0.25, ema_decay=0.999, eval_ema, learner_player=mixed
train log:
  iter 1: loss=-0.8462, episodes=6, wins=6, draws=0, SPS=1595
  iter 5: loss=-0.6967, episodes=23, wins=16, draws=0, SPS=58571
64 games/row eval:
  8x8 p0: 49/15/0, win rate 76.56%
  8x8 p1: 51/13/0, win rate 79.69%
  12x12 p0: 48/14/2, win rate 75.00%
  12x12 p1: 48/15/1, win rate 75.00%
  16x16 p0: 48/7/9, win rate 75.00%
  16x16 p1: 43/6/15, win rate 67.19%
  min_win_rate = 67.19%
```

结论：GPU v3-noarch smoke 证明 CUDA training、EMA checkpoint 保存和 evaluator 加载链路可用；5 iteration/64-row 样本太小，不能作为棋力结论。下一步要跑上面的 80-iteration GPU continuation，并用 256 games/row triage 判断是否值得升到 512-row。

Full 512-env command failed on the local 16GB RTX 5070 Ti:

```text
config: 512 envs, num_steps=256, minibatch_size=4096
failure: JaxRuntimeError RESOURCE_EXHAUSTED while allocating 1.88GiB inside train_minibatch_step
root cause: rollout storage plus 4096-sample backward pass is too large for this GPU
working local config: 256 envs, num_steps=256, minibatch_size=1024
```

256-env terminal-only EMA run:

```text
model: /tmp/generals-adaptive-ppo-v3-noarch-256env.eqx
base: /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
config: 256 envs, num_steps=256, num_iterations=40, minibatch_size=1024,
        reward_mode=terminal, gamma=1.0, gae_lambda=0.9, top_advantage_fraction=0.25,
        ema_decay=0.999, eval_ema, learner_player=mixed
train log:
  iter 1:  loss=-0.6703, episodes=96,  wins=73,  draws=0,  SPS=12613
  iter 10: loss=-0.7758, episodes=181, wins=142, draws=13, SPS=88363
  iter 20: loss=-0.7416, episodes=174, wins=130, draws=19, SPS=90196
  iter 30: loss=-0.7435, episodes=170, wins=130, draws=18, SPS=89218
  iter 40: loss=-0.7014, episodes=157, wins=113, draws=17, SPS=88203
256 games/row eval, seed 66030:
  final EMA: min_win_rate = 69.53%
  iter 10 EMA: min_win_rate = 67.97%
  iter 20 EMA: min_win_rate = 69.14%
  iter 30 EMA: min_win_rate = 69.14%
```

Same-seed base control:

```text
/tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
256 games/row, seed 66030:
  8x8 p0 68.75%, 8x8 p1 69.14%, 12x12 p0 80.86%, 12x12 p1 81.25%,
  16x16 p0 70.70%, 16x16 p1 72.27%, min_win_rate = 68.75%
```

Composite reward control:

```text
model: /tmp/generals-adaptive-ppo-v3-composite-256env.eqx
config: same as terminal run, but reward_mode=composite
train log:
  iter 1:  loss=-0.9399, episodes=97,  wins=78,  draws=0,  SPS=12529
  iter 10: loss=-0.9470, episodes=165, wins=127, draws=12, SPS=87647
  iter 20: loss=-0.9768, episodes=165, wins=131, draws=17, SPS=88454
  iter 30: loss=-0.9482, episodes=167, wins=131, draws=16, SPS=87882
  iter 40: loss=-0.9008, episodes=169, wins=127, draws=14, SPS=88464
256 games/row eval, seed 66030:
  min_win_rate = 68.75%
```

Terminal last-iterate control:

```text
model: /tmp/generals-adaptive-ppo-v3-terminal-last-256env.eqx
config: same as terminal EMA run, but without --eval-ema
256 games/row eval, seed 66030:
  8x8 p0 69.14%, 8x8 p1 69.14%, 12x12 p0 80.08%, 12x12 p1 82.03%,
  16x16 p0 71.88%, 16x16 p1 69.53%, min_win_rate = 69.14%
```

结论更新：v3-noarch infrastructure works on GPU, but this isolated recipe does not break the adaptive plateau. Against the same 256-row seed, terminal EMA is only noise-level above the base and below the earlier 512-row 71.29% candidate; composite and last-iterate controls also fail promotion. The next implementation step should be value-target quality, specifically HL-Gauss/by-size value or finish/draw auxiliary targets, rather than more reward/seat/CE weight tuning.

### Adaptive HL-Gauss value upgrade

2026-06-16 新增 adaptive categorical value path：

- `AdaptivePolicyValueNetwork(..., value_bins=N)` 会保留原 scalar value heads，同时新增 shared/per-size categorical value logits。
- `train_adaptive.py --value-loss hl-gauss` 使用 HL-Gauss target distribution 和 categorical cross-entropy 训练 value head；policy logits、old logprob 和 entropy 仍按原 PPO objective 更新。
- `--value-heads per-size --init-value-heads shared` 可从旧 shared scalar checkpoint warm start 到 per-size categorical value heads；policy trunk/pass/policy logits 保持可迁移。
- categorical/per-size checkpoint 评估时必须给 `evaluate_adaptive_policy.py` 传匹配的 `--value-heads`、`--value-loss hl-gauss --value-bins ... --value-sigma ...`，否则 loader 会按错误模板读 checkpoint。

本地 16GB RTX 5070 Ti 上，512 envs x 256 steps x minibatch 4096 已确认 OOM；下一轮 GPU triage 使用 256 envs、256 rollout、minibatch 1024：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/train_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 16384 \
  --num-steps 256 \
  --num-iterations 40 \
  --num-epochs 1 \
  --minibatch-size 1024 \
  --lr 0.000003 \
  --opponent expander \
  --learner-player mixed \
  --reward-mode terminal \
  --terminal-reward-scale 1.0 \
  --gamma 1.0 \
  --gae-lambda 0.9 \
  --top-advantage-fraction 0.25 \
  --ema-decay 0.999 \
  --eval-ema \
  --value-heads per-size \
  --init-value-heads shared \
  --value-loss hl-gauss \
  --init-value-loss mse \
  --value-bins 128 \
  --value-sigma 0.04 \
  --init-model-path /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx \
  --checkpoint-dir /tmp/generals-adaptive-ppo-v3-hlgauss-ckpts \
  --checkpoint-every 10 \
  --keep-checkpoints 4 \
  --model-path /tmp/generals-adaptive-ppo-v3-hlgauss.eqx \
  --seed 67000
```

256 games/row triage command:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/evaluate_adaptive_policy.py /tmp/generals-adaptive-ppo-v3-hlgauss.eqx \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --num-games 256 \
  --max-steps 750 \
  --opponent expander \
  --policy-mode sample \
  --map-generator generated \
  --value-heads per-size \
  --value-loss hl-gauss \
  --value-bins 128 \
  --value-sigma 0.04 \
  --json-output /tmp/generals-adaptive-ppo-v3-hlgauss-eval256.json \
  --seed 67030
```

GPU smoke result:

```text
model: /tmp/generals-adaptive-ppo-v3-hlgauss-smoke.eqx
base: /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
config: 64 envs, num_steps=32, num_iterations=2, minibatch_size=512,
        reward_mode=terminal, gamma=1.0, gae_lambda=0.9, top_advantage_fraction=0.25,
        ema_decay=0.999, eval_ema, value_heads=per-size, value_loss=hl-gauss, 128 bins
train log:
  iter 1: loss=13.1120, episodes=1, wins=1, draws=0, SPS=406
  iter 2: loss=13.3373, episodes=1, wins=1, draws=0, SPS=28134
16 games/row evaluator smoke, seed 67010:
  8x8 p0 50.00%, 8x8 p1 62.50%, 12x12 p0 25.00%, 12x12 p1 68.75%,
  16x16 p0 12.50%, 16x16 p1 0.00%, min_win_rate = 0.00%
```

The smoke only proves the categorical checkpoint can warm-start from the scalar search-distill checkpoint, train on CUDA, save EMA parameters, and be loaded by `evaluate_adaptive_policy.py` with the matching value-loss template. Its 16-row evaluation is intentionally too small and too short to judge strength.

256-env HL-Gauss triage:

```text
model: /tmp/generals-adaptive-ppo-v3-hlgauss.eqx
base: /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
config: 256 envs, num_steps=256, num_iterations=40, minibatch_size=1024,
        reward_mode=terminal, gamma=1.0, gae_lambda=0.9, top_advantage_fraction=0.25,
        ema_decay=0.999, eval_ema, value_heads=per-size, value_loss=hl-gauss, 128 bins
train log:
  iter 1:  loss=13.0792, episodes=96,  wins=73, draws=0,  SPS=11232
  iter 10: loss=8.2170,  episodes=151, wins=88, draws=27, SPS=84059
  iter 20: loss=5.5432,  episodes=146, wins=61, draws=35, SPS=84213
  iter 30: loss=4.2508,  episodes=142, wins=49, draws=37, SPS=84007
  iter 40: loss=3.6038,  episodes=140, wins=38, draws=39, SPS=84208
```

256 games/row eval, seed 67030:

```text
base scalar search-distill iter40:
  8p0 75.00%, 8p1 71.09%, 12p0 82.81%, 12p1 85.16%, 16p0 69.14%, 16p1 74.61%
  min_win_rate = 69.14%

HL-Gauss iter10:
  8p0 71.88%, 8p1 71.09%, 12p0 81.64%, 12p1 80.47%, 16p0 69.14%, 16p1 76.56%
  min_win_rate = 69.14%

HL-Gauss iter20:
  8p0 74.61%, 8p1 70.31%, 12p0 80.08%, 12p1 82.42%, 16p0 67.58%, 16p1 74.61%
  min_win_rate = 67.58%

HL-Gauss iter30:
  8p0 75.39%, 8p1 73.83%, 12p0 78.91%, 12p1 84.77%, 16p0 69.92%, 16p1 74.22%
  min_win_rate = 69.92%

HL-Gauss iter40/final:
  8p0 73.83%, 8p1 68.75%, 12p0 80.47%, 12p1 78.52%, 16p0 70.70%, 16p1 73.83%
  min_win_rate = 68.75%
```

结论更新：HL-Gauss/per-size value heads are implemented and trainable, but this direct PPO continuation does not break the adaptive plateau. The best retained checkpoint, iter 30, only improves the same-seed 256-row base by 0.78 percentage points and remains below the earlier 71.29%/512-row candidate. Do not promote this run to 512-row validation. The training log also shows rollout wins declining as categorical value loss falls, so the next step should change representation or auxiliary targets: memory stack/global context, finish/draw auxiliary, or search-to-Q/intent distillation. Repeating sparse PPO with the same CNN trunk is unlikely to fix the weak 8x8/16x16 rows.

Promotion rule remains unchanged: only if the six-row `min_win_rate` clearly beats the current 71.29%/512-row candidate on 256-row triage should retained checkpoints be promoted to 512-row evaluation. If HL-Gauss still does not move the weak 8x8/16x16 rows, the next implementation step should be memory-stack/global-context inputs or finish/draw auxiliary heads, not another pure PPO hyperparameter sweep.

### Adaptive outcome auxiliary head

2026-06-16 新增 `train_adaptive.py --outcome-aux-weight`。它给 `AdaptivePolicyValueNetwork` 加一个 3-class auxiliary head，预测 learner 视角的 `loss/draw/win`。标签不是从未来猜测来的：trainer 对每个 rollout 逆向扫描，只给同一 rollout 内已经出现 terminal/truncated transition 的 episode segment 打标签；rollout 末尾仍未结束的 segment 权重为 0。这样 outcome auxiliary 提供 finish/draw 表征信号，但不改变 PPO reward，也不把未知未来硬塞进训练。

新的 checkpoint 如果带 outcome head，评估时要传 `--outcome-head` 才能按正确 Equinox tree 加载。

下一轮 GPU triage 从当前 71.29% search-distill candidate 启动，保留 HL-Gauss/per-size value，但额外加入 outcome auxiliary：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/train_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 16384 \
  --num-steps 256 \
  --num-iterations 40 \
  --num-epochs 1 \
  --minibatch-size 1024 \
  --lr 0.000003 \
  --opponent expander \
  --learner-player mixed \
  --reward-mode terminal \
  --terminal-reward-scale 1.0 \
  --gamma 1.0 \
  --gae-lambda 0.9 \
  --top-advantage-fraction 0.25 \
  --ema-decay 0.999 \
  --eval-ema \
  --value-heads per-size \
  --init-value-heads shared \
  --value-loss hl-gauss \
  --init-value-loss mse \
  --value-bins 128 \
  --value-sigma 0.04 \
  --outcome-aux-weight 0.05 \
  --init-model-path /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx \
  --checkpoint-dir /tmp/generals-adaptive-ppo-v3-outcome-ckpts \
  --checkpoint-every 10 \
  --keep-checkpoints 4 \
  --model-path /tmp/generals-adaptive-ppo-v3-outcome.eqx \
  --seed 68000
```

256 games/row triage command:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/evaluate_adaptive_policy.py /tmp/generals-adaptive-ppo-v3-outcome.eqx \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --num-games 256 \
  --max-steps 750 \
  --opponent expander \
  --policy-mode sample \
  --map-generator generated \
  --value-heads per-size \
  --value-loss hl-gauss \
  --value-bins 128 \
  --value-sigma 0.04 \
  --outcome-head \
  --json-output /tmp/generals-adaptive-ppo-v3-outcome-eval256.json \
  --seed 68030
```

GPU smoke result:

```text
model: /tmp/generals-adaptive-ppo-v3-outcome-smoke.eqx
base: /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
config: 64 envs, num_steps=32, num_iterations=2, minibatch_size=512,
        value_heads=per-size, value_loss=hl-gauss, outcome_aux_weight=0.05
train log:
  iter 1: loss=16.3944, episodes=0, wins=0, draws=0, SPS=339
  iter 2: loss=16.5487, episodes=3, wins=3, draws=0, SPS=16970
16 games/row evaluator smoke, seed 68010:
  8p0 56.25%, 8p1 75.00%, 12p0 31.25%, 12p1 50.00%, 16p0 6.25%, 16p1 31.25%
  min_win_rate = 6.25%
```

The smoke only verifies that scalar checkpoints can warm-start into an outcome-head model, the CUDA trainer can update/save it, and `evaluate_adaptive_policy.py --outcome-head` can load it. Its 16-row evaluation is intentionally too small and truncated at 300 steps, so it is not a strength result.

Outcome auxiliary GPU triage:

```text
model: /tmp/generals-adaptive-ppo-v3-outcome.eqx
config: outcome_aux_weight=0.05, 256 envs, num_steps=256, num_iterations=40
train log:
  iter 1:  loss=16.0880, episodes=87,  wins=68, draws=0,  SPS=9391
  iter 10: loss=10.0618, episodes=161, wins=98, draws=23, SPS=79565
  iter 20: loss=6.3880,  episodes=132, wins=47, draws=38, SPS=78496
  iter 30: loss=5.0360,  episodes=143, wins=33, draws=41, SPS=77142
  iter 40: loss=4.2419,  episodes=145, wins=33, draws=39, SPS=78720
256 games/row eval, seed 68030:
  iter10 min = 66.41%
  iter20 min = 64.84%
  iter30 min = 67.58%
  iter40/final min = 65.62%
```

`0.05` is too strong and damages 16x16 seat stability. A lower-weight probe looked better at 256-row but failed promotion:

```text
model: /tmp/generals-adaptive-ppo-v3-outcome-x005.eqx
config: outcome_aux_weight=0.005, 256 envs, num_steps=256, num_iterations=20
train log:
  iter 1:  loss=13.4302, episodes=99,  wins=71,  draws=0,  SPS=9376
  iter 10: loss=8.1656,  episodes=147, wins=100, draws=19, SPS=77645
  iter 20: loss=5.6662,  episodes=149, wins=67,  draws=29, SPS=78331
256 games/row eval, seed 68130:
  same-seed scalar base min = 70.31%
  iter10 min = 71.88%
  iter20/final min = 68.75%
512 games/row promotion eval for iter10, seed 68140:
  8p0 73.63%, 8p1 69.14%, 12p0 78.91%, 12p1 80.66%, 16p0 72.85%, 16p1 68.95%
  min_win_rate = 68.95%
```

结论更新：outcome auxiliary infrastructure is useful for future representation work, but the simple rollout-local loss is not enough by itself. Weight `0.05` overpowers PPO and hurts 16x16; weight `0.005` produced a 256-row blip but failed 512-row validation. Do not continue sweeping this exact auxiliary loss. The next aligned step is to change what the network can observe or what the teacher provides: memory/global context channels, scoreboard-history tokens, or search-to-Q/intent distillation.

### Adaptive global context branch

2026-06-16 新增 `train_adaptive.py --global-context` 和 `evaluate_adaptive_policy.py --global-context`。开启后，`adaptive_obs_to_array` 从 15 通道扩展到 20 通道：

```text
0-8:   原 fogged observation spatial planes
9-14:  active/padding/row/col/size/area adaptive planes
15-19: normalized own land, own army, opponent land, opponent army, timestep
```

`AdaptivePolicyValueNetwork` 同时新增一个很小的 global-context MLP。它从 active cells 上的 `size/area/scoreboard/time` 取 7 维均值，投到 conv4 feature map；第二层零初始化，所以从旧 15 通道 checkpoint 扩展时初始 policy logits/value 与源 checkpoint 对齐。旧 checkpoint warm start 推荐传：

```bash
--global-context \
--init-input-channels 15
```

如果加载已经带 global context 的 checkpoint 继续训练，才额外传 `--init-global-context`。评估 global checkpoint 时必须传 `--global-context`，否则 Equinox tree 和输入通道模板会不匹配。

CUDA smoke 已通过：

```text
model: /tmp/generals-adaptive-ppo-v3-global-smoke.eqx
base: /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
config: 64 envs, num_steps=32, num_iterations=2, mixed learner, terminal reward,
        ema_decay=0.999, eval_ema, value_heads=per-size, value_loss=hl-gauss,
        global_context=True, init_input_channels=15
device: cuda:0
```

对应 `evaluate_adaptive_policy.py --global-context --value-heads per-size --value-loss hl-gauss` 也能在 CUDA 上加载并运行。该 smoke 使用 8 games/row 和 64 max steps，只验证训练/保存/加载链路，不作为强度结果。

下一轮 GPU triage 应从当前 71.29% search-distill candidate 启动，保留 v3-noarch 控制项和 HL-Gauss/per-size value，只加 global context：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/train_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 4096 \
  --num-steps 256 \
  --num-iterations 40 \
  --num-epochs 1 \
  --minibatch-size 1024 \
  --lr 0.000003 \
  --opponent expander \
  --learner-player mixed \
  --reward-mode terminal \
  --terminal-reward-scale 1.0 \
  --truncation-reward-scale 0.0 \
  --gamma 1.0 \
  --gae-lambda 0.9 \
  --top-advantage-fraction 0.25 \
  --ema-decay 0.999 \
  --eval-ema \
  --global-context \
  --init-input-channels 15 \
  --value-heads per-size \
  --init-value-heads shared \
  --value-loss hl-gauss \
  --init-value-loss mse \
  --value-bins 128 \
  --value-sigma 0.04 \
  --init-model-path /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx \
  --checkpoint-dir /tmp/generals-adaptive-ppo-v3-global-ckpts \
  --checkpoint-every 10 \
  --keep-checkpoints 4 \
  --model-path /tmp/generals-adaptive-ppo-v3-global.eqx \
  --seed 69030
```

GPU triage result, 256 games/row, seed 69040:

```text
source:
  checkpoint = /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
  rows = 8p0 74.22%, 8p1 73.05%, 12p0 84.77%, 12p1 82.81%, 16p0 73.83%, 16p1 64.45%
  min = 64.45%

global iter10:
  rows = 8p0 72.27%, 8p1 74.61%, 12p0 84.38%, 12p1 80.86%, 16p0 75.39%, 16p1 69.14%
  min = 69.14%

global iter20:
  rows = 8p0 68.36%, 8p1 73.83%, 12p0 84.77%, 12p1 77.34%, 16p0 74.22%, 16p1 64.06%
  min = 64.06%

global iter30:
  rows = 8p0 68.36%, 8p1 72.27%, 12p0 85.16%, 12p1 79.30%, 16p0 73.83%, 16p1 65.62%
  min = 65.62%

global iter40/final:
  rows = 8p0 71.48%, 8p1 69.92%, 12p0 83.20%, 12p1 77.73%, 16p0 73.05%, 16p1 68.36%
  min = 68.36%
```

结论更新：global-context branch is trainable and improves the same-seed 16p1 weak row versus the source checkpoint, but it does not beat the existing 512-row `71.29%` search-distill candidate and should not be promoted. The later PPO checkpoints again show drift: value loss falls while 8x8 and 16x16 seat balance degrades. The next useful step should preserve this representation path but add either scoreboard history/memory channels, search-to-Q/intent targets, or a lower-risk distillation objective that can train the global branch without destabilizing the old policy.

### Adaptive scoreboard history branch

2026-06-16 新增 `--scoreboard-history`。它在 `--global-context` 的 20 通道基础上追加 10 个通道：

```text
20-24: previous normalized own land, own army, opponent land, opponent army, timestep
25-29: current - previous one-step deltas for the same five features
```

训练时每个 vectorized environment carry 一份 previous scoreboard feature。episode done/truncated 后该 row 清零，避免新 reset 局面继承上一局的 scoreboard 历史。评估时也在 `lax.scan` carry history，因此 `evaluate_adaptive_policy.py --scoreboard-history` 会使用同样的 30 通道输入模板。

从旧 15 通道 checkpoint warm start 的命令仍然是：

```bash
--scoreboard-history \
--init-input-channels 15
```

如果从已经带 global context 的 20 通道 checkpoint 扩到 history，则需要显式传 `--init-global-context --init-input-channels 20`。如果继续训练已经带 history 的 30 通道 checkpoint，不传 `--init-input-channels` 即可。

CUDA smoke 已通过：

```text
model: /tmp/generals-adaptive-ppo-v3-history-smoke.eqx
base: /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
config: 64 envs, num_steps=32, num_iterations=2, mixed learner, terminal reward,
        ema_decay=0.999, eval_ema, value_heads=per-size, value_loss=hl-gauss,
        scoreboard_history=True, init_input_channels=15
device: cuda:0
```

Focused verification used `tests/test_adaptive_ppo.py` (`43 passed`), compileall, `git diff --check`, and CUDA train/eval smoke. Full pytest was intentionally interrupted after 53 passing tests to keep iteration speed high; this branch should be judged by GPU training feedback.

Next fast triage:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/train_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 4096 \
  --num-steps 256 \
  --num-iterations 40 \
  --num-epochs 1 \
  --minibatch-size 1024 \
  --lr 0.000003 \
  --opponent expander \
  --learner-player mixed \
  --reward-mode terminal \
  --terminal-reward-scale 1.0 \
  --truncation-reward-scale 0.0 \
  --gamma 1.0 \
  --gae-lambda 0.9 \
  --top-advantage-fraction 0.25 \
  --ema-decay 0.999 \
  --eval-ema \
  --scoreboard-history \
  --init-input-channels 15 \
  --value-heads per-size \
  --init-value-heads shared \
  --value-loss hl-gauss \
  --init-value-loss mse \
  --value-bins 128 \
  --value-sigma 0.04 \
  --init-model-path /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx \
  --checkpoint-dir /tmp/generals-adaptive-ppo-v3-history-ckpts \
  --checkpoint-every 10 \
  --keep-checkpoints 4 \
  --model-path /tmp/generals-adaptive-ppo-v3-history.eqx \
  --seed 70030
```

GPU triage result:

```text
256 envs, minibatch 1024: OOM in train_minibatch_step while allocating 1.88 GiB.
256 envs, minibatch 512: same OOM.
128 envs, minibatch 512: completed at about 46k SPS.

source, seed 70040:
  rows = 8p0 69.14%, 8p1 77.34%, 12p0 79.69%, 12p1 74.22%, 16p0 71.88%, 16p1 68.75%
  min = 68.75%

history iter10:
  rows = 8p0 71.48%, 8p1 77.73%, 12p0 83.98%, 12p1 79.30%, 16p0 68.36%, 16p1 71.48%
  min = 68.36%

history iter20:
  rows = 8p0 70.31%, 8p1 75.78%, 12p0 79.69%, 12p1 78.12%, 16p0 71.88%, 16p1 68.36%
  min = 68.36%

history iter30:
  rows = 8p0 67.19%, 8p1 75.78%, 12p0 80.86%, 12p1 82.03%, 16p0 69.92%, 16p1 69.92%
  min = 67.19%

history iter40/final:
  rows = 8p0 73.83%, 8p1 76.95%, 12p0 79.69%, 12p1 78.52%, 16p0 70.31%, 16p1 71.09%
  min = 70.31%
```

结论更新：scoreboard history has useful signal on this seed, improving source min from 68.75% to 70.31%, but it still does not beat the existing 512-row 71.29% candidate. It also shifts weakness between 16p0/16p1 and does not fix PPO drift. Do not promote to 512-row validation. The next fast follow-up should reduce policy drift: either lower LR/shorter continuation for the history model, or train the new history/global branch with distillation/replay instead of full PPO updates.

Lower-LR history follow-up, `lr=1e-6`, 128 envs, 20 iterations, seed 70130 train / 70140 eval:

```text
source:
  rows = 8p0 73.44%, 8p1 71.48%, 12p0 81.25%, 12p1 84.77%, 16p0 66.02%, 16p1 68.75%
  min = 66.02%

history lr1e-6 iter10:
  rows = 8p0 75.39%, 8p1 70.31%, 12p0 83.20%, 12p1 83.20%, 16p0 66.41%, 16p1 68.36%
  min = 66.41%

history lr1e-6 iter20/final:
  rows = 8p0 75.78%, 8p1 71.48%, 12p0 84.38%, 12p1 83.59%, 16p0 68.75%, 16p1 70.31%
  min = 68.75%
```

结论更新：lower LR reduces the visible training-collapse pattern and improves the same-seed weak 16p0 row, but it still does not approach the current promotion bar. Plain PPO continuation, even with history inputs, remains too unstable and too low-signal. The next implementation target should be a distillation/replay path that can train the added global/history representation while anchoring the old policy, instead of applying full PPO pressure to every shared parameter.

### Adaptive scoreboard history distillation

2026-06-16 `adaptive_search_distill.py` now supports `--global-context` and `--scoreboard-history` for the student network. The design deliberately keeps `base_network`, rollout search, and opponent execution on the old 15-channel template, while the student receives 20/30-channel observations and a per-env scoreboard-history carry. This lets the new global/history branch learn from KL/search targets without turning the frozen teacher into a moving architecture target.

Warm-start from the current 15-channel best remains:

```bash
--scoreboard-history \
--init-input-channels 15
```

CUDA smoke passed on `cuda:0`:

```text
student/base: /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx
config: 16 envs, num_steps=1, num_iterations=2, top_k=2,
        rollout_steps=1, rollouts_per_action=1, learner_player=1,
        target_mode=soft, soft_weight_mode=active, scoreboard_history=True
result: saved /tmp/generals-adaptive-history-distill-smoke.eqx
```

Single-seat p0/p1 probes improved the weak same-seed 16p1 row slightly but still shifted weakness between seats and degraded by iter20. The next probe should therefore use `--learner-player mixed`, so p0 and p1 search labels enter the same optimizer update.

2026-06-16 follow-up added `--freeze-legacy-weights` for distillation. When used with global/history inputs, gradients are kept only for conv1 weights connected to channels after the legacy 15-channel observation plus the global/history MLP. The old conv trunk and policy/value heads stay frozen, so the history branch can learn additive corrections without dragging the existing 15-channel policy away from the 71.29% anchor.

2026-06-16 follow-up added `--search-value-weight` and `--search-value-scale`. The teacher target is `tanh(max_topk_search_score / scale)`, weighted on active samples. This is a first Q/value-style teacher signal: rollout search can now supervise the scalar value/shared representation instead of only a single-step action distribution. Default weight is `0.0`, preserving previous action-CE distillation runs.

2026-06-16 follow-up added `--search-outcome-weight`. The trainer now tracks the outcome class from the best rollout-search candidate: terminal win/loss is mapped from the learner perspective, and non-terminal within the rollout horizon is mapped to the draw/unfinished class. This uses the existing adaptive outcome head and gives search a direct finish/draw teaching channel. New checkpoints trained with this flag need `evaluate_adaptive_policy.py --outcome-head` when evaluated.

Next fast GPU probe:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/adaptive_search_distill.py 64 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 1024 \
  --base-model-path /tmp/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx \
  --model-path /tmp/generals-adaptive-history-distill-mixed-v1.eqx \
  --target-mode soft \
  --soft-weight-mode active \
  --soft-improvement-extra-weight 0.02 \
  --search-value-weight 0.1 \
  --search-value-scale 100 \
  --search-outcome-weight 0.1 \
  --learner-player mixed \
  --num-steps 8 \
  --num-iterations 20 \
  --num-epochs 1 \
  --minibatch-size 512 \
  --lr 0.000001 \
  --top-k 4 \
  --rollout-steps 8 \
  --rollouts-per-action 1 \
  --channels 32,32,32,16 \
  --base-channels 32,32,32,16 \
  --init-channels 32,32,32,16 \
  --scoreboard-history \
  --init-input-channels 15 \
  --freeze-legacy-weights \
  --checkpoint-dir /tmp/generals-adaptive-history-distill-mixed-v1-ckpts \
  --checkpoint-every 10 \
  --keep-checkpoints 3 \
  --seed 71100
```

GPU triage on eval seed 71140:

```text
source / current 71.29%-512 candidate:
  rows = 8p0 73.05%, 8p1 71.88%, 12p0 82.03%, 12p1 80.47%, 16p0 70.31%, 16p1 64.84%
  min = 64.84%

history p1-only, lr=1e-6:
  iter10 rows = 8p0 71.88%, 8p1 75.78%, 12p0 80.86%, 12p1 82.81%, 16p0 68.36%, 16p1 66.80%
  iter10 min = 66.80%
  iter20 min = 64.45%

history p0-only, lr=1e-6:
  iter10 rows = 8p0 71.09%, 8p1 75.78%, 12p0 80.86%, 12p1 82.03%, 16p0 67.97%, 16p1 67.19%
  iter10 min = 67.19%
  iter20 min = 66.02%

history mixed-seat, lr=1e-6:
  iter10 rows = 8p0 71.48%, 8p1 76.17%, 12p0 80.08%, 12p1 83.98%, 16p0 69.14%, 16p1 65.62%
  iter10 min = 65.62%
  iter20 min = 61.33%

history mixed-seat freeze-legacy, lr=1e-6:
  iter10 min = 65.23%
  iter20 min = 64.06%

history mixed-seat freeze-legacy, lr=1e-4:
  iter10 rows = 8p0 70.31%, 8p1 73.44%, 12p0 80.47%, 12p1 80.08%, 16p0 68.75%, 16p1 67.19%
  iter10 min = 67.19%
  iter20 min = 64.84%
```

结论更新：history/global channels can be wired into search distillation, but action-level active-soft search CE still does not transfer the useful long-horizon signal. Freezing the legacy trunk prevents collapse but mostly preserves the old policy; raising LR lets the new branch move, yet the best 256-row min is still only `67.19%`, far below the current `71.29%`/512 candidate. Do not promote any history-distill checkpoint. Stop this search-CE/history sweep; the next useful implementation should train value/finish/Q/intent targets from search or full-state outcomes, not more single-action KL/CE variants.

Search-value distillation follow-up:

```text
mixed history, search_value_weight=0.1, scale=100, improve_weight=0.05, extra=0.02:
  train: value loss fell from 0.0160 to 0.0071, KL stayed <= 0.00007.
  iter10 rows = 8p0 71.88%, 8p1 74.22%, 12p0 79.69%, 12p1 83.20%, 16p0 69.14%, 16p1 67.19%
  iter10 min = 67.19%
  iter20 rows = 8p0 73.05%, 8p1 75.39%, 12p0 80.86%, 12p1 82.81%, 16p0 68.36%, 16p1 71.48%
  iter20 min = 68.36%

p0 continuation from mixed search-value iter20, lr=5e-7:
  iter10 rows = 8p0 73.05%, 8p1 73.05%, 12p0 82.03%, 12p1 83.59%, 16p0 69.14%, 16p1 67.58%
  iter10 min = 67.58%

value-first mixed history, search_value_weight=0.2, scale=20, improve_weight=0.02, extra=0:
  train: value loss fell from 0.0132 to 0.0077, KL stayed <= 0.00005.
  iter10 min = 67.58%
  iter20 rows = 8p0 72.27%, 8p1 75.39%, 12p0 80.47%, 12p1 80.86%, 16p0 71.09%, 16p1 67.97%
  iter20 min = 67.97%
```

结论更新：search-value supervision is learnable and improves the weak 16p1 row in one configuration, but it still shifts the bottleneck to 16p0/16p1 rather than raising the six-row floor. No search-value checkpoint beats the current `71.29%`/512 candidate. The next implementation should use an explicit finish/draw/outcome target from actual rollout terminal status, because the remaining 16x16 problem is dominated by high draw rate and failure to convert decisive positions before timeout.

Search-outcome distillation follow-up:

```text
mixed history, search_value_weight=0.1, search_outcome_weight=0.1, lr=1e-6:
  train: outcome loss stayed high around 2.8-3.0 and KL stayed <= 0.00015.
  iter10 rows = 8p0 72.27%, 8p1 76.56%, 12p0 78.52%, 12p1 81.64%, 16p0 67.58%, 16p1 61.33%
  iter10 min = 61.33%
  iter20 rows = 8p0 75.78%, 8p1 72.27%, 12p0 79.69%, 12p1 81.25%, 16p0 68.36%, 16p1 65.23%
  iter20 min = 65.23%

mixed history, freeze legacy, search_value_weight=0.1, search_outcome_weight=0.1, lr=1e-4:
  train: outcome loss fell from 0.3183 to 0.0795 and KL stayed <= 0.00003.
  iter10 rows = 8p0 70.70%, 8p1 75.78%, 12p0 82.42%, 12p1 84.77%, 16p0 70.70%, 16p1 70.31%
  iter10 min = 70.31% over 256 games/row
  iter10 512-row promotion check = 8p0 72.46%, 8p1 74.80%, 12p0 81.05%, 12p1 82.81%, 16p0 65.82%, 16p1 66.60%
  iter10 512-row min = 65.82%
  iter20 min = 66.41%

mixed history, freeze legacy, search_value_weight=0.1, search_outcome_weight=0.02, lr=1e-4:
  train: outcome loss fell from 0.6184 to 0.3536 and KL stayed <= 0.00004.
  iter10 rows = 8p0 73.05%, 8p1 69.53%, 12p0 80.47%, 12p1 84.38%, 16p0 70.31%, 16p1 68.36%
  iter10 min = 68.36%
```

结论更新：search-outcome CE is learnable only when the legacy trunk is frozen, but the 256-row `70.31%` candidate collapsed to `65.82%` on the 512-row promotion check. The current target also treats "not terminal inside the short search horizon" as draw/unfinished, so the label is dominated by horizon artifacts rather than true strategic outcome. Do not promote outcome-distill checkpoints. If this line is revisited, prefer a binary finish head or terminal-only weighted outcome labels over the current three-class best-candidate CE.

PPO v3-noarch GPU rerun after the 2026 Generals.io review:

```text
checkpoint structure probe:
  /tmp/generals-adaptive-search-distill-p1-v1-iter-000040.eqx loads as 15 input channels with no global branch.
  It fails as 30-channel global because no global_linear1/global_linear2 leaves exist.
  Therefore PPO v3 history runs are 15-channel source -> 30-channel history/global warm starts, not pure same-architecture continuations.

source control, seed 71140, 256 games/row:
  rows = 8p0 73.05%, 8p1 71.88%, 12p0 82.03%, 12p1 80.47%, 16p0 70.31%, 16p1 64.84%
  min = 64.84%

source controls already on disk:
  seed 68130 min = 70.31%
  seed 69040 min = 64.45%

terminal-only PPO, HL-Gauss per-size value, 256 env x 128 steps, EMA saved, lr=1e-5:
  train: loss fell 13.46 -> 3.19, but rollout wins fell after iter10.
  iter10 min = 64.06%
  final min = 64.45%

terminal-only PPO, shared MSE value, 128 env x 256 steps, EMA saved, lr=3e-6:
  256 env x 256 steps OOMed during batch flatten/shuffle even with minibatch 512 on the 16GB RTX 5070 Ti.
  128 env x 256 steps ran around 49k SPS.
  final seed71140 min = 64.84%
  final seed68130 min = 70.31%

terminal-only PPO, shared MSE value, 128 env x 256 steps, last iterate saved, lr=3e-6:
  seed71140 rows = 8p0 72.27%, 8p1 69.92%, 12p0 80.47%, 12p1 82.42%, 16p0 66.80%, 16p1 69.14%
  seed71140 min = 66.80%
  seed68130 rows = 8p0 75.39%, 8p1 69.92%, 12p0 83.59%, 12p1 82.03%, 16p0 68.36%, 16p1 71.48%
  seed68130 min = 68.36%

composite+terminal PPO, shared MSE value, 128 env x 256 steps, last iterate saved, lr=3e-6:
  seed71140 rows = 8p0 74.22%, 8p1 76.17%, 12p0 82.03%, 12p1 83.20%, 16p0 72.27%, 16p1 66.80%
  seed71140 min = 66.80%
  seed68130 rows = 8p0 72.66%, 8p1 71.09%, 12p0 82.42%, 12p1 87.50%, 16p0 68.75%, 16p1 76.17%
  seed68130 min = 68.75%
  seed69040 rows = 8p0 72.27%, 8p1 69.92%, 12p0 85.16%, 12p1 76.95%, 16p0 72.27%, 16p1 72.66%
  seed69040 min = 69.92%

balanced composite PPO, size weights 8:2,12:1,16:2, top_advantage_fraction=0.5:
  seed71140 min = 69.14%
  seed68130 min = 67.58%
  seed69040 min = 66.41%
```

结论更新：PPO v3-noarch 的机制已经可用，但在当前小 CNN 上继续调 reward/value/EMA 只是在迁移 weak row。Composite+terminal can repair weak 16 rows on some seeds, especially seed69040 (`64.45% -> 69.92%`), but it creates new 8p1/12p1 or 8p0 bottlenecks and never beats the current `71.29%`/512 candidate. `ema_decay=0.999` is too slow for 20-30 iteration probes; short triage should save both last iterate and EMA or use lower decay. The next useful implementation is true seat x size stratified PPO batches or gradient-conflict mitigation, followed by memory/Transformer architecture work. Do not spend more GPU time on plain noarch PPO reward sweeps unless the batch construction changes.

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

## 2026-06-17 Strategy Aux Phase 1 转向

根据前面 adaptive 结果，plain PPO continuation、global/history 输入、search-value/outcome、truncation、扩容、weighted/alternate 都没有稳定突破 `71.29%` 六行 minimum。下一轮不继续扫 PPO 权重，转向 Phase 1：保留 primitive policy/value 头，给 adaptive search distillation 加 intent/Q/finish/belief auxiliary heads，让 search teacher 和 full `GameState` 先教“局面表示”，少教单步 argmax 动作。

本轮实现内容：

- `AdaptivePolicyValueNetwork(strategy_aux=True)` 新增四类辅助输出：8-way intent、finish binary、enemy-general belief heatmap、active-action Q values。
- `adaptive_strategy_aux.py` 生成弱 intent 标签和 full-state enemy general one-hot belief 标签；finish 来自 top-k search rollout outcome。
- `adaptive_search_distill.py` 新增 `--strategy-q-weight`、`--strategy-intent-weight`、`--strategy-finish-weight`、`--strategy-belief-weight` 和 `--init-strategy-aux`。训练 batch 现在额外保存 top-k search score Q target、intent、finish、enemy-general heatmap，并在 soft distill loss 中混合这些 auxiliary losses。
- `evaluate_adaptive_policy.py --strategy-aux` 可加载带辅助头的 checkpoint；评估仍只使用旧 policy/value 行为。
- 新训练输出默认落在项目内 `runs/`，并且 `runs/` 已加入 `.gitignore`。后续不要把模型写到 `/tmp` 或缓存目录。

CUDA smoke 使用合成 4/6 padded 小模型，只验证训练、保存、加载和 auxiliary loss 链路：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/adaptive_search_distill.py 8 \
  --grid-sizes 4,6 \
  --grid-size-weights 4:1,6:1 \
  --pad-to 6 \
  --map-generator simple \
  --pool-size 32 \
  --base-model-path runs/strategy-aux-smoke/adaptive-base.eqx \
  --model-path runs/strategy-aux-smoke-postpatch/generals-adaptive-strategy-aux-smoke.eqx \
  --target-mode soft \
  --soft-weight-mode active \
  --soft-improvement-extra-weight 0.0 \
  --search-value-weight 0.0 \
  --search-outcome-weight 0.0 \
  --strategy-q-weight 0.1 \
  --strategy-intent-weight 0.05 \
  --strategy-finish-weight 0.05 \
  --strategy-belief-weight 0.02 \
  --learner-player mixed \
  --num-steps 1 \
  --num-iterations 1 \
  --num-epochs 1 \
  --minibatch-size 8 \
  --top-k 2 \
  --rollout-steps 1 \
  --rollouts-per-action 1 \
  --channels 16,16,16,8 \
  --base-channels 16,16,16,8 \
  --init-channels 16,16,16,8 \
  --scoreboard-history \
  --init-input-channels 15 \
  --freeze-legacy-weights \
  --checkpoint-dir runs/strategy-aux-smoke-postpatch/ckpts \
  --checkpoint-every 1 \
  --keep-checkpoints 1 \
  --seed 73102
```

Smoke result on `cuda:0`:

```text
iter 1: loss=0.19085, StratQ=0.4628, Intent=2.0550, Belief=0.5982
saved: runs/strategy-aux-smoke-postpatch/generals-adaptive-strategy-aux-smoke.eqx
```

Load/eval smoke:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/evaluate_adaptive_policy.py \
  runs/strategy-aux-smoke-postpatch/generals-adaptive-strategy-aux-smoke.eqx \
  --grid-sizes 4,6 \
  --pad-to 6 \
  --num-games 4 \
  --max-steps 30 \
  --opponent expander \
  --policy-mode sample \
  --map-generator simple \
  --scoreboard-history \
  --strategy-aux \
  --channels 16,16,16,8 \
  --json-output runs/strategy-aux-smoke-postpatch/eval-smoke.json \
  --seed 73103
```

注意：这不是强度结果。当前工作区里历史 best adaptive checkpoint 已不在原 `/tmp` 路径，尝试从 `generals-adaptive-search-distill-p1-v1-iter-000040.eqx` warm start 会失败。因此本轮只完成 Phase 1 训练机制转向和 CUDA smoke。下一步应把真实 71.29% 候选迁移到 `runs/` 后，启动 mixed-seat scoreboard-history strategy-aux distill；若 256 games/row triage 超过当前 71.29%/512-row 候选，再做 512-row promotion。

## 2026-06-17 History-Base Strategy Aux Probe

用户提示检查 `legacymodels/` 后，结论是：目录里没有 exact historical best `generals-adaptive-search-distill-p1-v1-iter-000040.eqx`，但有多组 30-channel scoreboard-history adaptive probe。文件结构按 checkpoint 大小可分为：

- `147880` bytes：30-channel global/history shared scalar value，无 outcome/strategy aux。
- `148916` bytes：30-channel global/history shared scalar value + outcome head。
- `309684` bytes：30-channel per-size HL-Gauss value head。

当前 `adaptive_search_distill.py` 只给 base/search/opponent teacher 增加了 20/30-channel shared-MSE 模板支持：`--base-global-context` / `--base-scoreboard-history`。这足够加载 `generals-adaptive-ppo-v3-composite-balanced-probe1.eqx` 这类 shared-MSE history checkpoint；outcome 或 HL-Gauss teacher 还需要额外的 base template 参数，暂未加入。

64 games/row CUDA quick rank，generated maps，`--scoreboard-history`，seed `74000`：

| candidate | min win rate | bottleneck |
| --- | ---: | --- |
| `generals-adaptive-ppo-v3-composite-balanced-probe1.eqx` | `70.31%` | 16p0/16p1 |
| `generals-adaptive-ppo-v3-noarch-probe1.eqx` | `70.31%` | per-size HL, not usable as current base template |
| `generals-adaptive-search-value-mixed-v1-iter-000020.eqx` | `68.75%` | 16p0 |
| `generals-adaptive-search-outcome-freeze-v1-iter-000010.eqx` | `65.62%` | 16p0 |
| `generals-adaptive-ppo-v3-mse-last-probe1.eqx` | `65.62%` | 16p0 |
| `generals-adaptive-ppo-v3-composite-last-probe1.eqx` | `60.94%` | 16p0 |

因此本轮选择 `legacymodels/generals-adaptive-ppo-v3-composite-balanced-probe1.eqx` 作为 history-base teacher。新增实现点：

- search prior、KL base logits、opponent execution 都能按 `--base-scoreboard-history` 构造 30-channel 输入。
- search rollout 内为 p0/p1 分别 carry previous scoreboard，候选第一步后的 rollout 会继续使用一致的 one-step delta history。
- 新增 focused collector test，覆盖 student/base 都是 30-channel history 网络时的 soft batch shape。

CUDA smoke:

```text
runs/history-base-strategy-smoke/generals-adaptive-strategy-history-base-smoke.eqx
Iter 1 | Loss 65.83665 | StratQ 643.7910 | Intent 23.6869 | Belief 6.7736
```

这只验证 30-channel base/search teacher 可以在 GPU 上编译、采样、保存，不作为强度结果。

随后从 `composite-balanced` 跑两个短 strategy-aux probe，输出均在项目内 `runs/`，没有写缓存目录：

| run | main change | seed 74000 min | seed 74210 min | conclusion |
| --- | --- | ---: | ---: | --- |
| `runs/adaptive-strategy-aux-v1/` | action soft + Q/intent/finish/belief, `lr=1e-6` | `62.50%` final, `65.62%` iter4 | `64.06%` final | action soft target still shifts 16x16; not promote |
| `runs/adaptive-strategy-aux-v2/` | stronger KL, no improve CE, lighter Q, `lr=5e-7` | `68.75%` final | `64.06%` final | more stable than v1, still below base seed74000 `70.31%`; not promote |

v1 same-seed `74210` did improve the base control from `59.38%` min to `64.06%`, and v2 kept most rows closer to the base. But neither crosses the 71.29% historical bar, and neither is stronger than the `composite-balanced` seed74000 control. The useful result is engineering rather than promotion: history-base strategy auxiliary training is now runnable, and the next iteration should avoid all-active action soft labels. Recommended next probe:

```text
use shared-MSE history base:
  --base-scoreboard-history
  --scoreboard-history
  --learner-player mixed
  --kl-weight high
  --improve-weight 0
  Q/intent/finish/belief on

change target sampling:
  train action CE only on high-margin rows, or disable action CE entirely
  keep Q/intent/belief as representation losses

optional:
  add base value-head/outcome/HL-Gauss template flags so noarch_hl can be used as teacher
```

### Follow-up: high-margin, spawn curriculum, and aux-only pretrain

2026-06-17 继续跑三组 GPU probes，目标是验证上一节推荐的“不要 all-active action CE，只保留 high-margin action rows + Q/intent/belief”的判断。

`v3` 使用同一个 `composite-balanced` history base，关闭 all-active search CE：

```text
--kl-weight 2.0
--improve-weight 0.0
--soft-improvement-extra-weight 0.02
--strategy-q-weight 0.0005
--strategy-intent-weight 0.02
--strategy-finish-weight 0.02
--strategy-belief-weight 0.01
--lr 5e-7
```

64 games/row seed `74000` 最佳为 iter8，min `70.31%`；seed `74210` min `67.19%`。随后对 iter8 做 256 games/row same-seed 对照：

| checkpoint | 256-row min | bottleneck | note |
| --- | ---: | --- | --- |
| `composite-balanced` base | `66.80%` | 16p0 | draw `19.53%` |
| `v3 iter8` | `67.19%` | 16p0 | draw `22.66%` |

v3 对 12p0 有明显改善（`+3.52%`），但 16p1 和 draw rate 变差，整体只比 base 多 `+0.39%` min，不 promotion。

`v4-spawn8` 在 distill data 上加 `--max-generals-distance 8`，评估仍使用完整 generated maps。结果反而更差：seed `74000` 最佳 min 只有 `67.19%`，iter8 掉到 `57.81%` 16p0。结论：固定 close-spawn curriculum 会造成 full-distribution mismatch；未来如果使用 spawn curriculum，需要 schedule 从小 cap 逐步放开，而不是单段 close-spawn distill。

`v5` 新增 `--freeze-strategy-aux-only`，第一段只训练 strategy auxiliary heads，第二段从该 checkpoint 解冻低 LR fine-tune。实现上这个 flag 会把 policy/trunk/global/value gradients 置零，只保留 intent/finish/Q/belief heads。GPU 结果没有提升：fine-tune seed `74000` 最佳 min 为 `67.19%`，低于 v3 iter8。

当前判断：

- high-margin-only action CE 明显比 all-active action CE 稳，但仍不足以突破 16x draw/finish 瓶颈。
- close-spawn 单段训练不泛化到完整 generated maps。
- aux-head-only pretrain 可以降低 fine-tune loss，但没有转化为 policy strength。

下一步不应继续扫这三个权重。更有信息量的方向是：

```text
1. 把 search/intent/belief 做成离线 replay，先训练表示，再用 PPO/finish reward 做长 rollout 更新。
2. 给 adaptive_search_distill.py 增加 base value-head/outcome/HL-Gauss 模板只作为 teacher 对照，但 noarch_hl seed74210 只有 60.94% min，优先级低。
3. 转 Worker 预训练：用目标 heatmap + BFS/path-assignment 训练执行层，直接攻击 16x draw/finish 的执行问题。
```

## 2026-06-17 Worker BFS Pretraining

本轮按 Commander/Worker 路线先做 Worker，不接 Commander。新增 `examples/_experimental/ppo/adaptive_worker_pretrain.py`：

- 复用 `AdaptivePolicyValueNetwork`，但输入改为 18 通道：原 adaptive 15 通道 + target heatmap + eligible source heatmap + BFS route potential。
- 数据来自 Expander-vs-Expander generated-map rollouts，同时为 p0/p1 生成 Worker 标签。
- Worker target family 支持 `general` / `city` / `frontier` / `random`。第一轮重点看 `general`，因为当前 adaptive 平台主要卡在 16x draw/finish。
- 标签来自 full `GameState` 的 passable shortest-path distance：所有让距离下降的 valid moves 组成 soft target distribution，top label 仅用于指标。

调试结论：硬 one-hot top action CE 不合适。诊断显示 Worker labels 100% 都在 legal mask 内，但 top label share 只有约 `2.1%`，大量源格/方向是近似等价的路径推进动作。因此 v1 hard-label run 的 exact accuracy 低不是 index/mask bug，而是目标过碎。改成 soft BFS-progress target 后，新增两个指标：

```text
Useful: predicted action has nonzero teacher probability, i.e. it is any BFS-progress move
Mass:   teacher probability assigned to the predicted action
```

GPU runs under `runs/`:

| run | target | objective | result |
| --- | --- | --- | --- |
| `adaptive-worker-pretrain-v1` | random | hard top-label CE | top accuracy fell to `13.1%`; not useful |
| `adaptive-worker-pretrain-v2` | random | soft BFS-progress CE | top accuracy around `11-14%`; mixed target families remain broad |
| `adaptive-worker-pretrain-general-v1` | general | soft CE, no Useful metric yet | top accuracy reached `19.3%` transiently |
| `adaptive-worker-pretrain-general-v2` | general | soft CE + Useful/Mass metrics | Useful rose from `7.1%` to `62.7%` over 100 iters |

`general-v2` final log:

```text
Iter 99 | Loss 3.7667 | Acc 9.9% | Useful 62.7% | Mass 0.089 | Valid 97.8%
```

This is not a promotion model: the Worker checkpoint expects target/source/route command channels and is not wired into `evaluate_adaptive_policy.py`. The result is a usable execution-layer pretraining path. Next engineering step is to split Worker into explicit `source_heatmap` + `direction` heads or add a thin Commander wrapper that supplies target/route channels, then evaluate whether Worker-controlled finish attempts reduce 16x draw rate.

### Worker command wrapper evaluation

2026-06-17 added `evaluate_worker_policy.py`, which supplies 18-channel Worker commands from fogged observation only:

```text
auto:
  visible enemy general if observed
  else visible/fogged city-like structures
  else frontier/fog

hybrid:
  fallback adaptive checkpoint handles normal play
  Worker can take over on always / visible-general / contact / turn trigger
```

Pure Worker with the `general-v2` checkpoint is not viable as a policy: `auto`, `frontier`, and `visible-general` all produced six-row min `0.00%` over 64 games/row on seed `74710`.

Hybrid evaluation used `composite-balanced` history checkpoint as fallback:

| mode | min win rate | conclusion |
| --- | ---: | --- |
| Worker never triggers | `64.06%` | fallback path is functional and same order as base |
| trigger on contact | `0.00%` | Worker destroys normal play after contact |
| trigger on turn 80 | `1.56%` | Worker cannot replace policy after opening |
| trigger on visible general, greedy Worker | `1.56%` | Worker fails at finish execution |
| trigger on visible general, sampled Worker | `15.62%` | less catastrophic but still far below fallback |

This validates the negative result: the current Worker single flat action head learned a broad BFS support signal (`Useful`), but it is not precise enough to take over actual moves. Next Worker iteration should not be another flat-action CE run. It should split the problem:

```text
source head:
  supervised by soft source heatmap over eligible owned cells

direction head:
  conditioned on selected/source cell, supervised by BFS-progress direction mask

execution:
  only use Worker as action reranker/candidate proposer at first
  do not hard-switch control from the adaptive policy
```

## 2026-06-17 Worker Split-Loss Probe

Implemented the first half of the split Worker branch in `adaptive_worker_pretrain.py` without changing the network schema. The flat adaptive policy logits are now marginalized into:

```text
source logits: logsumexp over the 8 move planes per source cell
direction logits: logsumexp over all source cells and full/half planes per direction
```

New CLI weights:

```text
--action-loss-weight
--source-loss-weight
--direction-loss-weight
```

Default behavior is unchanged (`action=1, source=0, direction=0`). The split probe used `action=0.1, source=1.0, direction=1.0` on GPU:

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/adaptive_worker_pretrain.py 64 \
  --grid-sizes 8,12,16 --grid-size-weights 8:1,12:1,16:2 --pad-to 16 \
  --map-generator generated --target-family general --target-temperature 2.0 \
  --num-steps 32 --num-iterations 100 --pool-size 1024 --truncation 300 \
  --lr 1e-4 --channels 32,32,32,16 \
  --action-loss-weight 0.1 --source-loss-weight 1.0 --direction-loss-weight 1.0 \
  --model-path runs/adaptive-worker-split-general-v1/generals-adaptive-worker-split-general-v1.eqx \
  --checkpoint-dir runs/adaptive-worker-split-general-v1/ckpts --checkpoint-every 50 --keep-checkpoints 2 \
  --seed 74800
```

Training result:

| run | final Acc | final Src | final Dir | final Useful | final Mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adaptive-worker-pretrain-general-v2` flat soft CE | `9.9%` | n/a | n/a | `62.7%` | `0.089` |
| `adaptive-worker-split-general-v1` split loss | `21.7%` | `40.0%` | `51.8%` | `91.0%` | `0.172` |

The split loss substantially improves supervised route-support metrics, but it is still not a promotion policy. A 64 games/row hybrid visible-general takeover check against Expander scored:

| size/seat | win/loss/draw | win rate |
| --- | ---: | ---: |
| 8x8 p0 | `18/41/5` | `28.12%` |
| 8x8 p1 | `19/39/6` | `29.69%` |
| 12x12 p0 | `23/25/16` | `35.94%` |
| 12x12 p1 | `23/16/25` | `35.94%` |
| 16x16 p0 | `15/12/37` | `23.44%` |
| 16x16 p1 | `9/10/45` | `14.06%` |

Interpretation: source/direction supervision made the Worker much better at recognizing useful BFS-support moves, but hard-switching control still corrupts the policy. Next step should use Worker logits as a small rerank bias/candidate proposer under the adaptive fallback policy, not as a replacement action source.

## 2026-06-17 Worker Rerank Probe

Added `evaluate_worker_policy.py --hybrid-mode rerank --worker-logit-scale <x>`. In rerank mode, the evaluator computes both fallback and Worker logits, centers Worker logits over legal actions, and applies them only as a bias:

```text
combined_logits = fallback_logits + scale * centered_legal_worker_logits
```

This keeps fallback policy control and tests whether the split Worker can help finish when the enemy general is visible. GPU sweep used the same split Worker checkpoint and the shared-MSE history fallback:

```text
worker:   runs/adaptive-worker-split-general-v1/generals-adaptive-worker-split-general-v1.eqx
fallback: /home/codeboy/research/generals-bots/legacymodels/generals-adaptive-ppo-v3-composite-balanced-probe1.eqx
trigger:  visible-general
mode:     sample
games:    64 per size/seat
seed:     74820
```

| scale | min win rate | bottleneck row | observation |
| ---: | ---: | --- | --- |
| `0.00` | `68.75%` | 16x16 p0 | rerank path equals fallback baseline |
| `0.05` | `68.75%` | 16x16 p0 | 16x16 p1 fell from `76.56%` to `73.44%`; draw rose |
| `0.10` | `67.19%` | 16x16 p0 | 12x16/16x16 draw damage increased |

Conclusion: the split Worker is useful as a supervised representation probe but its raw action logits should not bias the production policy yet. The next Worker branch should either expose explicit source/direction heads and use them only to generate a small candidate set, or train a dedicated rerank head against fallback-success/search-Q labels instead of reusing the Worker action logits.

## 2026-06-17 Strategy-Q Rerank Probe

Added `evaluate_adaptive_policy.py --strategy-q-rerank-scale <x>` for checkpoints loaded with `--strategy-aux`. This uses the strategy auxiliary Q head at inference:

```text
combined_logits = policy_logits + scale * centered_legal_strategy_q
```

The probe targets the question left open by the strategy-aux runs: the policy logits from v3/v5 did not promote, but the search-Q head might still contain useful candidate-ranking signal.

Tested checkpoint:

```text
runs/adaptive-strategy-aux-v3/ckpts/generals-adaptive-strategy-aux-v3-iter-000008.eqx
template: --channels 32,32,32,16 --scoreboard-history --strategy-aux
```

64 games/row sweep on seed `74900`:

| scale | min win rate | key rows |
| ---: | ---: | --- |
| `0.00` | `64.06%` | 16p0 `64.06%`, 16p1 `67.19%` |
| `0.01` | `65.62%` | 16p0 improved to `76.56%`; 8p0 bottleneck `65.62%` |
| `0.02` | `67.19%` | 8p0 `71.88%`, 16p0 `75.00%`, 16p1 bottleneck `67.19%` |
| `0.03` | `60.94%` | over-bias hurts 16p0 |

256 games/row same-seed triage on seed `74920`:

| scale | min win rate | rows summary |
| ---: | ---: | --- |
| `0.00` | `67.58%` | 16p0 and 16p1 both `67.58%` |
| `0.01` | `67.97%` | 16p0 `73.05%`, 16p1 `67.97%`; small positive but not promotion |
| `0.02` | `67.19%` | 16p0 `72.27%`, 16p1 `67.19%`; seat tradeoff returns |

Conclusion: strategy-Q inference bias is a useful diagnostic and mildly improves the best 256-row row balance at scale `0.01`, but it is nowhere near the existing `71.29%` promotion bar. The failure mode is again seat/size transfer: Q bias helps 16p0 finish/draw, then pushes weakness into 16p1 or 8p0. Next step should train/calibrate Q for inference directly, e.g. Q-rank loss on candidate set with seat/size-balanced batches, not merely reuse the high-noise auxiliary Q head.

### Strategy-Q pairwise rank loss

Added optional pairwise ranking supervision for the strategy-Q head:

```text
--strategy-q-rank-weight
--strategy-q-rank-min-margin
```

For each top-k search candidate set, this loss compares candidate pairs whose search-Q target gap exceeds the margin and applies a `softplus(-(q_i - q_j))` ranking loss when target `i` should outrank target `j`. It is intended to train inference ordering without relying on noisy absolute search score regression.

GPU probe:

```text
init:      adaptive-strategy-aux-v3 iter8
run:       runs/adaptive-strategy-q-rank-v1/
settings:  mixed seats, 8/12/16, top_k=4, rollout_steps=4, rollouts/action=1
loss:      strategy_q_weight=0.0001, strategy_q_rank_weight=0.05, rank_min_margin=0.02
freeze:    --freeze-strategy-aux-only
```

Training log signal:

```text
iter 1:  StratQ 90.0352, StratRank 0.1746
iter 24: StratQ 76.4928, StratRank 0.0381
```

Evaluation on seed `75020`, 64 games/row:

| checkpoint | rerank scale | min win rate | bottleneck |
| --- | ---: | ---: | --- |
| `strategy-q-rank-v1` | `0.01` | `64.06%` | 16p1 |
| `strategy-q-rank-v1` | `0.02` | `57.81%` | 16p1 |

Conclusion: pairwise rank loss is implemented and optimizes, but this short aux-only run overfits/miscalibrates the Q head for inference. Do not promote. The next useful Q direction would require online validation inside training, stronger target normalization, or a rerank head trained on policy-action replacement outcomes rather than raw short-horizon search scores.

## 2026-06-17 8x8 P1 League Best-Response Probe

Ran the missing player-1 fixed 8x8 checkpoint-league probe against the historical v2-v5 pool. This tests whether ordinary PPO continuation can produce a v5-beating pure checkpoint when the learner is trained from the weaker mirrored seat instead of the earlier player-0 branch.

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/train.py 512 \
  --grid-size 8 --map-generator generated \
  --mountain-density-min 0.12 --mountain-density-max 0.22 \
  --num-cities-min 4 --num-cities-max 8 --min-generals-distance 5 \
  --pool-size 16384 --num-steps 64 --num-iterations 160 \
  --num-epochs 4 --minibatch-size 4096 --lr 0.000001 \
  --truncation 500 \
  --init-model-path /home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-pool /home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v2.eqx,/home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v3.eqx,/home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v4.eqx,/home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v5.eqx \
  --opponent-policy-pool-modes sample,sample,sample,sample \
  --learner-player 1 --terminal-reward-scale 1.0 \
  --checkpoint-dir runs/8x8-league-p1-v1/ckpts --checkpoint-every 40 --keep-checkpoints 4 \
  --model-path runs/8x8-league-p1-v1/generals-ppo-8x8-league-p1-v1.eqx \
  --seed 76100
```

Training stayed noisy but did not show a decisive improvement trend. Rollout win snapshots ranged from `41%` to `58%` after warmup, with no sustained climb over the v5/pool baseline. Checkpoints were evaluated directly against frozen v5 sample mode:

| checkpoint | games/seat | seed | p0 win rate | p1 win rate | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| iter040 | 256 | 76160 | `48.83%` | `43.75%` | below v5 |
| iter080 | 256 | 76160 | `46.48%` | `48.44%` | below v5 |
| iter120 | 256 | 76160 | `46.48%` | `48.44%` | below v5 |
| iter160 | 256 | 76160 | `46.88%` | `42.58%` | below v5 |
| final | 512 | 76150 | `47.85%` | `43.55%` | below v5 |

For reference, the older augmented full-state best-response checkpoint in the repo root also failed the v5 gate under the same evaluator: player 0 reached `48.44%` over 512 games and player 1 reached `41.60%` over 512 games on seed `76000`.

Conclusion: fixed 8x8 ordinary PPO best-response against the v2-v5 checkpoint pool is not a productive route to a pure v5-beating checkpoint, even when trained from player 1. This supports the broader adaptive conclusion from the strategy/Q/Worker probes: the next useful work should not be more small-learning-rate PPO continuation. Either keep rollout-search as an inference-time planner and distill replacement outcomes, or move the adaptive branch toward true seat/size-balanced long-rollout training with richer memory and strategic auxiliary targets.

## 2026-06-17 Strategy-Q Replacement Outcome Target

Added `adaptive_search_distill.py --strategy-q-target {score,outcome,outcome-score}`. The default `score` preserves the previous behavior. The new modes use rollout-search candidate replacement outcomes:

```text
outcome:
  loss/draw-or-unfinished/win -> -1/0/+1

outcome-score:
  outcome + strategy_q_outcome_score_weight * tanh(search_score / search_value_scale)
```

The purpose is to make strategy-Q/rank supervision care first about whether replacing the base action leads to a win/loss/draw, instead of directly fitting the shaped material/prior rollout score. `outcome-score` exists because short rollout candidates often all end as draw/unfinished; the shaped-score tie-break keeps pairwise rank supervision from going empty in those rows.

Short CUDA probes used the existing v3 strategy-aux iter8 checkpoint and froze policy/trunk:

```text
base/search: legacymodels/generals-adaptive-ppo-v3-composite-balanced-probe1.eqx
init:        runs/adaptive-strategy-aux-v3/ckpts/generals-adaptive-strategy-aux-v3-iter-000008.eqx
mode:        mixed seats, 8/12/16, scoreboard history, strategy aux only
search:      top_k=4, rollout_steps=4, rollouts/action=1
loss:        strategy_q_weight=0.1, strategy_q_rank_weight=0.05
```

Training signal:

| run | target | rank margin | final StratQ | final StratRank | observation |
| --- | --- | ---: | ---: | ---: | --- |
| `adaptive-strategy-q-outcome-v1` | outcome | `0.5` | `76.9965` | `0.0000` | pure outcome labels usually tied as draw/unfinished |
| `adaptive-strategy-q-outcome-score-v1` | outcome-score | `0.02` | `86.5230` | `0.1168` | tie-break creates rank signal, but not clean convergence |

64 games/row rerank triage against Expander used `max_steps=750`:

| run | rerank scale | min win rate | key note |
| --- | ---: | ---: | --- |
| outcome v1 | `0.00` | `70.31%` | policy-preservation check only |
| outcome v1 | `0.01` | `62.50%` | Q bias hurts 16x rows |
| outcome-score v1 | `0.00` | `68.75%` | policy-preservation check only |
| outcome-score v1 | `0.005` | `67.19%` | small bias still hurts min row |
| outcome-score v1 | `0.01` | `65.62%` | 16x p1 becomes bottleneck |

A follow-up diagnostic increased candidate search to `rollout_steps=16` with pure outcome targets:

| run | rollout steps | rerank scale | min win rate | key note |
| --- | ---: | ---: | ---: | --- |
| outcome-r16 v1 | 16 | `0.00` | `62.50%` | policy seed had weak 16x draw rows |
| outcome-r16 v1 | 16 | `0.005` | `59.38%` | small Q bias still hurts min row |

The r16 training log showed outcome rank signal only intermittently (`StratRank 0.6351` at iter5, `0.0000` again by iter10/final), so longer short-rollout search did not reliably solve the sparse terminal-outcome label problem.

Candidate `rollout_steps=64` finally created stable terminal-outcome rank signal:

| run | iterations | final StratQ | final StratRank | note |
| --- | ---: | ---: | ---: | --- |
| outcome-r64 v1 | 8 | `82.4372` | `6.6519` | confirms outcome diversity appears at 64 search steps |
| outcome-r64 v2 | 64 | `64.4716` | `4.2998` | Q MSE improves, rank remains high |
| outcome-r64 v3 | 128 | `36.3317` | `2.9618` | Q MSE improves further, rank still noisy |

Rerank triage:

| run | games/row | seed | scale | min win rate | note |
| --- | ---: | ---: | ---: | ---: | --- |
| outcome-r64 v2 | 64 | 76580 | `0.000` | `56.25%` | weak policy-preservation seed |
| outcome-r64 v2 | 64 | 76580 | `0.002` | `65.62%` | same-seed improvement, still below gate |
| outcome-r64 v2 | 64 | 76580 | `0.005` | `62.50%` | over-bias |
| outcome-r64 v3 | 64 | 76680 | `0.000` | `65.62%` | policy-preservation check |
| outcome-r64 v3 | 64 | 76680 | `0.001` | `70.31%` | promising 64-row triage only |
| outcome-r64 v3 | 64 | 76680 | `0.002` | `67.19%` | over-bias begins |
| outcome-r64 v3 | 256 | 76720 | `0.001` | `66.80%` | failed promotion check; 16p1 bottleneck |

Conclusion: replacement-outcome Q targets are now available, and `rollout_steps=64` is the first setting that produces real candidate outcome diversity. However, direct Q-as-logit-bias still fails 256-row validation and remains below the current `71.29%` platform. Do not promote the current outcome-Q checkpoints. If this route continues, it should stop treating the auxiliary Q head as a raw inference bias and instead train a dedicated accepted-replacement rerank head or online policy-improvement gate.

### Accepted-replacement Q weighting

Added `--strategy-q-weight-mode accepted`. This keeps the previous `active` default but lets Q/rank supervision ignore rows unless the candidate set contains a credible replacement for the top-prior action:

```text
accepted if:
  best outcome candidate switches away from top-prior
  and either:
    outcome improves loss -> draw/win or draw -> win
    OR same outcome and shaped-score margin passes --min-margin
```

The first accepted-only r64 probe used the same setup as outcome-r64, but only trained Q/rank on accepted rows:

```text
run:        runs/adaptive-strategy-q-accepted-r64-v1/
init:       strategy-aux-v3 iter8
search:     top_k=4, rollout_steps=64, rollouts/action=1
loss:       strategy_q_weight=0.1, strategy_q_rank_weight=0.05
weighting:  --strategy-q-weight-mode accepted
```

Training signal showed why this is not yet enough by itself:

| metric | observation |
| --- | --- |
| `StratS` | only `6-22` accepted Q rows per `512` flattened samples |
| `StratQ` | highly unstable, roughly `51-157` during the 32-iter run |
| `StratRank` | stayed high, roughly `5.2-7.9` |

64 games/row max750 triage:

| scale | min win rate | bottleneck |
| ---: | ---: | --- |
| `0.000` | `59.38%` | 16p0 |
| `0.001` | `62.50%` | 8p1/16p0/16p1 |

A second probe initialized from the already calibrated `outcome-r64-v3` checkpoint and used lower LR `1e-5`:

```text
run:  runs/adaptive-strategy-q-accepted-r64-v2/
init: runs/adaptive-strategy-q-outcome-r64-v3/generals-adaptive-strategy-q-outcome-r64-v3.eqx
```

It still saw sparse accepted rows (`5-26` per `512`) and did not improve inference:

| scale | min win rate | bottleneck |
| ---: | ---: | --- |
| `0.000` | `68.75%` | 16p1 |
| `0.001` | `62.50%` | 16p0 |
| `0.002` | `67.19%` | 16p1 |

Conclusion: accepted-replacement weighting is implemented and confirms the data problem. True accepted replacements are present but sparse under one-step top-k candidate collection, so aux-only online batches are too noisy even when starting from the better r64-calibrated Q head. Keep the option for future replay/oversampling, but do not continue accepted-only r64 training without a buffer that accumulates and balances accepted rows.

## 2026-06-17 Long-Rollout Mixed PPO Recheck

After the Q/rerank branches failed promotion, ran two direct PPO rechecks from the best available `legacymodels` shared-MSE history checkpoint:

```text
base: /home/codeboy/research/generals-bots/legacymodels/generals-adaptive-ppo-v3-composite-balanced-probe1.eqx
sizes: 8,12,16 with weights 8:1,12:1,16:2
learner: mixed player 0 + player 1 in the same update
rollout: 128 envs x 256 steps
ppo: 1 epoch, minibatch 1024, top_advantage_fraction=0.25
checkpoint output: runs/
```

### Terminal + HL-Gauss v1

This run tested the latest-paper style knobs with more architecture/value-head changes:

```text
run: runs/adaptive-ppo-terminal-hlgauss-mixed-v1/
reward_mode=terminal
terminal_reward_scale=1.0
value_heads=per-size
value_loss=hl-gauss, bins=128, sigma=0.04
outcome_aux_weight=0.02
lr=3e-5
ema_decay=0.999, eval_ema
```

Training rollout wins collapsed after the first iteration:

```text
iter 1:  Episodes 49, Wins 41, Draws 0
iter 10: Episodes 73, Wins 7,  Draws 19
iter 40: Episodes 73, Wins 8,  Draws 21
```

64 games/row max750 evaluation on seed `77060`:

| checkpoint | min win rate | bottleneck |
| --- | ---: | --- |
| same-seed base | `65.62%` | 8p1 |
| iter010 EMA | `67.19%` | 8p1 |
| final EMA | `65.62%` | 8p1 |

Conclusion: this combination is too disruptive from the current checkpoint. It slightly improves one early EMA sample but quickly loses rollout quality, so do not continue this exact terminal+HL-Gauss recipe.

### Conservative composite v2

This run kept the original shared-MSE/composite reward surface and only tested long rollout + mixed seats + top-advantage + EMA:

```text
run: runs/adaptive-ppo-long-mixed-v2/
reward_mode=composite
value_heads=shared
value_loss=mse
lr=1e-5
gamma=0.999
gae_lambda=0.90
ema_decay=0.999, eval_ema
```

Training rollout wins stayed healthy:

```text
iter 10: Episodes 104, Wins 76, Draws 8
iter 40: Episodes 94,  Wins 70, Draws 6
```

64 games/row max750 evaluation on seed `77160`:

| checkpoint | min win rate | bottleneck |
| --- | ---: | --- |
| same-seed base | `64.06%` | 16p0 |
| iter010 EMA | `65.62%` | 16p0 |
| final EMA | `64.06%` | 8p0 |

Conclusion: conservative long-rollout mixed PPO is stable but still not strong enough. It slightly improves the same-seed bottleneck at iter10, then shifts weakness back to 8p0/16p0. This supports the current diagnosis: continuing small CNN PPO continuation moves the weak row rather than solving the size/seat conflict. A stronger next step should either add replay-balanced accepted replacements or change the architecture/data representation, not just extend these PPO runs.

## 2026-06-17 Accepted Replacement Replay and Policy Distill

Implemented two accepted-replacement follow-ups in `adaptive_search_distill.py`:

```text
--strategy-q-replay-capacity
--strategy-q-replay-ratio
```

These keep a bounded buffer of accepted Q rows. Replay samples are appended to future flat batches with only Q/rank weights preserved; KL/search/value/outcome/intent/finish/belief weights are cleared so replay does not amplify unrelated losses.

Also added:

```text
--soft-weight-mode accepted
```

This lets policy distillation update only rows where long rollout search finds a credible replacement over the top-prior action.

### Accepted Q replay

Run:

```text
runs/adaptive-strategy-q-accepted-replay-r64-v1/
init: outcome-r64-v3
search: top_k=4, rollout_steps=64
q replay: capacity=4096, ratio=2.0
loss: Q/rank only, policy frozen
```

Replay solved the sample-count issue mechanically:

```text
accepted rows entering replay: usually 6-21 per 512 current samples
training Q rows after replay: about 1030 per update
replay size by iter64: 726 rows
```

But Q calibration did not clearly improve:

```text
StratQ stayed around 33-43 after warmup
StratRank stayed around 2.6-3.4
```

64 games/row max750 on seed `77280` looked promising:

| scale | min win rate |
| ---: | ---: |
| `0.000` | `70.31%` |
| `0.001` | `71.88%` |
| `0.002` | `70.31%` |

The 256 games/row confirmation on seed `77320` rejected the Q-bias promotion:

| model / scale | min win rate | note |
| --- | ---: | --- |
| replay scale `0.001` | `68.75%` | 16p0 bottleneck |
| replay scale `0.002` | `70.31%` | below platform |
| replay scale `0.000` | `73.05%` | identical to init policy, not replay improvement |
| init `strategy-aux-v3 iter8`, scale `0.000` | `73.05%` | confirms scale-0 replay did not change policy |
| shared-MSE base | `71.88%` | same-seed baseline is already high |

Conclusion: Q replay is implemented and useful for diagnostics, but the current raw Q bias still fails promotion. The high no-bias score came from seed/base policy variance, not replay learning.

### Accepted policy distill

Run:

```text
runs/adaptive-accepted-policy-r64-v1/
init/base: shared-MSE history base
soft_weight_mode=accepted
improve_weight=0.02
kl_weight=1.0
lr=1e-6
```

Accepted policy samples remained sparse (`0.4-4.1%` selected rows). Even with tiny LR and strong KL, direct policy update damaged the weak row:

| model | 64 games/row min | bottleneck |
| --- | ---: | --- |
| same-seed base | `70.31%` | 8p0 |
| accepted-policy v1 | `60.94%` | 16p0 |

Conclusion: direct accepted action distillation is still unsafe in the current CNN policy. It helps 8x rows on some seeds but pushes failure into 16p0, matching the broader seat/size tradeoff pattern. Do not continue action-level accepted distill without either a safer replay-balanced objective or a different architecture.

## 2026-06-17 Context Residual PPO Probes

Implemented an optional adaptive CNN residual context branch:

```text
train_adaptive.py --context-residual
evaluate_adaptive_policy.py --context-residual
train_adaptive.py --context-only-update
```

The branch is two 5x5 convs after the existing four-layer CNN trunk. The second conv is zero-initialized, so adding the branch to a legacy checkpoint preserves initial policy/value outputs. `--context-only-update` keeps PPO gradients only on this branch for the first-stage probe.

### v1: context branch only

Run:

```text
runs/adaptive-context-residual-only-v1/
base: legacymodels/generals-adaptive-ppo-v3-composite-balanced-probe1.eqx
rollout: 128 envs x 256 steps
learner: mixed p0+p1
lr=1e-4
ema_decay=0.99, eval_ema
context_only_update=true
```

Training rollout stayed alive:

```text
iter 1:  Episodes 48, Wins 40, Draws 0
iter 10: Episodes 86, Wins 65, Draws 9
iter 20: Episodes 99, Wins 77, Draws 6
```

64 games/row on seed `77580` looked promising:

| model | min win rate | bottleneck |
| --- | ---: | --- |
| context final | `73.44%` | 16p0/16p1 |
| same-seed base | `68.75%` | 8p0 |

The 256 games/row confirmation on seed `77600` rejected promotion:

| model | min win rate | bottleneck |
| --- | ---: | --- |
| context final | `70.70%` | 16p1 |
| same-seed base | `71.88%` | 16p0 |

Additional 64-row seed `77620` showed that all retained context-only checkpoints were noisy and weak (`57.81%` to `62.50%` min), matching the same-seed base bottleneck at `62.50%` rather than improving it.

### v2: low-LR joint update

Run:

```text
runs/adaptive-context-residual-joint-v2/
same base and rollout
lr=1e-5
context residual enabled
all trainable weights updated
```

64 games/row on seed `77780`:

| model | min win rate | bottleneck |
| --- | ---: | --- |
| joint context final | `65.62%` | 16p0 |
| same-seed base | `71.88%` | 16p0/16p1 |

Conclusion: the zero-init context branch and context-only update path are implemented and runnable on GPU, but these PPO probes do not promote. v1 can improve a favorable 64-row seed but fails 256-row control; v2 drifts below the base immediately. This suggests the architecture hook alone is insufficient under the current PPO objective. The next representation step should either train the context branch from supervised belief/finish/intent targets before PPO, or move to an explicit U-Net/Transformer-style torso instead of another small residual PPO continuation.

## 2026-06-17 Context Auxiliary Pretrain Probe

Implemented distillation support for context-branch representation pretraining:

```text
adaptive_search_distill.py --context-residual
adaptive_search_distill.py --init-context-residual
adaptive_search_distill.py --freeze-context-strategy-aux
train_adaptive.py --init-strategy-aux
```

`--freeze-context-strategy-aux` keeps gradients only on the residual context branch and strategy auxiliary heads. This lets KL keep policy logits near the base while full-state labels train intent/finish/belief representations. `train_adaptive.py --init-strategy-aux` lets PPO warm start from such a checkpoint while dropping the auxiliary heads.

### Context aux v1

Run:

```text
runs/adaptive-context-aux-v1/
base/init: composite-balanced history checkpoint
student: context_residual + strategy_aux
freeze: context + strategy heads only
loss: KL + intent 0.05 + finish 0.05 + belief 0.02
search: top_k=2, rollout_steps=16, rollouts_per_action=1
```

The auxiliary losses learned quickly while KL stayed small:

```text
iter 1:  KL 0.00045, Intent 10.25, Belief 20.25
iter 10: KL 0.00352, Intent 8.58,  Belief 16.66
iter 20: KL 0.01711, Intent 4.98,  Belief 8.38
```

64 games/row on seed `77880`:

| model | min win rate | bottleneck |
| --- | ---: | --- |
| context aux direct | `65.62%` | 16p0 |
| same-seed base | `64.06%` | 16p0 |

This is a small same-seed improvement but nowhere near promotion.

### Aux -> PPO v1

Run:

```text
runs/adaptive-context-aux-ppo-v1/
init: adaptive-context-aux-v1 final
train_adaptive.py --init-strategy-aux --init-context-residual
update scope: context branch only
rollout: 128 envs x 256 steps x 20 iters
```

64 games/row on seed `77980`:

| model | min win rate | bottleneck |
| --- | ---: | --- |
| aux -> PPO final | `64.06%` | 16p0 |
| same-seed base | `60.94%` | 16p1 |

Conclusion: supervised context pretraining works mechanically and improves auxiliary losses, but it still does not solve 16x draw/finish. It can patch some weak seeds, yet the promoted policy remains below the current adaptive platform. The next high-value architecture step should be a real U-Net/Transformer torso or an explicit belief-map input/memory stack, not further 5x5 context-branch PPO.

## 2026-06-17 Pyramid Context Torso Probe

Implemented a stronger U-Net-style optional torso branch:

```text
train_adaptive.py --pyramid-context
train_adaptive.py --init-pyramid-context
evaluate_adaptive_policy.py --pyramid-context
adaptive_search_distill.py --pyramid-context
adaptive_search_distill.py --init-pyramid-context
```

The branch operates after the existing four-layer adaptive CNN trunk:

```text
16x16 trunk features
  -> avg pool 8x8 -> 3x3 conv
  -> avg pool 4x4 -> 3x3 conv
  -> nearest upsample + skip
  -> nearest upsample 16x16
  -> zero-output 3x3 conv
  -> residual add to trunk features
```

Like the 5x5 context branch, its final conv is zero-initialized so legacy checkpoint behavior is preserved at load time. `--context-only-update` now also supports this pyramid branch.

### Pyramid-only PPO v1

Run:

```text
runs/adaptive-pyramid-context-only-v1/
base: composite-balanced history checkpoint
rollout: 128 envs x 256 steps x 20 iters
update scope: pyramid branch only
lr=1e-4
```

64 games/row on seed `78080` initially looked strong:

| model | min win rate | bottleneck |
| --- | ---: | --- |
| pyramid-only final | `73.44%` | 8p1/16p1 |
| same-seed base | `67.19%` | 16p1 |

But 256 games/row confirmation on seed `78100` rejected promotion:

| model | min win rate | bottleneck |
| --- | ---: | --- |
| pyramid-only final | `64.84%` | 16p1 |
| same-seed base | `67.97%` | 16p1 |

### Low-LR joint pyramid v2

Run:

```text
runs/adaptive-pyramid-context-joint-v2/
same base and rollout
all trainable weights updated
lr=1e-5
```

64 games/row on seed `78280`:

| model | min win rate | bottleneck |
| --- | ---: | --- |
| joint pyramid final | `67.19%` | 16p1 |
| same-seed base | `62.50%` | 16p1 |

Retained-checkpoint sweep on seed `78300`:

| checkpoint | min win rate |
| --- | ---: |
| iter5 | `62.50%` |
| iter10 | `62.50%` |
| iter15 | `62.50%` |
| iter20/final | `67.19%` |
| same-seed base | `60.94%` |

This repairs weak seeds but remains far below the existing adaptive platform.

### Pyramid auxiliary v1

Run:

```text
runs/adaptive-pyramid-aux-v1/
student: pyramid_context + strategy_aux
freeze: pyramid + strategy heads only
loss: KL + intent 0.05 + finish 0.05 + belief 0.02
```

Auxiliary optimization worked with very small KL:

```text
iter 1:  KL 0.00000, Intent 5.67, Belief 0.058
iter 10: KL 0.00157, Intent 4.95, Belief 0.057
iter 20: KL 0.00206, Intent 3.63, Belief 0.056
```

64 games/row on seed `78480`:

| model | min win rate | bottleneck |
| --- | ---: | --- |
| pyramid aux direct | `64.06%` | 8p0 |
| same-seed base | `64.06%` | 8p0 |

Conclusion: the pyramid/U-Net-style branch is implemented and trainable, but current PPO and weak-label aux objectives still mostly shift bottlenecks. It can repair some weak seeds but does not produce a promotion candidate. This is stronger evidence that the next architecture step should not be another zero-init add-on branch; it should either replace the trunk with a real U-Net/Transformer policy backbone, or add explicit memory/belief input channels so the policy head receives strategic state rather than relying on weak auxiliary losses to bend a small CNN.

## 2026-06-17 Adaptive U-Net Trunk v1

Implemented the first true trunk-replacement path:

```text
AdaptiveUNetPolicyValueNetwork
train_adaptive.py --network-arch unet
evaluate_adaptive_policy.py --network-arch unet
adaptive fog-memory input planes
teacher KL anchor
teacher rollout action bootstrap
teacher action CE
```

The U-Net is not a zero-init add-on branch. It replaces the old four-layer CNN torso with:

```text
input planes
  -> enc1 16x16
  -> avg pool 8x8 -> enc2
  -> avg pool 4x4 -> bottleneck
  -> nearest upsample + skip
  -> nearest upsample + skip
  -> policy/value heads
```

The first experiment used `channels=64,96,128,64`, 35 input planes (`15` adaptive base + `5` fog memory + `5` current scoreboard + `10` previous/delta history), per-size HL-Gauss value heads, and the legacy history CNN checkpoint only as teacher:

```text
teacher: legacymodels/generals-adaptive-ppo-v3-composite-balanced-probe1.eqx
teacher input: 30 channels
student input: 35 channels
```

Important implementation note: from-scratch U-Net checkpoints should not save high-decay EMA during bootstrap. `adaptive-unet-v1` used `ema_decay=0.99 --eval-ema`; after only 20 iterations the saved EMA remained close to random and scored essentially 0% independent win rate. Later runs saved last iterate.

### Bootstrap Results

All models below use independent student sample-policy evaluation, not teacher-driven rollout.

64 games/row on seed `78680`:

| model | recipe | min win rate | bottleneck |
| --- | --- | ---: | --- |
| v1 EMA | KL 0.2, teacher actions, saved EMA | `0.00%` | all rows |
| v1b | KL 2.0, teacher actions, saved last | `45.31%` | 8p1 |
| v1c | KL 1.0 + CE 1.0 | `48.44%` | 8p1 |
| v1d | KL 0.5 + CE 3.0, weights 8:2/12:1/16:2 | `62.50%` | 8p1 |
| v1e | continue v1d, weights 8:2/12:2/16:1 | `59.38%` | 8p1/16p1 |
| v1f | zero reward, KL 1.0 + CE 10.0 | `59.38%` | 16p1 |

The best U-Net bootstrap candidate was v1d. It passed only the 64-row smoke gate, so it received a 256-row triage against the same-seed base:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base CNN, seed 78700 | `71.09%` | `69.53%` | `87.50%` | `80.86%` | `69.14%` | `71.88%` | `69.14%` |
| U-Net v1d, seed 78700 | `66.80%` | `66.41%` | `75.39%` | `76.56%` | `70.70%` | `71.09%` | `66.41%` |

Conclusion: the U-Net trunk, fog-memory input, and teacher bootstrap are wired correctly and train on GPU, but PPO-in-rollout imitation is not enough to reliably clone the legacy CNN across all size/seat rows. The U-Net already matches or slightly improves the large-map rows in some seeds (`16p0=70.70%` vs base `69.14%` in the 256-row triage), but it gives up too much 8x/12x strength.

Do not promote the current U-Net checkpoints. The next U-Net step should be a dedicated offline teacher-imitation or search-to-strategy dataset:

```text
collect teacher-driven trajectories with stored obs/mask/action/logits/size/seat
train U-Net with all-active teacher KL + action CE for many epochs
verify it can match the CNN teacher at 256-row before PPO
then add belief/finish auxiliary targets and only then resume sparse PPO
```

This result also confirms that trunk replacement should be staged differently from add-on branches: first clone the existing policy distribution well enough, then fine-tune strategic behavior. PPO bootstrap alone still shifts weak rows before the new trunk has a stable policy prior.

## 2026-06-17 Adaptive U-Net Teacher Imitation v2/v3

Implemented `adaptive_teacher_imitation.py`, a dedicated policy-checkpoint imitation trainer for adaptive trunks. Unlike `train_adaptive.py --teacher-rollout-actions`, this script removes PPO reward/value/advantage from the update and trains the student only from teacher behavior:

```text
teacher-driven mixed-seat rollouts
student input: 35 channels with fog memory + scoreboard history
teacher input: 30-channel legacy history observation
loss = all-action KL(teacher || student) + teacher action CE - entropy bonus
multiple shuffled epochs per collected rollout
```

This directly tests the staging hypothesis from the previous section: first make the new U-Net trunk match the old CNN policy, then use PPO/search/belief losses.

### Imitation v2: sampled teacher actions

Run:

```text
runs/adaptive-unet-imitation-v2/
init: runs/adaptive-unet-v1d/generals-adaptive-unet-v1d.eqx
teacher: legacymodels/generals-adaptive-ppo-v3-composite-balanced-probe1.eqx
teacher_policy_mode=sample
loss: KL 1.0 + CE 3.0
rollout/update: 128 envs x 128 steps x 30 iters x 3 epochs
weights: 8:2,12:2,16:1
```

Training metrics did not improve cleanly because sampled teacher actions are noisy labels:

```text
iter 1:  KL 0.0580, CE 1.3094, Acc 57.2%
iter 30: KL 0.0652, CE 2.0977, Acc 42.9%
```

64 games/row on seed `78760`:

| model | min win rate | bottleneck |
| --- | ---: | --- |
| imitation v2 | `59.38%` | 8p1 |

Conclusion: sampled teacher action CE is too noisy. It can improve individual rows but keeps moving the bottleneck.

### Imitation v3: greedy teacher actions + KL distribution anchor

Run:

```text
runs/adaptive-unet-imitation-v3/
init: runs/adaptive-unet-v1d/generals-adaptive-unet-v1d.eqx
teacher_policy_mode=greedy
loss: KL 1.0 + CE 5.0
rollout/update: 128 envs x 128 steps x 40 iters x 3 epochs
weights: 8:2,12:2,16:1
```

Greedy labels gave a much cleaner behavioral target:

```text
iter 1:  KL 0.1398, CE 0.7332, Acc 78.4%
iter 40: KL 0.4862, CE 0.6900, Acc 79.3%
```

64 games/row on seed `78800`:

| row | win rate |
| --- | ---: |
| 8p0 | `84.38%` |
| 8p1 | `70.31%` |
| 12p0 | `84.38%` |
| 12p1 | `78.12%` |
| 16p0 | `73.44%` |
| 16p1 | `79.69%` |
| min | `70.31%` |

256 games/row on seed `78820`:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base CNN | `69.92%` | `72.66%` | `85.55%` | `76.95%` | `71.88%` | `72.27%` | `69.92%` |
| U-Net imitation v3 | `72.66%` | `76.95%` | `80.86%` | `83.20%` | `79.30%` | `76.17%` | `72.66%` |

512 games/row on seed `78840`:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base CNN | `75.00%` | `70.12%` | `80.27%` | `82.42%` | `71.68%` | `72.07%` | `70.12%` |
| U-Net imitation v3 | `77.73%` | `75.00%` | `79.69%` | `81.45%` | `75.20%` | `75.98%` | `75.00%` |

Draw rates also improved on the large rows:

```text
base 16p0/16p1 draw: 18.36% / 18.55%
v3   16p0/16p1 draw: 15.04% / 16.21%
```

Conclusion: offline teacher imitation with greedy action labels is the first U-Net trunk replacement route that clears the 512 games/row promotion-candidate gate. It does not yet prove final replacement over 2048 games/row, but it is strong enough to become the new U-Net base for the next branch.

Recommended next branch:

```text
init: runs/adaptive-unet-imitation-v3/generals-adaptive-unet-imitation-v3.eqx
train: sparse PPO / search-to-strategy auxiliary
keep: fog memory + scoreboard history + per-size HL-Gauss value
avoid: high-decay EMA from random init
gate: 512-row first, then 2048-row final evidence
```

This also resolves the prior uncertainty: trunk replacement should not start from PPO bootstrap. The practical sequence is now:

```text
1. teacher-imitation U-Net until 256/512-row >= legacy CNN
2. add belief/finish/intent/search-value heads
3. sparse PPO or search-to-strategy fine-tune from the imitated U-Net
```

## 2026-06-17 Adaptive U-Net 2048 Confirmation and PPO v4

### Imitation v3 2048-row confirmation

After v3 cleared the 512 games/row gate, it received a same-seed 2048 games/row confirmation against the legacy adaptive CNN.

2048 games/row on seed `78860`:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base CNN | `74.66%` | `72.85%` | `79.69%` | `81.69%` | `70.65%` | `71.34%` | `70.65%` |
| U-Net imitation v3 | `75.29%` | `73.68%` | `81.45%` | `82.71%` | `76.51%` | `76.76%` | `73.68%` |

Large-map draw rates:

```text
base 16p0/16p1 draw: 18.36% / 19.04%
v3   16p0/16p1 draw: 14.45% / 14.36%
```

Conclusion: v3 is a real replacement candidate, not a short-eval false positive. It improves the 2048-row min by `+3.03 pp` and materially reduces 16x draw rate.

### Sparse PPO v4 from v3

Run:

```text
runs/adaptive-unet-ppo-v4/
init: runs/adaptive-unet-imitation-v3/generals-adaptive-unet-imitation-v3.eqx
rollout: 128 envs x 256 steps x 20 iters
update: 1 epoch, minibatch 1024
reward: terminal only, terminal_reward_scale=1.0
optimizer: lr=1e-5
top_advantage_fraction=0.25
sizes: 8:1,12:1,16:2
```

Training stayed stable:

```text
iter 1:  Loss 2.2066, Episodes 48, Wins 37, Draws 0
iter 10: Loss 2.1813, Episodes 93, Wins 70, Draws 8
iter 20: Loss 2.1520, Episodes 94, Wins 78, Draws 4
```

256 games/row on seed `78900`:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net v3 | `66.02%` | `75.39%` | `80.86%` | `84.38%` | `76.95%` | `77.73%` | `66.02%` |
| U-Net PPO v4 | `71.48%` | `76.95%` | `89.06%` | `82.81%` | `78.12%` | `75.78%` | `71.48%` |

512 games/row on seed `78920`:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net v3 | `69.14%` | `74.22%` | `82.23%` | `79.88%` | `75.00%` | `76.56%` | `69.14%` |
| U-Net PPO v4 | `72.27%` | `74.02%` | `80.66%` | `80.47%` | `76.17%` | `79.30%` | `72.27%` |

2048 games/row on seed `78940`:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net v3 | `72.66%` | `74.80%` | `81.40%` | `80.86%` | `77.88%` | `77.25%` | `72.66%` |
| U-Net PPO v4 | `73.05%` | `74.51%` | `82.18%` | `82.32%` | `79.64%` | `78.66%` | `73.05%` |

Large-map draw rates at 2048 games/row:

```text
v3 16p0/16p1 draw: 14.06% / 14.60%
v4 16p0/16p1 draw: 12.16% / 12.89%
```

Conclusion: v4 is a modest but verified improvement over v3 at the 2048-row gate. It raises min win rate by `+0.39 pp` and improves 12x/16x, especially draw reduction, without creating a new severe weak row. The 8p1 row regresses slightly (`74.80% -> 74.51%`), so the gain is not large enough to stop exploring, but v4 is now the stronger U-Net base.

Recommended next branch:

```text
init: runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx
goal: recover/boost 8p1 while preserving 16x draw reduction
options:
  1. short mixed PPO with weights 8:2,12:1,16:2 and lr <= 5e-6
  2. search-to-strategy auxiliary on v4, no raw Q rerank
  3. add belief/finish heads trained from full state, then PPO
gate:
  256-row smoke
  512-row promotion candidate
  2048-row final evidence
```

## 2026-06-17 U-Net PPO v5-v7 Top-Advantage Diagnostics

### Global top-advantage v5

v5 tested the first recommended follow-up from v4: a conservative low-LR continuation with extra 8x sampling.

```text
runs/adaptive-unet-ppo-v5/
init: runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx
rollout: 128 envs x 256 steps x 20 iters
update: 1 epoch, minibatch 1024
reward: terminal only
optimizer: lr=5e-6
top_advantage_fraction=0.25, mode=global
sizes: 8:2,12:1,16:2
```

Training remained stable:

```text
iter 1:  Loss 2.1636, Episodes 67, Wins 56, Draws 0
iter 10: Loss 2.2130, Episodes 107, Wins 87, Draws 1
iter 20: Loss 2.2196, Episodes 115, Wins 90, Draws 4
```

256 games/row on seed `78980`:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net PPO v4 | `73.83%` | `76.56%` | `77.73%` | `84.38%` | `79.30%` | `85.55%` | `73.83%` |
| U-Net PPO v5 | `67.97%` | `78.91%` | `81.25%` | `85.16%` | `82.81%` | `81.25%` | `67.97%` |

Conclusion: v5 is rejected. Extra 8x sampling with global top-advantage improved 8p1 and large rows but sacrificed 8p0, which confirms that the global filter can move gradient budget into easier rows instead of protecting the bottleneck.

### Stratified top-advantage v6

To test that diagnosis, `train_adaptive.py` now supports:

```text
--top-advantage-mode global      # old behavior, one global top-k over rollout samples
--top-advantage-mode stratified  # independent top-k per effective size and learner seat
```

v6 reused the v5 recipe but enabled `--top-advantage-mode stratified`.

Training:

```text
iter 1:  Loss 2.1380, Episodes 63, Wins 55, Draws 0
iter 10: Loss 2.2171, Episodes 115, Wins 89, Draws 6
iter 20: Loss 2.2240, Episodes 109, Wins 84, Draws 6
```

256 games/row on seed `79020` looked strong:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net PPO v4 | `70.70%` | `73.05%` | `84.38%` | `80.86%` | `77.73%` | `80.47%` | `70.70%` |
| U-Net PPO v6 | `75.78%` | `76.56%` | `82.42%` | `83.98%` | `80.08%` | `80.08%` | `75.78%` |

512 games/row on seed `79040` was only a narrow gain:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net PPO v4 | `74.41%` | `72.46%` | `85.35%` | `83.79%` | `80.47%` | `79.30%` | `72.46%` |
| U-Net PPO v6 | `74.02%` | `73.05%` | `84.77%` | `82.03%` | `82.03%` | `77.15%` | `73.05%` |

2048 games/row on seed `79060` rejected the checkpoint:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net PPO v4 | `74.51%` | `72.17%` | `83.01%` | `81.05%` | `78.81%` | `80.32%` | `72.17%` |
| U-Net PPO v6 | `70.80%` | `73.54%` | `84.57%` | `81.49%` | `79.20%` | `79.64%` | `70.80%` |

Large-map draw rates at 2048 games/row:

```text
v4 16p0/16p1 draw: 13.43% / 12.11%
v6 16p0/16p1 draw: 12.55% / 13.13%
```

Conclusion: stratified filtering is useful as a trainer diagnostic and may reduce easy-row domination in short gates, but v6 is still a short-eval false positive. It does not replace v4.

### EMA-saved stratified v7

v7 tested whether saving EMA parameters could keep the stratified continuation closer to v4:

```text
runs/adaptive-unet-ppo-v7/
same recipe as v6
ema_decay=0.99, eval_ema=true
```

Training:

```text
iter 1:  Loss 2.1898, Episodes 62, Wins 49, Draws 0
iter 10: Loss 2.2403, Episodes 104, Wins 75, Draws 8
iter 20: Loss 2.1723, Episodes 97, Wins 81, Draws 3
```

512 games/row on seed `79120`:

| model | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net PPO v4 | `71.48%` | `74.80%` | `81.64%` | `82.23%` | `77.93%` | `78.12%` | `71.48%` |
| U-Net PPO v7 | `70.70%` | `72.66%` | `81.64%` | `82.03%` | `78.52%` | `79.10%` | `70.70%` |

Conclusion: v7 is rejected at the 512-row gate; no 2048-row run is warranted.

Current promotion state:

```text
promoted/current base: runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx
rejected: v5, v6, v7
keep code tool: --top-advantage-mode stratified
```

Next direction:

```text
1. Stop simple low-LR PPO continuation from v4 unless it changes the objective or data path.
2. Use stratified top-advantage only when paired with a stronger trust region, row-wise KL cap, or row-balanced replay.
3. Prefer the next hard shift: belief/finish auxiliary heads trained from full state, or search-to-strategy supervision that teaches finish/draw risk rather than direct policy replay.
```

## 2026-06-17 Adaptive U-Net vs Fixed 8x8 v5 Gate

### Evaluation support

Added policy-checkpoint opponent support to the adaptive evaluator and trainer:

```text
evaluate_adaptive_policy.py --opponent-policy-path <fixed-policy.eqx>
train_adaptive.py --opponent-policy-path <fixed-policy.eqx>
adaptive_teacher_imitation.py --fixed-teacher-model-path <fixed-policy.eqx>
adaptive_teacher_imitation.py --opponent-policy-path <fixed-policy.eqx>
```

Because adaptive checkpoints may store per-size value heads for `8,12,16` while a v5 gate evaluates only `8`, the scripts also accept:

```text
--value-head-sizes 8,12,16
--init-value-head-sizes 8,12,16
```

Fixed 8x8 policy logits use `9*8*8` actions; adaptive policy logits use `8*pad*pad + 1` actions. The fixed-teacher imitation path maps the first 8 move planes into the padded adaptive lattice and combines the fixed pass plane with `logsumexp` into the adaptive global pass logit.

### Current U-Net v4 against fixed v5

512 games/seat on seed `79220`, `max_steps=250`, fixed v5 sample opponent:

| model | 8p0 | 8p1 | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net PPO v4 | `10.55%` | `9.57%` | `9.57%` | `52.34%` | `57.42%` |

Conclusion: v4 is an Expander specialist, not a v5 beater. The explicit 8x8-vs-v5 requirement is still far from satisfied.

### Direct PPO against fixed v5

Run:

```text
runs/adaptive-unet-v5br-v1/
init: runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx
opponent: fixed generals-ppo-8x8-expander-gpu-v5.eqx sample
rollout: 128 envs x 256 steps x 40 iters
reward: composite + terminal_reward_scale=1.0
lr=1e-5
```

Training stayed in the same weak band:

```text
iter 1:  Loss 3.8458, Episodes 138, Wins 15, Draws 62
iter 20: Loss 3.4732, Episodes 153, Wins 9,  Draws 95
iter 40: Loss 3.4294, Episodes 153, Wins 10, Draws 98
```

512 games/seat on seed `79280`, `max_steps=250`:

| model | 8p0 | 8p1 | min |
| --- | ---: | ---: | ---: |
| U-Net PPO v4 | `11.33%` | `7.42%` | `7.42%` |
| v5br-v1 final | `10.55%` | `7.62%` | `7.62%` |

128-row checkpoint sweep on seed `79300` found no useful intermediate checkpoint; min stayed around `9.38-10.94%`.

Conclusion: direct PPO from a v5-weak U-Net policy does not create a v5 best response. It mostly moves losses/draws around.

### Fixed-v5 imitation bootstrap

`adaptive_teacher_imitation.py` now supports fixed 8x8 policy teachers. v1 used fixed v5 as teacher but Expander as rollout opponent:

```text
runs/adaptive-fixed-v5-imitation-v1/
teacher: fixed v5 greedy labels + full-logit KL
opponent: Expander
init: U-Net PPO v4
rollout/update: 128 envs x 128 steps x 40 iters x 3 epochs
loss: KL 1.0 + CE 5.0
```

Training reached `82.1%` action accuracy. It improved decisive behavior against v5 but produced many 250-step draws:

| model | max steps | 8p0 | 8p1 | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-v5 imitation v1 | 250 | `11.52%` | `11.33%` | `11.33%` | `74.22%` | `73.44%` |
| fixed-v5 imitation v1 | 750 | `48.05%` | `42.77%` | `42.77%` | `2.34%` | `2.15%` |

Conclusion: v1 learned a slow v5-like policy. It can survive to near-decisive v5-vs-v5 strength by 750 steps, but fails the 250-step gate because it finishes too slowly.

v2 changed the data distribution so both teacher and opponent are fixed v5:

```text
runs/adaptive-fixed-v5-imitation-v2/
teacher: fixed v5 greedy labels + full-logit KL
opponent: fixed v5 sample
init: U-Net PPO v4
rollout/update: 128 envs x 128 steps x 40 iters x 3 epochs
loss: KL 1.0 + CE 5.0
```

Training reached `83.1%` action accuracy and improved the 250-step gate:

| model | max steps | 8p0 | 8p1 | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-v5 imitation v2 | 250 | `15.23%` | `12.89%` | `12.89%` | `64.65%` | `68.16%` |
| fixed-v5 imitation v2 | 750 | `45.51%` | `40.43%` | `40.43%` | `0.39%` | `1.95%` |

v3 continued from v2 with longer self-distribution imitation:

```text
runs/adaptive-fixed-v5-imitation-v3/
init: fixed-v5 imitation v2
rollout/update: 128 envs x 128 steps x 80 iters x 4 epochs
loss: KL 1.0 + CE 8.0
```

Training reached only `85.9%` action accuracy. Final 512 games/seat on seed `79740`, `max_steps=250`:

| model | 8p0 | 8p1 | min | p0 decisive | p1 decisive | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-v5 imitation v3 | `16.02%` | `13.48%` | `13.48%` | `51.90%` | `48.25%` | `69.14%` | `72.07%` |

Intermediate checkpoints did not beat final in a 128-row sweep.

### Finish fine-tunes

Two short PPO finish fine-tunes with `truncation_reward_scale=0.05` were tested:

| model | init | 512-row max250 result |
| --- | --- | --- |
| `adaptive-unet-v5finish-v1` | imitation v1 | `11.52%` min, worse than same-seed imitation v1 `12.50%` |
| `adaptive-unet-v5finish-v2` | imitation v2 | `12.11%` min, worse than imitation v2's best observed `12.89%` |

Conclusion: the fixed-v5 imitation path is useful bootstrap infrastructure, but the current clone is still too slow and inaccurate to beat v5. Plain PPO finish fine-tuning with a small timeout penalty does not solve the anti-draw problem.

Current state:

```text
Expander base remains: runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx
Best fixed-v5 gate artifact so far: runs/adaptive-fixed-v5-imitation-v3/generals-adaptive-fixed-v5-imitation-v3.eqx
Best fixed-v5 250-step min observed: 13.48%
Best fixed-v5 750-step min observed: 42.77% from imitation v1
```

Next direction:

```text
1. Train explicit finish/draw-risk heads from v5-vs-v5 rollouts, not just action CE.
2. Use search teacher or outcome-labeled replacement actions to teach earlier general capture.
3. Preserve v4 as the Expander large-map base; do not promote fixed-v5 imitation checkpoints yet.
```

## 2026-06-17 Fixed-v5 Imitation Outcome Weighting

Added `adaptive_teacher_imitation.py --outcome-weight-mode terminal`, which uses known rollout outcomes to weight imitation KL/CE samples:

```text
--win-action-weight
--loss-action-weight
--draw-action-weight
--unknown-action-weight
```

The intent was to bias v5 imitation toward teacher trajectories that actually produce decisive wins inside the 250-step gate, instead of giving slow draw trajectories equal weight.

### Winner-biased imitation v4

Run:

```text
runs/adaptive-fixed-v5-imitation-v4/
init: runs/adaptive-fixed-v5-imitation-v3/generals-adaptive-fixed-v5-imitation-v3.eqx
teacher/opponent: fixed v5 sample distribution
teacher action mode: greedy
loss: KL 1.0 + CE 8.0
outcome weights: win=3.0, loss=0.5, draw=0.1, unknown=0.5
rollout/update: 128 envs x 128 steps x 80 iters x 4 epochs
```

Training stayed numerically stable, with weighted action accuracy around `86-88%`:

```text
iter 1:  Loss 3.0882, KL 0.2162, CE 0.3590, Acc 87.0%, W 0.62
iter 40: Loss 3.3363, KL 0.2684, CE 0.3835, Acc 86.7%, W 0.46
iter 80: Loss 3.3722, KL 0.2907, CE 0.3852, Acc 86.6%, W 0.45
```

512 games/seat on seed `79880`, `max_steps=250`:

| model | 8p0 | 8p1 | min | p0 decisive | p1 decisive | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-v5 imitation v3 | `12.70%` | `15.82%` | `12.70%` | `50.39%` | `52.60%` | `74.80%` | `69.92%` |
| fixed-v5 imitation v4 | `13.48%` | `15.43%` | `13.48%` | `54.76%` | `51.30%` | `75.39%` | `69.92%` |

Conclusion: outcome weighting slightly improved same-seed min and p0 decisive rate, but not enough to change the strategic picture. It did not reduce draw rate.

### Sample-teacher / KL-dominant imitation v5

Greedy CE was suspected of making the student too deterministic and slow, so v5 switched to sampled teacher actions and KL-dominant loss:

```text
runs/adaptive-fixed-v5-imitation-v5/
init: fixed-v5 imitation v3
teacher/opponent: fixed v5 sample distribution
teacher action mode: sample
loss: KL 2.0 + CE 1.0 + entropy 0.001
rollout/update: 128 envs x 128 steps x 40 iters x 3 epochs
```

Training reduced KL and raised student entropy:

```text
iter 1:  KL 0.1566, CE 0.8832, Acc 69.6%, Ent 0.58
iter 40: KL 0.0696, CE 0.9208, Acc 67.8%, Ent 0.92
```

512 games/seat on seed `79940`, `max_steps=250`:

| model | 8p0 | 8p1 | min | p0 decisive | p1 decisive | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-v5 imitation v5 final | `14.26%` | `15.62%` | `14.26%` | `48.67%` | `49.08%` | `70.70%` | `68.16%` |
| fixed-v5 imitation v5 iter030 | `14.65%` | `15.04%` | `14.65%` | `45.45%` | `45.03%` | `67.77%` | `66.60%` |

The 128-row checkpoint sweep on seed `79960` suggested iter030/iter040 were better than earlier checkpoints, and the 512-row same-seed check confirmed iter030 is the best artifact in this sub-branch so far.

### KL-heavy continuation v6

v6 continued from v5 iter030 with even more KL and very small sampled CE:

```text
runs/adaptive-fixed-v5-imitation-v6/
init: v5 iter030
loss: KL 3.0 + CE 0.2 + entropy 0.002
```

512 games/seat on seed `80020`, `max_steps=250`:

| model | 8p0 | 8p1 | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-v5 imitation v6 | `12.70%` | `14.84%` | `12.70%` | `68.16%` | `63.28%` |

Conclusion: too much KL-only calibration regressed the gate. The best current pure checkpoint for the fixed-v5 gate is:

```text
runs/adaptive-fixed-v5-imitation-v5/ckpts/generals-adaptive-fixed-v5-imitation-v5-iter-000030.eqx
512-row max250 min: 14.65%
```

Overall conclusion:

```text
Outcome weighting and sample-teacher KL both help a little, but action-distribution imitation is still not enough.
The bottleneck is now explicit finish/search outcome selection, not merely matching v5 logits.
Next experiment should add finish/draw-risk supervision or search-labeled finish actions on v5-vs-v5 states.
```

## 2026-06-17 Fixed-v5 Finish/Draw-Risk Follow-up

This round added three trainer capabilities:

```text
1. U-Net adaptive checkpoints can now warm start into structurally expanded U-Net templates.
   This is needed when adding outcome/strategy heads to an existing U-Net checkpoint.

2. adaptive_teacher_imitation.py can train the 3-class loss/draw/win outcome head:
   --outcome-head --outcome-aux-weight W

3. action KL/CE outcome weighting now supports:
   --terminal-action-window N
   --p0-action-weight / --p1-action-weight
```

The immediate target was the fixed 8x8 v5 gate at `max_steps=250`, where the best previous artifact remained:

```text
runs/adaptive-fixed-v5-imitation-v5/ckpts/generals-adaptive-fixed-v5-imitation-v5-iter-000030.eqx
512 games/seat seed 79940: min 14.65%, draw 67.77% / 66.60%
```

### Outcome auxiliary fine-tune

```text
runs/adaptive-fixed-v5-outcome-v1/
init: fixed-v5 imitation v5 iter030
teacher/opponent: fixed v5 sample
loss: KL 2.0 + CE 1.0 + entropy 0.001 + outcome aux 0.05
outcome class weights: win=3, loss=3, draw=1
rollout/update: 64 envs x 256 steps x 30 iters x 2 epochs
```

Training was healthy but only affected representation:

```text
iter 1:  KL 0.0926, CE 0.9036, Acc 68.1%, Out 1.3617 / 46.2%
iter 20: KL 0.0724, CE 0.9108, Acc 68.0%, Out 0.9817 / 53.1%
iter 30: KL 0.0681, CE 0.8528, Acc 69.7%, Out 0.9767 / 49.5%
```

Evaluation:

| model | seed | games/seat | 8p0 | 8p1 | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base iter030 | 80240 | 256 | `12.50%` | `14.45%` | `12.50%` | `66.41%` | `69.14%` |
| outcome-v1 | 80240 | 256 | `13.67%` | `15.62%` | `13.67%` | `66.02%` | `67.19%` |
| outcome-v1 | 80300 | 512 | `12.30%` | `13.28%` | `12.30%` | `71.09%` | `65.62%` |

Conclusion: outcome CE is wired and learnable, but it does not reliably improve policy decisions. Do not promote.

### Terminal-window action weighting

The next attempt weighted action imitation toward known terminal windows:

```text
runs/adaptive-fixed-v5-terminal-window-v1/
init: fixed-v5 imitation v5 iter030
loss: KL 2.0 + CE 1.0 + entropy 0.001 + outcome aux 0.02
action weights: win=4, loss=0.5, draw=0.05, unknown=0.2
terminal window: last 64 rollout steps before known done
```

Training stayed numerically stable:

```text
iter 1:  KL 0.0734, CE 0.8548, Acc 69.5%, Out 4.0858 / 24.0%, W 0.12
iter 20: KL 0.0932, CE 0.8558, Acc 69.7%, Out 0.8993 / 65.2%, W 0.22
iter 40: KL 0.0650, CE 0.8903, Acc 68.2%, Out 0.9441 / 56.0%, W 0.20
```

But the gate exposed a seat tradeoff:

| model | seed | games/seat | 8p0 | 8p1 | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base iter030 | 80480 | 256 | `14.06%` | `11.33%` | `11.33%` | `72.66%` | `71.09%` |
| terminal-window-v1 | 80480 | 256 | `17.97%` | `8.98%` | `8.98%` | `68.36%` | `68.75%` |

`terminal-window-v1` made player 0 more decisive but damaged player 1.

### P1-focused terminal-window weighting

```text
runs/adaptive-fixed-v5-terminal-window-p1-v2/
same as v1, plus:
  --p0-action-weight 0.5
  --p1-action-weight 3.0
```

Training stayed stable:

```text
iter 1:  KL 0.0787, CE 0.8006, Acc 69.4%, Out 4.5599 / 20.8%, W 0.19
iter 20: KL 0.0661, CE 0.9423, Acc 67.4%, Out 0.9796 / 56.4%, W 0.32
iter 40: KL 0.0706, CE 0.8376, Acc 70.4%, Out 0.9921 / 50.8%, W 0.29
```

But same-seed evaluation showed clear regression:

| model | seed | games/seat | 8p0 | 8p1 | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base iter030 | 80560 | 256 | `19.14%` | `19.14%` | `19.14%` | `59.77%` | `60.16%` |
| terminal-window-p1-v2 | 80560 | 256 | `14.84%` | `12.89%` | `12.89%` | `67.19%` | `65.23%` |

Conclusion: terminal-window imitation creates another seat tradeoff and should not be continued by simple reweighting.

### KL-anchored PPO against fixed v5

One final online-credit probe used terminal-only PPO with a teacher KL anchor to the best imitation checkpoint:

```text
runs/adaptive-fixed-v5-ppo-kl-v1/
init/teacher: fixed-v5 imitation v5 iter030
opponent: fixed v5 sample
reward: terminal only, terminal_reward_scale=1.0
PPO: 64 envs x 256 steps x 20 iters, epochs=1
top advantage: stratified 0.25
teacher KL: 0.05
outcome aux: 0.01
lr: 2e-6
```

Rollout signal was weak:

```text
iter 1:  Episodes 66, Wins 12, Draws 42
iter 10: Episodes 80, Wins 12, Draws 51
iter 20: Episodes 71, Wins 9, Draws 56
```

256 games/seat seed `80660`:

| model | 8p0 | 8p1 | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO-KL-v1 | `9.77%` | `12.89%` | `9.77%` | `63.28%` | `68.75%` |

Conclusion: low-LR PPO + outcome aux + KL anchor still cannot repair the fixed-v5 gate.

### Updated decision

Do not continue these sub-branches:

```text
pure outcome-head fine-tune
terminal-window action reweighting
seat-specific terminal-window action reweighting
KL-anchored low-LR PPO against fixed v5
```

The new negative evidence is specific: fixed-v5 action traces contain useful local finish moves, but reweighting them by outcome/window/seat still shifts weakness between rows. The next useful branch should construct actual replacement-outcome/search data:

```text
state
legal top-k actions
base action
candidate action
full policy replacement outcome under fixed v5 opponent
time-to-terminal / draw-risk delta
seat
```

Then train a dedicated rerank/finish head or policy correction on accepted replacement rows, rather than globally changing the primitive policy with weighted imitation.

## 2026-06-17 U-Net Imitation v3 Two-Seed 2048-Row Validation

Phase 0 follow-up validated the U-Net imitation v3 checkpoint on a second 2048 games/row seed:

```text
runs/adaptive-unet-imitation-v3/generals-adaptive-unet-imitation-v3.eqx
eval command template:
  evaluate_adaptive_policy.py
  --grid-sizes 8,12,16
  --num-games 2048
  --max-steps 750
  --opponent expander
  --policy-mode sample
  --network-arch unet
  --channels 64,96,128,64
  --scoreboard-history
  --fog-memory
  --value-heads per-size
  --value-loss hl-gauss
  --outcome-head
```

Two-seed evidence:

| checkpoint | seed | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net imitation v3 | 78860 | `75.29%` | `73.68%` | `81.45%` | `82.71%` | `76.51%` | `76.76%` | `73.68%` |
| U-Net imitation v3 | 80720 | `74.61%` | `75.05%` | `81.01%` | `81.10%` | `77.29%` | `76.32%` | `74.61%` |

Draw rates:

| checkpoint | seed | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net imitation v3 | 78860 | `0.20%` | `0.29%` | `3.96%` | `3.81%` | `14.45%` | `14.36%` |
| U-Net imitation v3 | 80720 | `0.68%` | `0.49%` | `4.49%` | `4.39%` | `14.36%` | `15.14%` |

Comparison anchor:

```text
runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx
seed 78940 2048-row min: 73.05%
rows: 8p0 73.05%, 8p1 74.51%, 12p0 82.18%, 12p1 82.32%, 16p0 79.64%, 16p1 78.66%
draw: 16p0 12.16%, 16p1 12.89%
```

Decision:

```text
U-Net imitation v3 is validated as a stable supervised base, not a 512-row false positive.
U-Net PPO v4 remains the active Expander base because it has better 12/16 rows and lower 16x draw.
For strategy-dataset work, use v3 as the clean supervised-policy teacher/base and v4 as the stronger Expander active baseline.
```

## 2026-06-17 Adaptive Strategy Dataset v0

Implemented `examples/_experimental/ppo/adaptive_strategy_dataset.py`, a first offline shard collector for strategy supervision.

The collector supports:

```text
teacher-kind:
  adaptive checkpoint
  fixed 8x8 PolicyValueNetwork checkpoint
  Expander target distribution

mixed learner seats:
  p0 and p1 are collected in the same shard

saved arrays:
  obs
  legal_mask
  active
  teacher_logits
  teacher_action_index
  teacher_greedy_index
  action
  grid_size
  seat
  done / winner
  outcome / outcome_known
  steps_to_terminal / terminal_time
  finish_within_50 / 100 / 250
  draw_risk
  enemy_general_heatmap
  enemy_owned_map
  hidden_enemy_owned_map
  hidden_enemy_army_map
  city_map
  source_heatmap
  target_heatmap
  weak intent
  time
  visible_enemy_count
  visible_enemy_density
  contact
```

Smoke checks:

```text
adaptive U-Net v3 smoke:
  4 envs x 8 steps
  output: runs/adaptive-strategy-dataset-smoke2/unet-v3-smoke-00000.npz
  samples: 32
  obs: (32, 35, 16, 16)
  teacher_logits: (32, 2049), finite float16 after clipping masked logits to -1e4

fixed-v5 smoke:
  4 envs x 260 steps, max250
  output: runs/adaptive-strategy-dataset-fixed-v5-smoke/fixed-v5-smoke-00000.npz
  samples: 1040
  episodes: 4
  draws: 4
  draw_risk mean: 0.961
```

Initial v0 shards:

| shard | samples | size | grid distribution | seat distribution | outcome known | finish250 | draw risk | contact |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `runs/adaptive-strategy-dataset-v0/unet-v3-expander/unet-v3-expander-00000.npz` | 2048 | 1.2M | 8:768, 12:384, 16:896 | p0:1024, p1:1024 | `0.145` | `0.145` | `0.000` | `0.401` |
| `runs/adaptive-strategy-dataset-v0/fixed-v5-max250/fixed-v5-max250-00000.npz` | 4160 | 1.9M | 8:4160 | p0:2080, p1:2080 | `0.832` | `0.029` | `0.661` | `0.859` |

Interpretation:

```text
The v0 collector is ready for the next stage: training frozen-trunk finish/belief/intent heads.
The fixed-v5 shard captures the desired anti-draw regime: high contact, high draw-risk, low finish-within-250.
The U-Net/Expander shard captures successful adaptive policy states and gives positive finish labels for contrast.
```

Known limitations for v0:

```text
target_heatmap is currently the true enemy general heatmap.
source_heatmap is the own largest-stack cell.
intent labels are weak rule labels without search outcomes.
No replacement-outcome top-k search rows are saved yet.
```

### Midgame decisive save filters

`adaptive_strategy_dataset.py` now supports save-time row filters for decisive
trajectory imitation:

```text
--min-save-turn / --max-save-turn
--require-contact
--min-visible-enemy-cells
--min-visible-enemy-density
--require-outcome-known
--require-win
--require-finish-within-250
--require-win-or-finish-within-250
--draw-only
--terminal-window
```

The collector also saves current `time` and `visible_enemy_count`. Filters run
after rollout labels are computed and before writing each shard; they do not
change teacher actions or privileged labels. Empty filtered shards are skipped.

This is the data path for Midgame Decisive Trajectory Imitation:

```text
A1: fixed v5 / rollout-search winning terminal windows
A2: active U-Net winning/contact terminal windows
A3: Plan-Q oracle best-plan leads-to-win windows

primary windows:
  terminal - 120 to terminal
  contact-heavy midgame states
  turn 80 to 180 gather/attack transition
```

The immediate training target is policy-coupled U-Net strategy supervision:

```text
policy KL anchor: 1.0
teacher action CE: 0.3
finish: 0.5
outcome: 0.4
enemy-general belief: 0.25
intent: 0.2
```

Current trainer uses the binary `finish_within_250` head; the saved dataset
already includes `finish_within_50/100/250` for a later multi-horizon finish
head.

### Midgame decisive trajectory imitation v0

Collected GPU shards under ignored `runs/adaptive-strategy-midgame-decisive-v0/`:

| source | shards | rows | known wins | draws | finish50 / 100 / 250 | median turn | notes |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `unet-v4-expander-win` | 4 | 4059 | 4059 | 0 | `2180 / 3659 / 4059` | 236 | 8/12/16 active U-Net wins vs Expander |
| `fixed-v5-expander-win` | 4 | 5432 | 5432 | 0 | `3379 / 5102 / 5432` | 134 | fixed v5 8x8 wins vs Expander |
| `fixed-v5-vs-v5-draw` | 2 | 4853 | 0 | 4853 | `0 / 0 / 0` | 193 | draw-heavy contact states |
| `unet-v4-vs-v5-win` | 8 | 3168 | 3168 | 0 | not separately aggregated | contact terminal window | active U-Net true wins vs fixed v5 |

Training probes:

| model | init | data | key settings | fixed-v5 max250 64-row | fixed-v5 max250 256-row | Expander smoke | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `adaptive-midgame-decisive-imitation-v0` | `adaptive-strategy-spatial-v1` | first 3 dirs | KL `1`, CE `0.3`, finish/outcome/belief/intent | min `3.12%` | not run | Expander 64 min `71.88%` | inherited spatial bias, not useful |
| `adaptive-midgame-decisive-imitation-v1` | `adaptive-unet-ppo-v4` | first 3 dirs | KL `2`, CE `0.2`, lr `2e-6` | min `15.62%` | min `10.16%` | Expander 64 min `71.88%` | 64-row false positive |
| `adaptive-midgame-decisive-imitation-v2` | `adaptive-unet-ppo-v4` | all 4 dirs capped to 512 rows/shard | KL `2`, CE `0.3`, includes v5-win rows | min `14.06%` | min `12.11%` | Expander 64 min `70.31%` | weak positive vs same-seed base, not promotion |
| `adaptive-midgame-decisive-imitation-v3` | `adaptive-unet-ppo-v4` | win-only dirs | KL `1`, CE `0.5`, no outcome | min `12.50%` | min `9.77%` | not run | finish head learned but gameplay regressed |
| `adaptive-midgame-decisive-imitation-v2-cont` | v2 | all 4 dirs capped | continue lr `1e-6` | min `10.94%` | not run | not run | continuing v2 did not help |

Same-seed active-base control for fixed-v5 max250 256-row:

```text
runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx
seed 94600:
  p0 25/95/136 = 9.77%
  p1 24/97/135 = 9.38%
  min 9.38%
```

Interpretation:

```text
Midgame decisive filtering works and gives dense, relevant contact/terminal rows.
Adding true U-Net-vs-v5 win trajectories improves fixed-v5 256-row slightly over
the same-seed active base, but the gain is too small and Expander 64-row drops.

The current binary finish head is a bottleneck: mixed win/draw training keeps
finish accuracy near the negative-class baseline for many epochs, while win-only
training learns finish but does not improve gameplay. The next iteration should
either add a multi-horizon finish head (50/100/250 BCE logits) or upweight/oversample
true wins-vs-v5 before any more full-policy imitation.

Do not promote v0-v3 or v2-cont. Best diagnostic artifact is v2, but only as a
weak positive data-quality signal.
```

### Multi-horizon finish head

Implemented a backward-compatible strategy finish head size:

```text
adaptive_network.py:
  strategy_finish_outputs default 2
  strategy_finish_outputs=3 for multi-horizon finish logits
  old 2-logit checkpoints can be prefix-copied into a 3-logit target

adaptive_strategy_supervised.py:
  --finish-head-mode binary
  --finish-head-mode multi-horizon
  --init-finish-head-mode binary|multi-horizon

evaluate_adaptive_policy.py:
  --strategy-finish-outputs 3

train_adaptive.py:
  --init-strategy-finish-outputs 3
```

GPU smoke confirmed `adaptive-unet-ppo-v4` can warm-start into a 3-logit finish
head:

```text
strategy_finish_linear2.weight.shape = (3, 64)
```

Training probes:

| model | data | finish mode | key settings | fixed-v5 max250 | verdict |
| --- | --- | --- | --- | --- | --- |
| `adaptive-midgame-decisive-imitation-v4-mh` | v2 mixed data | 50/100/250 BCE | KL `2`, CE `0.3`, outcome `0.4` | 64-row min `10.94%`; 256-row min `8.98%` | head learns, gameplay worse than v2 |
| `adaptive-midgame-decisive-imitation-v5-v5win-mh` | true U-Net-vs-v5 wins only | 50/100/250 BCE | KL `1.5`, CE `0.6`, no outcome/intent | 64-row min `7.81%` | conservative imitation still does not improve gate |

The multi-horizon head fixes the training pathology: mixed data reaches
finish-label accuracy `69.9%`, while true-v5-win-only reaches `79.6%`. Gameplay
still does not improve, so the next bottleneck is not the finish head alone.

Updated diagnosis:

```text
Midgame decisive filtering is good infrastructure.
Multi-horizon finishability is learnable and should be kept.
Current teacher trajectories are too narrow and do not teach actions that beat v5 reliably.
Next useful data should come from rollout-search winning trajectories or accepted Plan-Q oracle
executions, not only vanilla U-Net/v5 sampled winning windows.
```

## 2026-06-17 Frozen Strategy-Head Supervision v0

Implemented `examples/_experimental/ppo/adaptive_strategy_supervised.py`, a first offline trainer for strategy heads on top of the validated U-Net imitation v3 base.

Training setup:

```text
init:
  runs/adaptive-unet-imitation-v3/generals-adaptive-unet-imitation-v3.eqx

datasets:
  runs/adaptive-strategy-dataset-v0/unet-v3-expander/*.npz
  runs/adaptive-strategy-dataset-v0/fixed-v5-max250/*.npz

network:
  U-Net 64,96,128,64
  input_channels 35
  global_context + scoreboard_history + fog_memory
  per-size HL-Gauss value heads
  outcome head loaded but not trained in v0

update scope:
  strategy_intent_linear2
  strategy_finish_linear2
  strategy_enemy_general_conv
  strategy_q heads receive no loss yet
  trunk/policy/value frozen by gradient mask
```

GPU smoke:

```text
command:
  1024 samples, 3 epochs, CUDA_VISIBLE_DEVICES=0

result:
  Loss 15.3574 -> 15.0802
  Intent loss 6.6846 -> 6.3125
  Finish loss 0.4217 -> 0.4077
  Belief BCE 7.9549 -> 7.6889
```

The smoke used `--outcome-weight 0.4` and exposed a useful calibration issue:

```text
Outcome loss stayed very high at about 28.4.
The label encoding is correct (0/1/2), so the issue is the warm-start outcome head being poorly calibrated on fixed-v5 draw-heavy states.
Decision: keep outcome_weight at 0 for the first frozen-head stage and train outcome later with balanced shards or a fresh head.
```

Full v0 head training:

```text
command:
  all v0 shards
  6208 samples
  20 epochs
  minibatch_size 512
  lr 1e-4
  outcome_weight 0

artifact:
  runs/adaptive-strategy-supervised-v0/generals-adaptive-strategy-supervised-v0.eqx
```

Training curve:

| epoch | loss | intent loss / acc | finish loss / acc | belief BCE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `2.3954` | `4.5516 / 0.2%` | `2.4715 / 11.1%` | `1.6549` |
| 5 | `1.1971` | `2.6910 / 0.3%` | `1.0957 / 11.9%` | `0.7353` |
| 10 | `0.5513` | `1.3896 / 76.6%` | `0.4589 / 89.0%` | `0.2993` |
| 20 | `0.3566` | `0.8600 / 76.4%` | `0.3583 / 89.0%` | `0.1375` |

Policy preservation check:

```text
128 held dataset states
base:    U-Net imitation v3 without strategy aux
trained: strategy-supervised v0 with strategy aux

max_policy_logit_abs_diff:  0.0
mean_policy_logit_abs_diff: 0.0
```

Interpretation:

```text
The v0 strategy labels are learnable, and the frozen-head update path is safe.
This checkpoint is not expected to improve gameplay yet because policy logits are identical to U-Net imitation v3.
The next useful step is to expand the dataset with fixed-v5 max500/max750 and more max250 draw/decisive rows, then run policy-coupled strategy training where the U-Net trunk/policy is allowed a small KL-anchored update.
```

## 2026-06-17 Policy-Coupled Strategy Supervision v1

Extended `adaptive_strategy_supervised.py` with:

```text
--update-scope strategy-heads|all
--policy-kl-weight
--action-ce-weight
--max-samples-per-shard
```

`strategy-heads` remains the default frozen mode. `all` updates the full network, but is guarded by a required positive policy KL weight so trunk/policy coupling cannot run without an action-distribution anchor. `--max-samples-per-shard` randomly caps each shard before concatenation, which is needed because long fixed-v5 rollouts otherwise dominate mixed offline batches.

Expanded dataset shards:

| shard | samples | finish250 | draw risk | contact | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `fixed-v5-max500-00000.npz` | 8320 | `0.292` | `0.180` | `0.903` | 21 episodes, 11 learner wins, 3 draws |
| `fixed-v5-max750-00000.npz` | 12160 | `0.297` | `0.000` | `0.909` | 29 episodes, 17 learner wins, 0 draws |
| `v4-expander-00000.npz` | 8192 | `0.213` | `0.000` | `0.704` | v4 teacher anchor, 14 episodes, 11 wins |
| `v4-expander-balanced-00000.npz` | 8192 | `0.187` | `0.000` | `0.627` | v4 balanced anchor |
| `v4-expander-balanced-00001.npz` | 8192 | `0.148` | `0.000` | `0.835` | v4 balanced anchor |
| `v4-expander-balanced-00002.npz` | 8192 | `0.164` | `0.145` | `0.837` | v4 balanced anchor with some draw states |

### Coupled v1 from U-Net imitation v3 heads

Training:

```text
init:
  runs/adaptive-strategy-supervised-v0/generals-adaptive-strategy-supervised-v0.eqx
datasets:
  v3 Expander v0
  fixed-v5 max250/max500/max750
epochs: 10
lr: 5e-7
loss:
  policy_kl 1.0
  action_ce 0.05
  intent 0.1
  finish 0.2
  belief 0.1
artifact:
  runs/adaptive-strategy-coupled-v1/generals-adaptive-strategy-coupled-v1.eqx
```

Training curve:

```text
KL 0.6248 -> 0.5333
ActCE 1.0973 -> 1.0354
teacher action match 64.5% -> 65.0%
finish loss 0.7564 -> 0.6744
```

Expander 256-row, seed 81100:

| checkpoint | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | `75.00%` | `75.00%` | `78.52%` | `76.56%` | `77.34%` | `75.78%` | `75.00%` |
| v4 | `73.83%` | `78.52%` | `82.42%` | `81.25%` | `77.73%` | `78.12%` | `73.83%` |
| coupled v1 | `76.56%` | `79.69%` | `81.25%` | `82.42%` | `76.95%` | `76.95%` | `76.56%` |

Expander 512-row, seed 81140:

| checkpoint | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 | `73.63%` | `75.59%` | `85.35%` | `83.01%` | `80.27%` | `79.10%` | `73.63%` |
| coupled v1 | `74.80%` | `75.98%` | `83.79%` | `84.18%` | `77.34%` | `74.61%` | `74.61%` |

Fixed-v5 max250, 256-row, seed 81120:

| checkpoint | p0 | p1 | min | draw p0/p1 |
| --- | ---: | ---: | ---: | --- |
| v3 | `8.98%` | `7.42%` | `7.42%` | `55.47% / 55.47%` |
| coupled v1 | `10.55%` | `10.94%` | `10.55%` | `54.30% / 57.81%` |

Interpretation:

```text
Coupling from v3 is a real method signal: it improved same-seed Expander 256 and 512 min and improved fixed-v5 over v3.
It is not a promotion candidate because it weakens v4's 16x rows and remains far below the fixed-v5 best gate.
```

### Coupled from v4 with unbalanced v4 anchor

Frozen heads:

```text
init: v4
datasets: one v4 Expander shard + fixed-v5 max250/max500/max750
samples: 32832
intent acc: 16.9% -> 78.9%
finish acc: 61.8% -> 66.7%
belief BCE: 0.1176 -> 0.0789
artifact: runs/adaptive-strategy-heads-v4-v1/generals-adaptive-strategy-heads-v4-v1.eqx
```

Coupled training:

```text
artifact: runs/adaptive-strategy-coupled-v4-v1/generals-adaptive-strategy-coupled-v4-v1.eqx
KL 0.4801 -> 0.4134
teacher action match stayed about 70%
```

Results:

| eval | p0/8p0 | p1/8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Expander 256 | `70.70%` | `76.17%` | `87.11%` | `79.69%` | `76.95%` | `77.73%` | `70.70%` |
| fixed-v5 max250 256 | `8.98%` | `15.23%` | - | - | - | - | `8.98%` |

Interpretation:

```text
Unbalanced v4-coupled training overfits the mixed offline objective and hurts 8p0 badly.
Do not continue this exact recipe.
```

### Balanced v4-coupled probe

Frozen heads:

```text
init: v4
datasets: 3 v4 Expander shards + fixed-v5 max250/max500/max750
max_samples_per_shard: 4096
samples: 24576
intent acc: 15.6% -> 79.1%
finish acc: 35.9% -> 66.5%
belief BCE: 0.2002 -> 0.0835
artifact: runs/adaptive-strategy-heads-v4-balanced-v1/generals-adaptive-strategy-heads-v4-balanced-v1.eqx
```

Coupled training:

```text
artifact: runs/adaptive-strategy-coupled-v4-balanced-v1/generals-adaptive-strategy-coupled-v4-balanced-v1.eqx
KL 0.3131 -> 0.2805
teacher action match 80.0% -> 78.5%
```

Expander 256-row, seed 81320:

| checkpoint | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 | `75.39%` | `73.83%` | `81.64%` | `79.30%` | `80.47%` | `80.08%` | `73.83%` |
| balanced coupled | `76.56%` | `73.44%` | `80.08%` | `82.03%` | `79.69%` | `79.30%` | `73.44%` |

Fixed-v5 max250, 256-row, seed 81340:

| checkpoint | p0 | p1 | min | draw p0/p1 |
| --- | ---: | ---: | ---: | --- |
| v4 | `13.28%` | `10.55%` | `10.55%` | `44.92% / 53.12%` |
| balanced coupled | `11.72%` | `9.77%` | `9.77%` | `44.92% / 53.91%` |

Conclusion:

```text
Balanced policy-coupled supervision preserves 16x better than the unbalanced run, but it still does not beat v4 on 256-row Expander or fixed-v5.
The useful artifact is the trainer/data infrastructure, not the checkpoints.
Current offline action KL/CE is still too blunt: it can reduce KL and learn heads, but it does not reliably convert finish/belief/intent representation into stronger decisions.
Next direction should make strategy heads affect inference explicitly, e.g. finish/Q reranking, target-conditioned action bias, or search-labeled replacement-outcome heads, rather than pushing the whole trunk with plain action KL.
```

## 2026-06-17 Frozen Strategy-Q Rerank Probe

Extended `adaptive_strategy_supervised.py` with direct strategy-Q supervision:

```text
--q-kl-weight
--q-action-ce-weight
```

The loss treats `strategy_auxiliary().action_q_values` as logits over the padded adaptive action space, masks illegal actions using the stored teacher logits, and trains:

```text
QKL: KL(teacher_logits || strategy_q_logits)
QCE: CE(teacher_sample_action)
```

This keeps the main policy frozen and uses `evaluate_adaptive_policy.py --strategy-q-rerank-scale` to apply the learned Q head as a centered legal-action bias at inference time.

Training:

```text
init:
  runs/adaptive-strategy-heads-v4-balanced-v1/generals-adaptive-strategy-heads-v4-balanced-v1.eqx

datasets:
  v4-expander-balanced x3
  fixed-v5 max250/max500/max750

sampling:
  max_samples_per_shard 4096

loss:
  q_kl 1.0
  q_action_ce 0.05
  intent 0.05
  finish 0.1
  belief 0.05

artifact:
  runs/adaptive-strategy-q-rerank-v1/generals-adaptive-strategy-q-rerank-v1.eqx
```

Training curve:

```text
QKL:       5.4646 -> 0.5882
QCE:       6.7705 -> 1.4958
Q action:  10.7%  -> 58.9%
Intent:    78.8%  -> 81.2%
Finish:    67.3%  -> 70.8%
Belief BCE 0.0806 -> 0.0589
```

Expander 128-row, seed 81420:

| scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `71.88%` | `78.91%` | `84.38%` | `84.38%` | `84.38%` | `81.25%` | `71.88%` |
| 0.02 | `69.53%` | `78.12%` | `84.38%` | `83.59%` | `82.81%` | `78.12%` | `69.53%` |
| 0.05 | `73.44%` | `75.78%` | `86.72%` | `76.56%` | `84.38%` | `80.47%` | `73.44%` |
| 0.10 | `72.66%` | `71.88%` | `80.47%` | `76.56%` | `83.59%` | `80.47%` | `71.88%` |

Expander 256-row, seed 81460:

| scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `79.69%` | `74.61%` | `85.16%` | `79.69%` | `73.44%` | `81.64%` | `73.44%` |
| 0.05 | `77.73%` | `75.00%` | `83.20%` | `82.42%` | `76.56%` | `82.81%` | `75.00%` |

Expander 512-row, seed 81480:

| scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `71.88%` | `76.56%` | `82.23%` | `83.59%` | `80.27%` | `78.32%` | `71.88%` |
| 0.05 | `71.68%` | `75.78%` | `82.42%` | `84.38%` | `80.47%` | `76.95%` | `71.68%` |

Fixed-v5 max250, 128-row, seed 81440:

| scale | p0 | p1 | min | draw p0/p1 |
| ---: | ---: | ---: | ---: | --- |
| 0 | `10.94%` | `14.06%` | `10.94%` | `46.88% / 47.66%` |
| 0.05 | `10.94%` | `11.72%` | `10.94%` | `46.88% / 50.00%` |
| 0.10 | `10.16%` | `11.72%` | `10.16%` | `45.31% / 50.78%` |

Conclusion:

```text
The strategy-Q head clearly learns the offline teacher distribution, but direct centered all-action reranking does not pass promotion.
Scale 0.05 looked promising at 128 and 256 rows, but 512-row confirmation regressed slightly versus scale 0.
Fixed-v5 max250 did not improve; rerank changed draw/loss mix but did not create wins.
Do not promote q-rerank-v1. The next strategy-inference branch should be more structured: target-conditioned movement bias, finish-only gating, or replacement-outcome search Q, rather than a global action-logit bias over every step.
```

## 2026-06-17 Target-Conditioned Rerank Probe

Added inference-only target-conditioned reranking to `evaluate_adaptive_policy.py`:

```text
--strategy-target-rerank-scale
--strategy-target-finish-gate
```

Mechanism:

```text
1. Use strategy_auxiliary().enemy_general_logits as a belief map.
2. Convert the belief map to an expected target coordinate.
3. For each legal primitive move, compute:
     distance(source, target) - distance(destination, target)
4. Center that progress score over legal actions and add it to policy logits.
5. Optionally multiply the bias by finish-head P(finish_within_250).
```

This probe is more structured than global Q reranking: it only rewards moves that make spatial progress toward the learned target belief, and it does not update the checkpoint.

Expander 128-row, seed 81520:

| mode | scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ungated | 0.10 | `72.66%` | `76.56%` | `83.59%` | `88.28%` | `74.22%` | `79.69%` | `72.66%` |
| ungated | 0.25 | `75.00%` | `73.44%` | `78.91%` | `89.84%` | `70.31%` | `78.12%` | `70.31%` |
| ungated | 0.50 | `75.78%` | `74.22%` | `83.59%` | `82.03%` | `77.34%` | `78.12%` | `74.22%` |
| finish-gated | 0.10 | `71.09%` | `73.44%` | `86.72%` | `83.59%` | `77.34%` | `78.91%` | `71.09%` |
| finish-gated | 0.25 | `70.31%` | `78.12%` | `85.16%` | `85.94%` | `73.44%` | `80.47%` | `70.31%` |
| finish-gated | 0.50 | `78.12%` | `74.22%` | `82.81%` | `92.19%` | `73.44%` | `81.25%` | `73.44%` |

Expander 256-row confirmation, seed 81540:

| scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `76.56%` | `75.78%` | `83.59%` | `82.03%` | `77.73%` | `82.81%` | `75.78%` |
| target 0.50 | `73.44%` | `74.61%` | `85.94%` | `82.42%` | `81.64%` | `80.86%` | `73.44%` |

Fixed-v5 max250 128-row, seed 81560:

| scale | p0 | p1 | min | draw p0/p1 |
| ---: | ---: | ---: | ---: | --- |
| 0 | `10.16%` | `5.47%` | `5.47%` | `57.81% / 56.25%` |
| target 0.50 | `9.38%` | `6.25%` | `6.25%` | `60.16% / 52.34%` |

Conclusion:

```text
Target-conditioned reranking is directionally interpretable but still not promotable.
The best 128-row result was ungated scale 0.50, but 256-row confirmation showed it trades away 8x reliability for 16x gains.
Finish gating did not stabilize the bias, probably because the finish head is not calibrated well enough for gating primitive movement.
Fixed-v5 remained far below the gate; the p1 tick-up was too small and too noisy to matter.
Next step should learn a target/source head or replacement-outcome correction directly, rather than deriving a hand-coded Manhattan target bias from enemy-general belief.
```

## 2026-06-17 Explicit Source/Target Spatial Heads

Added optional source/target spatial strategy heads:

```text
Adaptive*PolicyValueNetwork:
  --strategy-aux
  --strategy-spatial-aux

StrategyAuxOutputs:
  source_logits
  target_logits

adaptive_strategy_supervised.py:
  --source-weight
  --target-weight
  --init-strategy-spatial-aux

evaluate_adaptive_policy.py:
  --strategy-spatial-aux
  --strategy-spatial-rerank-scale
```

The new `strategy_spatial_aux` flag is separate from `strategy_aux`, so older intent/finish/belief/Q checkpoints can still load with `init_strategy_spatial_aux=False` and expand only the new 1x1 source/target convs. This avoids invalidating `adaptive-strategy-q-rerank-v1` and prior strategy head checkpoints.

Training run:

```text
model: runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx
init:  runs/adaptive-strategy-q-rerank-v1/generals-adaptive-strategy-q-rerank-v1.eqx
data:  v4-expander-balanced + fixed-v5 max250/max500/max750
cap:   4096 samples/shard, 24576 samples total
loss:  source=0.5, target=0.5, all other losses 0
scope: frozen base, strategy heads only
device: cuda:0
```

Training curve:

| epoch | source CE / acc | target CE / acc |
| ---: | ---: | ---: |
| 1 | `6.1933 / 0.0%` | `5.1836 / 0.4%` |
| 4 | `3.4716 / 13.4%` | `4.5406 / 1.6%` |
| 8 | `3.1697 / 16.1%` | `4.3413 / 5.8%` |

The source head learns a real signal. The target head is much harder; this is expected because the current label is usually the true enemy general cell, which is often hidden and underdetermined from short fog-memory context.

Expander 128-row, seed 82040:

| spatial scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `70.31%` | `70.31%` | `80.47%` | `80.47%` | `81.25%` | `85.94%` | `70.31%` |
| 0.05 | `75.00%` | `75.78%` | `82.81%` | `83.59%` | `80.47%` | `78.12%` | `75.00%` |
| 0.10 | `69.53%` | `75.78%` | `78.12%` | `82.03%` | `79.69%` | `80.47%` | `69.53%` |

Expander 256-row confirmation, seed 82060:

| spatial scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `76.95%` | `73.83%` | `82.03%` | `83.59%` | `80.86%` | `80.47%` | `73.83%` |
| 0.05 | `73.05%` | `77.73%` | `81.25%` | `86.33%` | `81.64%` | `78.12%` | `73.05%` |

Fixed-v5 max250 128-row, seed 82080:

| spatial scale | p0 | p1 | min | draw p0/p1 |
| ---: | ---: | ---: | ---: | --- |
| 0 | `10.94%` | `11.72%` | `10.94%` | `52.34% / 55.47%` |
| 0.05 | `10.16%` | `9.38%` | `9.38%` | `55.47% / 55.47%` |

Conclusion:

```text
Do not promote adaptive-strategy-spatial-v1.
Explicit source/target heads are wired and trainable, but the current direct rerank still behaves like a diagnostic bias rather than a reliable policy improvement.
The 128-row Expander gain at scale 0.05 was another short-sample false positive; 256-row confirmation regressed versus scale 0.
Fixed-v5 max250 worsened and draw did not fall.

Next aligned step:
  keep the source/target heads as supervised probes,
  improve target labels from true-general one-hot to richer target/intent maps,
  or train replacement-outcome/search target heads before using spatial bias in inference.
```

## 2026-06-17 Adaptive Plan-Q Dataset v0

Added `adaptive_plan_q_dataset.py`, a source-target counterfactual shard collector.

Purpose:

```text
Stop training source/target as static CE labels.
Score source-target plans by short replacement rollout.
Save plan_q, plan_scores, plan_outcomes, source_score_probs, and target_score_probs.
Use these shards for source_q / target_q / plan_q supervision next.
```

Mechanism:

```text
1. Collect mixed-seat adaptive states from a base checkpoint rollout.
2. Pick source candidates from owned movable cells ranked by army mass.
3. Pick target candidates from enemy general, enemy cells, cities, and passable cells.
4. For each source-target pair:
     force the first primitive move toward the target
     roll out the base adaptive policy for a short horizon
     score final material/land/terminal state
5. Convert scores to plan_q = tanh(score / score_scale).
6. Save source/target marginals from softmax(plan_q / temperature).
```

The first implementation deliberately keeps execution simple: only the first action is plan-conditioned; subsequent actions use the base policy and the configured opponent. This isolates plan scoring from a new Worker head.

Smoke 1:

```text
command: 4 envs, 2 steps, 2x2 plans, 2-step rollout
model:   runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx
device:  cuda:0
output:  runs/adaptive-plan-q-smoke/smoke-00000.npz
result:  compiled and wrote 8 samples, but all outcomes draw and mean_gap=0.0000
```

The zero gap was not a collector failure. Raw scores were tiny because the default `score_scale=1000` was appropriate for terminal outcomes but too large for short nonterminal material/land scores.

Smoke 2:

```text
command: 8 envs, 16 steps, 4x4 plans, 8-step rollout
score_scale: 10
output: runs/adaptive-plan-q-v0-scale10/plan-q-00000.npz
samples: 128
```

Statistics:

| metric | value |
| --- | ---: |
| raw plan score min / mean / max | `-1.12 / 0.41 / 5.27` |
| plan_q min / mean / max | `-0.112 / 0.040 / 0.483` |
| mean plan_q_gap | `0.0689` |
| mean best_plan_q | `0.1094` |
| source max-prob mean | `0.2700` |
| target max-prob mean | `0.2704` |
| best plan win/draw | `0.000 / 1.000` |

Interpretation:

```text
The v0 collector produces non-uniform source/target Plan-Q marginals, so it is usable for ranking-supervision plumbing.
However, the 8-step horizon still yields no decisive outcome labels, so it is not yet sufficient for fixed-v5 anti-draw training.
Default score_scale is now 10.0 to preserve nonterminal score differences; terminal wins/losses still saturate plan_q.
```

Next data step:

```text
Run longer fixed-v5 max250 shards:
  grid_sizes=8
  opponent_policy_path=generals-ppo-8x8-expander-gpu-v5.eqx
  plan_rollout_steps=32 or 64
  rollouts_per_plan=1 initially
  source_count=4
  target_count=4

Acceptance:
  best_plan_q gap stays nonzero
  best_plan win/loss outcome is not all draw
  source/target marginals have enough entropy to train, but enough peak to rank
```

## 2026-06-17 Plan-Q Source/Target Supervision v0

Added `adaptive_plan_q_supervised.py`, a frozen-head trainer for Plan-Q shards.

Inputs:

```text
obs
legal_mask
active
source_indices
target_indices
source_score_probs
target_score_probs
teacher_logits / teacher_action_index
plan_q_gap
```

Loss:

```text
source_loss = sparse CE over candidate source indices using source_score_probs
target_loss = sparse CE over candidate target indices using target_score_probs
optional policy_kl/action_ce anchors are available for future joint updates
default update_scope=strategy-heads, so trunk/policy stay frozen
```

Smoke command:

```text
dataset: runs/adaptive-plan-q-v0-scale10/*.npz
init:    runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx
output:  runs/adaptive-plan-q-supervised-v0/generals-adaptive-plan-q-supervised-v0.eqx
epochs:  8
batch:   64
lr:      3e-4
device:  cuda:0
```

Training curve:

| epoch | source loss / acc | target loss / acc |
| ---: | ---: | ---: |
| 1 | `3.0467 / 19.5%` | `4.3221 / 2.3%` |
| 4 | `2.9552 / 22.7%` | `4.2447 / 2.3%` |
| 8 | `2.8660 / 28.1%` | `4.1629 / 1.6%` |

Interpretation:

```text
Plan-Q source marginals are learnable even on the tiny 128-row smoke shard.
Plan-Q target marginals also reduce CE but top1 accuracy remains weak, matching the earlier diagnosis that target choice is the harder strategic variable.
This checkpoint is not a promotion candidate because only auxiliary heads changed and the dataset lacks decisive outcome labels.
The next useful training data should be longer fixed-v5 max250 Plan-Q shards, then rerun this trainer with gap-weighting or stronger target supervision.
```

## 2026-06-17 Fixed-v5 Warmed Plan-Q Shard

Updated `adaptive_plan_q_dataset.py` with:

```text
--warmup-steps
truncation-aware counterfactual plan rollouts
```

Reason:

```text
Opening states cannot produce useful fixed-v5 max250 finish labels under short plan rollouts.
Warmup advances behavior games first, then spends counterfactual budget near the finish/draw decision region.
Plan rollouts now stop at --truncation, so wins after max250 are not counted as wins.
```

Fixed-v5 warm shard:

```text
output: runs/adaptive-plan-q-fixed-v5-warm190-v0/plan-q-00000.npz
model:  runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx
opponent: generals-ppo-8x8-expander-gpu-v5.eqx sample
grid: 8x8
truncation: 250
warmup_steps: 190
samples: 64
plans/state: 4x4
plan_rollout_steps: 64
rollouts/plan: 1
device: cuda:0
```

Statistics:

| metric | value |
| --- | ---: |
| behavior state time min / mean / max | `12 / 139.1 / 198` |
| plan outcomes | `186 loss / 838 draw / 0 win` |
| mean plan_q_gap | `0.2712` |
| mean best_plan_q | `0.1652` |
| source max-prob mean | `0.3479` |
| target max-prob mean | `0.3157` |

Interpretation:

```text
Warmup and longer rollouts produced strong safety ranking signal: many bad plans lose, best plans avoid loss.
They still did not produce winning plans against fixed-v5 max250.
This data can teach source/target heads to avoid losing plans, but not yet to finish games.
The low minimum behavior time means some rows reset during warmup; future shards should either keep reset rows marked or collect more rows to dilute resets.
```

Gap-weighted Plan-Q supervision v1:

```text
dataset: runs/adaptive-plan-q-fixed-v5-warm190-v0/*.npz
init: runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx
output: runs/adaptive-plan-q-fixed-v5-supervised-v1/generals-adaptive-plan-q-fixed-v5-supervised-v1.eqx
epochs: 12
gap_weighting: true
```

Training curve:

| epoch | source loss / candidate acc / grid acc | target loss / candidate acc / grid acc |
| ---: | ---: | ---: |
| 1 | `4.2827 / 28.1% / 0.0%` | `4.1558 / 28.1% / 0.0%` |
| 6 | `4.0483 / 25.0% / 0.0%` | `3.9930 / 25.0% / 0.0%` |
| 12 | `3.8626 / 26.6% / 0.0%` | `3.8177 / 28.1% / 0.0%` |

Conclusion:

```text
Plan-Q heads can fit fixed-v5 safety marginals, but this shard is not sufficient for anti-draw because it has no winning plans.
Do not evaluate this checkpoint as a promotion model; policy logits are unchanged.
Next useful change is to make the plan executor stronger, not just longer:
  add plan-conditioned multi-step Worker actions,
  or score candidates with rollout-search / stronger policy after the first forced move,
  or sample states from successful fixed-v5 imitation trajectories where winning plans exist.
```

## 2026-06-17 Plan-Q Worker Probe

Updated `adaptive_plan_q_dataset.py` with:

```text
--plan-worker-steps
```

Mechanism:

```text
Each plan still starts with the forced source -> target primitive move.
For the next N rollout steps, a deterministic worker reselects a live owned source near the original route and moves it toward the same target.
After N worker steps, rollout returns to the base policy.
Default N=0 preserves the old first-step-only Plan-Q collector.
```

Reason:

```text
The first-step-only fixed-v5 warm shard produced safety labels but no winning plans.
That means source/target heads could learn which plans avoid loss, but not which plans create a finish.
Multi-step plan execution is the smallest bridge toward a target-conditioned Worker without changing the policy network yet.
```

Fixed-v5 worker shard:

```text
output: runs/adaptive-plan-q-fixed-v5-worker-v0/plan-q-00000.npz
model: runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx
opponent: generals-ppo-8x8-expander-gpu-v5.eqx sample
grid: 8x8
truncation: 250
warmup_steps: 190
samples: 64
plans/state: 4x4
plan_rollout_steps: 64
plan_worker_steps: 16
rollouts/plan: 1
device: cuda:0
```

Statistics:

| metric | value |
| --- | ---: |
| behavior state time min / mean / max | `5 / 122.9 / 198` |
| all plan outcomes | `339 loss / 683 draw / 2 win` |
| best-plan outcomes | `4 loss / 58 draw / 2 win` |
| mean plan_q_gap | `0.3770` |
| mean best_plan_q | `-0.0748` |
| source entropy mean | `1.283` |
| target entropy mean | `1.160` |

Gap-weighted worker Plan-Q supervision:

```text
dataset: runs/adaptive-plan-q-fixed-v5-worker-v0/plan-q-00000.npz
init: runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx
output: runs/adaptive-plan-q-fixed-v5-worker-supervised-v0/generals-adaptive-plan-q-fixed-v5-worker-supervised-v0.eqx
epochs: 80
gap_weighting: true
```

Training curve:

| epoch | source loss / candidate acc / grid acc | target loss / candidate acc / grid acc |
| ---: | ---: | ---: |
| 1 | `4.3097 / 25.0% / 4.7%` | `4.6764 / 50.0% / 0.0%` |
| 40 | `3.9851 / 25.0% / 4.7%` | `4.3033 / 39.1% / 0.0%` |
| 80 | `3.7872 / 25.0% / 4.7%` | `3.9762 / 29.7% / 3.1%` |

Interpretation:

```text
Worker-conditioned counterfactuals produced the first winning plan labels in this Plan-Q pipeline.
The effect is real but still sparse: only 2/1024 plans won, so the supervised checkpoint is not a promotion model.
Next step is larger worker-conditioned collection, likely mixing fixed-v5 max250/max500 and keeping enough near-terminal successful rows for finish labels.
```

## 2026-06-17 Plan-Q Action-Q Supervision

Fixed a Plan-Q label bug:

```text
plan_action_indices previously encoded pass as 8*pad*pad + row*pad + col.
The adaptive policy action space has exactly one global pass index: 8*pad*pad.
This polluted action-Q supervision with invalid indices >= 2049 on 16-padded boards.
```

Corrected shard smoke:

```text
output: runs/adaptive-plan-q-action-index-smoke/smoke-00000.npz
plan_action_indices min/max: 37 / 2048
bad indices >= 2049: 0
teacher_action_index bad indices >= 2049: 0
```

Larger corrected worker shard:

```text
output: runs/adaptive-plan-q-fixed-v5-worker-v2/plan-q-00000.npz
grid: 8x8
truncation: 250
warmup_steps: 190
samples: 512
plans/state: 4x4
plan_rollout_steps: 64
plan_worker_steps: 16
rollouts/plan: 1
device: cuda:0
```

Statistics:

| metric | value |
| --- | ---: |
| action index min / max | `16 / 2048` |
| bad action indices >= 2049 | `0` |
| pass plan actions | `2096` |
| all plan outcomes | `3138 loss / 4917 draw / 137 win` |
| best-plan outcomes | `36 loss / 419 draw / 57 win` |
| mean plan_q_gap | `0.4299` |
| mean best_plan_q | `-0.0933` |

Action-Q trainer update:

```text
adaptive_plan_q_supervised.py now supports:
  --action-q-weight
  --action-q-mse-weight
  --action-q-temperature

The loss aggregates duplicate source-target plan slots onto the same primitive action with scatter-add,
then trains strategy action-Q with full legal-action ranking CE plus optional candidate MSE.
```

Action-Q v1 training:

```text
dataset: runs/adaptive-plan-q-fixed-v5-worker-v2/plan-q-00000.npz
init: runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx
output: runs/adaptive-plan-q-action-q-v1/generals-adaptive-plan-q-action-q-v1.eqx
loss: source=0, target=0, action_q=1.0, action_q_mse=0.1
epochs: 80
```

Training curve:

| epoch | action-Q rank / MSE / action acc | action-Q pred gap |
| ---: | ---: | ---: |
| 1 | `12.9955 / 9.8694 / 9.7%` | `5.569` |
| 40 | `6.4975 / 7.5601 / 25.7%` | `4.903` |
| 80 | `5.8049 / 5.3826 / 27.1%` | `4.576` |

Fixed-v5 max250 128-row scale sweep:

| q scale | p0 win | p1 win | min | draw notes |
| ---: | ---: | ---: | ---: | --- |
| `0` | `11.72%` | `13.28%` | `11.72%` | p0 draw `57.03%`, p1 draw `60.16%` |
| `0.01` | `12.50%` | `12.50%` | `12.50%` | draw increased |
| `0.02` | `10.16%` | `13.28%` | `10.16%` | worse p0 |
| `0.05` | `10.94%` | `13.28%` | `10.94%` | worse p0 |

Fixed-v5 max250 256-row confirmation:

| q scale | p0 win | p1 win | min | draw |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `12.11%` | `10.94%` | `10.94%` | p0 `54.69%`, p1 `61.33%` |
| `0.01` | `10.55%` | `11.72%` | `10.55%` | p0 `54.69%`, p1 `60.16%` |

Conclusion:

```text
Action-Q from worker-conditioned Plan-Q is learnable, and full-action aggregation is materially better than slot-wise plan CE.
Direct Q-rerank still fails fixed-v5 promotion: the 128-row weak positive at scale 0.01 did not confirm at 256 rows.
Do not promote adaptive-plan-q-action-q-v1.
Next step should make the learned plan signal execute through a target-conditioned Worker/mixture policy or train on larger, more decisive Plan-Q data; do not continue q-rerank scale sweeps.
```

## 2026-06-17 Plan-Q Policy Distillation Probe

Added a direct policy loss to `adaptive_plan_q_supervised.py`:

```text
--plan-policy-weight
```

Mechanism:

```text
Use the same corrected worker Plan-Q action target distribution as action-Q supervision.
Apply CE to policy logits directly.
Require --update-scope all and a positive --policy-kl-weight so the original policy remains an anchor.
```

Probe:

```text
dataset: runs/adaptive-plan-q-fixed-v5-worker-v2/plan-q-00000.npz
init: runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx
output: runs/adaptive-plan-q-plan-policy-v0/generals-adaptive-plan-q-plan-policy-v0.eqx
policy_kl_weight: 1.0
plan_policy_weight: 0.05
lr: 2e-5
epochs: 60
update_scope: all
```

Training:

| epoch | KL | plan-policy CE / top action acc | teacher action acc |
| ---: | ---: | ---: | ---: |
| 1 | `0.0002` | `8.8217 / 9.9%` | `99.4%` |
| 30 | `0.0412` | `5.3374 / 8.9%` | `91.8%` |
| 60 | `0.0348` | `4.9325 / 9.7%` | `94.3%` |

Fixed-v5 max250 128-row:

| seat | wins/losses/draws | win | draw |
| ---: | ---: | ---: | ---: |
| p0 | `14 / 54 / 60` | `10.94%` | `46.88%` |
| p1 | `8 / 49 / 71` | `6.25%` | `55.47%` |

Conclusion:

```text
Direct Plan-Q policy CE is not stable enough on this small shard.
It reduced draw for p0 but damaged p1 badly, matching earlier action-distill/seat-tradeoff failures.
Do not promote adaptive-plan-q-plan-policy-v0.
Next step should be an explicit target-conditioned Worker or mixture policy, where plan choice and execution are separated rather than forcing plan actions into the primitive policy distribution.
```

## 2026-06-17 Strategy Worker Mixture Probe

Added explicit worker execution support to `evaluate_adaptive_policy.py`:

```text
--strategy-worker-mix-prob
--strategy-worker-finish-gate
--strategy-worker-policy-margin
```

Mechanism:

```text
The evaluator samples the base policy as usual.
If worker mix triggers, it selects source and target from the strategy spatial heads.
The worker chooses a legal one-step move from the selected source toward the selected target.
Policy-margin gating only permits the worker action when its base-policy logit is near the best policy action.
Finish gating scales worker probability by P(finish) from the finish head.
```

Fixed-v5 max250 128-row, `adaptive-strategy-spatial-v1`:

| worker setting | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `11.72%` | `12.50%` | `11.72%` | `50.78%` | `53.91%` |
| mix `0.02` | `9.38%` | `8.59%` | `8.59%` | `50.00%` | `53.12%` |
| mix `0.05` | `11.72%` | `9.38%` | `9.38%` | `47.66%` | `51.56%` |
| mix `0.10` | `4.69%` | `3.91%` | `3.91%` | `39.06%` | `47.66%` |
| mix `0.20` | `5.47%` | `3.12%` | `3.12%` | `28.12%` | `39.06%` |
| mix `0.10` finish-gated | `8.59%` | `9.38%` | `8.59%` | `53.12%` | `43.75%` |
| mix `0.20` finish-gated | `3.91%` | `7.81%` | `3.91%` | `46.09%` | `47.66%` |
| mix `0.20`, margin `1` | `3.91%` | `3.12%` | `3.12%` | `57.81%` | `60.94%` |
| mix `0.20`, margin `2` | `6.25%` | `5.47%` | `5.47%` | `52.34%` | `57.03%` |
| mix `0.20`, margin `4` | `9.38%` | `8.59%` | `8.59%` | `57.03%` | `53.91%` |

Conclusion:

```text
The explicit worker lowers draw only by increasing losses.
Policy-margin and finish gates do not recover baseline strength.
This confirms the current source/target spatial heads are not reliable enough to directly drive deterministic execution.
Keep the evaluator worker path as a diagnostic, but do not use it for promotion.
Next useful Worker work needs supervised target-conditioned action data and a learned worker policy, not hand-coded source->target movement from weak spatial heads.
```

## 2026-06-17 Learned Worker Hybrid Probe

`evaluate_worker_policy.py` now supports current U-Net fallback checkpoints with:

```text
--fallback-network-arch unet
--fallback-scoreboard-history
--fallback-fog-memory
```

This matters because the active U-Net v4 Expander base uses 35 input planes
(`15` adaptive base + `5` fog memory + `5` current scoreboard + `10`
previous/delta history). The worker evaluator previously only constructed
15/20/30-channel fallback observations, so it could not load
`runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx`.

### General-target split Worker as rerank bias

Setup:

```text
worker:   runs/adaptive-worker-split-general-v1/generals-adaptive-worker-split-general-v1.eqx
fallback: runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx
opponent: Expander
mode:     fallback sample policy, worker rerank on contact
```

128 games/row on seed `83600`:

| worker scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00` | `71.09%` | `75.78%` | `75.00%` | `85.94%` | `75.00%` | `82.81%` | `71.09%` |
| `0.02` | `70.31%` | `76.56%` | `78.12%` | `86.72%` | `74.22%` | `82.81%` | `70.31%` |
| `0.05` | `70.31%` | `79.69%` | `76.56%` | `84.38%` | `73.44%` | `79.69%` | `70.31%` |
| `0.10` | `72.66%` | `82.03%` | `77.34%` | `89.84%` | `75.78%` | `85.94%` | `72.66%` |

The `0.10` row looked promising at 128 games/row, but failed confirmation.

256 games/row on seed `83620`:

| worker scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00` | `76.95%` | `76.95%` | `87.50%` | `80.08%` | `75.00%` | `83.59%` | `75.00%` |
| `0.10` | `73.83%` | `70.31%` | `85.16%` | `81.64%` | `78.91%` | `81.25%` | `70.31%` |

Conclusion: the general-target learned Worker can move behavior and sometimes
improve large-row finish pressure, but it does not confirm as a stable rerank
bias. The 256-row failure is a seat tradeoff: 16p0 improves, while 8p0/8p1
drop sharply. Do not promote or extend this to fixed-v5 evaluation.

### Mixed-target Worker continuation

To test whether the issue was over-specialization to enemy-general targets, a
mixed-target Worker was trained from `adaptive-worker-split-general-v1`:

```text
artifact: runs/adaptive-worker-split-random-v1/generals-adaptive-worker-split-random-v1.eqx
target_family=random
rollout: 64 envs x 32 steps x 200 iterations
loss: action=0.1, source=1.0, direction=1.0
```

Final supervised metrics:

```text
Acc 23.5%, Src 71.1%, Dir 48.7%, Useful 70.6%, Mass 0.210, Valid 98.8%
```

Expander hybrid, 128 games/row on seed `83600`, rerank scale `0.10`:

| worker | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline scale `0.00` | `71.09%` | `75.78%` | `75.00%` | `85.94%` | `75.00%` | `82.81%` | `71.09%` |
| random-target worker scale `0.10` | `69.53%` | `73.44%` | `75.00%` | `88.28%` | `80.47%` | `82.03%` | `69.53%` |

Conclusion:

```text
The learned Worker rerank path is not promotion-ready.
General-target Worker gives a 128-row false positive, then fails 256-row confirmation.
Mixed-target Worker improves 16p0 but sacrifices the 8x rows.
Do not keep sweeping worker-logit scale.
The next Worker route needs a dedicated policy-improvement gate or accepted-replacement dataset, not raw Worker logits as a centered fallback-policy bias.
```

## 2026-06-17 Strategy-Q Replacement Gate Probe

Added a conservative strategy-Q replacement gate to `evaluate_adaptive_policy.py`:

```text
--strategy-q-replace-threshold <q_advantage>
--strategy-q-replace-policy-margin <logit_margin>
```

Unlike `--strategy-q-rerank-scale`, this does not bias every legal action. The
policy first chooses its normal action. The evaluator then compares the learned
strategy-Q value of the best legal Q action with the chosen policy action:

```text
replace only if Q(best_legal_action) - Q(policy_action) >= threshold
optional: also require policy_logit(best_q_action) within policy_margin of top policy logit
```

This is a lightweight policy-improvement gate probe for
`runs/adaptive-plan-q-action-q-v1/generals-adaptive-plan-q-action-q-v1.eqx`.

Fixed-v5 max250, 128 games/row on seed `83720`:

| gate | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `7.81%` | `11.72%` | `7.81%` | `60.16%` | `42.97%` |
| threshold `0` | `9.38%` | `5.47%` | `5.47%` | `29.69%` | `31.25%` |
| threshold `2` | `6.25%` | `6.25%` | `6.25%` | `63.28%` | `55.47%` |
| threshold `4` | `9.38%` | `10.16%` | `9.38%` | `53.91%` | `52.34%` |

Only threshold `4` had a weak positive 128-row signal, so it received 256-row
confirmation.

Fixed-v5 max250, 256 games/row on seed `83740`:

| gate | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `9.77%` | `8.20%` | `8.20%` | `51.95%` | `51.56%` |
| threshold `4` | `13.28%` | `8.20%` | `8.20%` | `47.66%` | `55.08%` |
| threshold `4`, margin `2` | `10.55%` | `7.81%` | `7.81%` | `53.91%` | `50.78%` |
| threshold `4`, margin `4` | `10.55%` | `7.81%` | `7.81%` | `52.34%` | `51.17%` |

Conclusion:

```text
The thresholded gate is safer than raw centered Q bias, but the current action-Q head still does not improve the fixed-v5 gate.
Threshold 4 improves p0 in the 256-row confirmation, but p1 remains the bottleneck, so min win rate is unchanged.
Policy-margin support makes the gate more conservative but removes the p0 gain and slightly hurts p1.
Do not promote adaptive-plan-q-action-q-v1 or continue threshold sweeps.
Next step should train a true accepted-replacement/policy-improvement head from rows labeled by full replacement outcomes, rather than using the current action-Q scalar as the gate directly.
```

## 2026-06-17 Plan-Q Accepted-Replacement Gate Training

Added pairwise accepted-replacement supervision to `adaptive_plan_q_supervised.py`:

```text
--replacement-gate-weight
--replacement-score-margin
--replacement-target-margin
```

The loss uses rows where the teacher/base action appears in the Plan-Q candidate
set. It compares that teacher action against the best source-target plan action,
ordered first by replacement outcome and then by shaped plan score. A replacement
is labeled accepted when it changes the teacher action and either improves
loss/draw/win outcome or clears the same-outcome score margin. The action-Q head
then gets a pairwise margin objective:

```text
accepted: Q(best_plan_action) - Q(teacher_action) >= target_margin
rejected: Q(best_plan_action) - Q(teacher_action) <  target_margin
```

Training run:

```text
dataset: runs/adaptive-plan-q-fixed-v5-worker-v2/plan-q-00000.npz
init:    runs/adaptive-plan-q-action-q-v1/generals-adaptive-plan-q-action-q-v1.eqx
output:  runs/adaptive-plan-q-replacement-gate-v1/generals-adaptive-plan-q-replacement-gate-v1.eqx
loss:    replacement_gate_weight=1.0, score_margin=25, target_margin=1.0
scope:   frozen trunk/policy, strategy heads only
epochs:  120
```

Offline result:

| metric | start | end |
| --- | ---: | ---: |
| replacement loss | `1.6937` | `1.4027` |
| replacement accuracy | `41.6%` | `52.8%` |
| accepted fraction | `17.6%` | `17.6%` |
| comparable pair fraction | `47.9%` | `47.9%` |
| mean Q margin | `-1.042` | `-0.805` |

The loss is trainable, but the Q margin remains negative and separation is weak.
That immediately suggests risk: a full legal-action argmax over action-Q may
select actions that were never supervised by the candidate-pair loss.

Fixed-v5 max250, 128 games/row on seed `83840`:

| gate | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `7.81%` | `10.94%` | `7.81%` | `64.84%` | `61.72%` |
| threshold `0` | `1.56%` | `3.12%` | `1.56%` | `28.12%` | `27.34%` |
| threshold `0.5` | `2.34%` | `3.12%` | `2.34%` | `28.12%` | `35.16%` |
| threshold `1` | `2.34%` | `0.78%` | `0.78%` | `40.62%` | `44.53%` |

Low-threshold replacement lowers draw, but mostly by turning draws into losses.
The conservative policy-support gate was safer.

Fixed-v5 max250, 128 games/row on seed `83842`:

| gate | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `7.81%` | `12.50%` | `7.81%` | `53.91%` | `57.03%` |
| threshold `1`, margin `2` | `9.38%` | `12.50%` | `9.38%` | `56.25%` | `53.91%` |
| threshold `1`, margin `4` | `11.72%` | `9.38%` | `9.38%` | `51.56%` | `46.88%` |

The margin-4 row had the best 128-row tradeoff, so it received 256-row
confirmation.

Fixed-v5 max250, 256 games/row on seed `83860`:

| gate | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `12.50%` | `8.20%` | `8.20%` | `54.30%` | `60.94%` |
| threshold `1`, margin `4` | `10.16%` | `7.81%` | `7.81%` | `50.78%` | `43.36%` |

Conclusion:

```text
Pairwise accepted-replacement loss is implemented and trains, but v1 is not
promotion-ready.
The conservative gate can reduce fixed-v5 max250 draw, but 256-row min win rate
does not improve and p1 remains weak.
The likely failure is action-space coverage: the loss supervises only candidate
plan actions, while inference chooses the best Q among every legal primitive
action.
Do not continue threshold sweeps on this checkpoint.
Next useful step is either a larger accepted-replacement dataset with explicit
legal non-candidate negatives, or a separate replacement/gate head that scores
only generated source-target plan candidates before a target-conditioned Worker
executes them.
```

## 2026-06-17 Candidate-Only Strategy-Q Replacement Gate

Added `evaluate_adaptive_policy.py --strategy-q-replace-worker-candidate`.
When combined with `--strategy-q-replace-threshold`, the replacement gate no
longer searches over every legal primitive action. Instead it:

```text
1. samples/chooses the base policy action normally
2. builds one source-target worker candidate from the spatial heads
3. compares Q(worker_candidate) - Q(policy_action)
4. optionally requires policy logit support with --strategy-q-replace-policy-margin
5. replaces only if the candidate clears the gate
```

This tests whether the previous replacement failure was mostly caused by
unsupervised legal actions receiving arbitrary high Q values.

Fixed-v5 max250, 128 games/row on seed `83880`:

| gate | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `6.25%` | `12.50%` | `6.25%` | `57.81%` | `48.44%` |
| candidate threshold `0` | `5.47%` | `7.81%` | `5.47%` | `50.00%` | `42.19%` |
| candidate threshold `0`, margin `4` | `8.59%` | `7.81%` | `7.81%` | `53.12%` | `50.78%` |
| candidate threshold `1`, margin `4` | `7.81%` | `10.16%` | `7.81%` | `52.34%` | `53.12%` |

The conservative candidate gate again moved weakness between seats instead of
improving both. The best 128-row setting received 256-row confirmation.

Fixed-v5 max250, 256 games/row on seed `83900`:

| gate | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `10.55%` | `9.77%` | `9.77%` | `54.30%` | `54.69%` |
| candidate threshold `1`, margin `4` | `9.38%` | `11.72%` | `9.38%` | `60.55%` | `51.56%` |

Conclusion:

```text
Candidate-only replacement is safer than full legal-action Q argmax, but still
does not improve fixed-v5 max250.
The failure is no longer just arbitrary unsupervised legal actions; the
source/target worker candidate itself is not reliable enough and the Q gap is
not calibrated enough to pick useful interventions.
Do not continue candidate-only threshold sweeps.
Next useful implementation should train a target-conditioned Worker action head
from successful Plan-Q/worker trajectories, or collect a larger accepted-plan
dataset with explicit positive/negative candidate labels and a separate gate
head that scores candidates rather than primitive actions.
```

## 2026-06-17 Plan-Q Target-Conditioned Worker v0

Added `adaptive_plan_worker_supervised.py`, an offline trainer for a learned
target-conditioned Worker from Plan-Q shards. It consumes each Plan-Q row and
builds a Worker observation:

```text
worker_obs = adaptive_obs + source_one_hot + target_one_hot + route_potential
```

For v0, the trainer uses `selection=best`: one best non-pass source-target plan
per state, ordered by replacement outcome and shaped plan score. The Worker
network is a normal adaptive policy network over primitive actions. It can warm
start from the old 18-channel Worker and expand to the current 35-channel
U-Net/fog/history observation plus three command planes.

Added `evaluate_adaptive_policy.py` support:

```text
--strategy-plan-worker-path <worker.eqx>
--strategy-plan-worker-channels
--strategy-plan-worker-network-arch
--strategy-plan-worker-rerank-scale
```

At inference, the base strategy-spatial checkpoint chooses its normal policy
action. The evaluator also constructs a source-target command from the spatial
heads, runs the learned Worker, centers Worker logits over legal actions, and
adds them as a small policy-logit bias. This is still a rerank probe, not a hard
controller.

Training run:

```text
dataset: runs/adaptive-plan-q-fixed-v5-worker-v2/plan-q-00000.npz
examples: 403 best non-pass plans
init:    runs/adaptive-worker-split-random-v1/generals-adaptive-worker-split-random-v1.eqx
output:  runs/adaptive-plan-worker-best-v0/generals-adaptive-plan-worker-best-v0.eqx
input:   38 channels = 35 adaptive/fog/history + 3 command planes
loss:    action=0.2, source=1.0, direction=1.0
epochs:  120
```

Offline result:

| metric | start | end |
| --- | ---: | ---: |
| loss | `3.4277` | `2.5087` |
| action accuracy | `16.4%` | `41.2%` |
| source accuracy | `30.8%` | `54.6%` |
| direction accuracy | `45.2%` | `65.5%` |
| useful action | `16.4%` | `41.2%` |

Fixed-v5 max250, 128 games/row on seed `83940`:

| plan-worker scale | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `6.25%` | `8.59%` | `6.25%` | `53.91%` | `55.47%` |
| `0.02` | `8.59%` | `8.59%` | `8.59%` | `59.38%` | `56.25%` |
| `0.05` | `8.59%` | `11.72%` | `8.59%` | `58.59%` | `57.81%` |
| `0.10` | `7.81%` | `10.16%` | `7.81%` | `57.03%` | `58.59%` |

Fixed-v5 max250, 256 games/row on seed `83960`:

| plan-worker scale | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `9.77%` | `12.89%` | `9.77%` | `52.34%` | `50.39%` |
| `0.02` | `10.55%` | `11.33%` | `10.55%` | `51.95%` | `50.00%` |
| `0.05` | `11.33%` | `11.33%` | `11.33%` | `53.91%` | `50.39%` |

Scale `0.05` is best on fixed-v5, but it harms the adaptive Expander gate.

Expander adaptive 8/12/16, 128 games/row on seed `83980`:

| scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | `72.66%` | `71.09%` | `78.12%` | `84.38%` | `78.91%` | `80.47%` | `71.09%` |
| `0.02` | `76.56%` | `71.09%` | `78.12%` | `84.38%` | `73.44%` | `80.47%` | `71.09%` |
| `0.05` | `72.66%` | `68.75%` | `80.47%` | `80.47%` | `76.56%` | `82.81%` | `68.75%` |

Expander adaptive 8/12/16, 256 games/row on seed `84000`:

| scale | 8p0 | 8p1 | 12p0 | 12p1 | 16p0 | 16p1 | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | `77.73%` | `79.30%` | `82.03%` | `79.30%` | `76.56%` | `82.81%` | `76.56%` |
| `0.02` | `77.73%` | `80.08%` | `83.98%` | `81.25%` | `78.12%` | `76.17%` | `76.17%` |

Conclusion:

```text
Plan-Worker v0 is the first Worker-style probe that improves fixed-v5 max250
under 256-row confirmation without clearly collapsing the Expander adaptive
gate.
It is not a promotion setting: fixed-v5 min remains only 10-11%, and scale 0.05
hurts Expander 8p1 at 128-row.
Scale 0.02 is the safer diagnostic setting: fixed-v5 min improves by +0.78pp
and Expander 256-row min changes by -0.39pp, likely within evaluation noise.
This supports the next route: collect larger Plan-Q/Worker shards with more
decisive fixed-v5 states, train Plan-Worker on mixed best/all accepted plans,
and add a learned confidence gate before allowing stronger Worker influence.
Do not promote v0, but do continue Plan-Worker data scaling.
```

## 2026-06-17 Plan-Worker Data Scaling v1

Collected a larger fixed-v5 Plan-Q/Worker dataset with the same warmup and
worker rollout settings as v2, but four shards and the current
`adaptive-plan-q-replacement-gate-v1` checkpoint as the base policy:

```text
output: runs/adaptive-plan-q-fixed-v5-worker-v3/
grid: 8x8
truncation: 250
warmup_steps: 190
samples: 2048
plans/state: 4x4
plan_rollout_steps: 64
plan_worker_steps: 16
rollouts/plan: 1
device: cuda:0
```

Shard stats:

| shard | mean gap | best q | best win | best draw |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `0.4211` | `-0.0027` | `14.5%` | `78.7%` |
| 1 | `0.3789` | `-0.0076` | `11.5%` | `82.4%` |
| 2 | `0.3088` | `-0.0239` | `6.8%` | `91.4%` |
| 3 | `0.2915` | `-0.0189` | `3.1%` | `94.9%` |

Trained `adaptive-plan-worker-best-v1` from v0:

```text
dataset: runs/adaptive-plan-q-fixed-v5-worker-v3/plan-q-*.npz
examples: 1620 best non-pass plans
init:    runs/adaptive-plan-worker-best-v0/generals-adaptive-plan-worker-best-v0.eqx
output:  runs/adaptive-plan-worker-best-v1/generals-adaptive-plan-worker-best-v1.eqx
loss:    action=0.2, source=1.0, direction=1.0
epochs:  100
```

Offline result:

| metric | start | end |
| --- | ---: | ---: |
| loss | `3.1411` | `1.0056` |
| action accuracy | `29.4%` | `73.8%` |
| source accuracy | `44.5%` | `97.0%` |
| direction accuracy | `56.2%` | `76.0%` |

Fixed-v5 max250, 128 games/row on seed `84060`:

| plan-worker scale | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `11.72%` | `7.03%` | `7.03%` | `56.25%` | `58.59%` |
| `0.01` | `11.72%` | `7.81%` | `7.81%` | `51.56%` | `59.38%` |
| `0.02` | `12.50%` | `7.81%` | `7.81%` | `45.31%` | `55.47%` |
| `0.05` | `10.94%` | `7.81%` | `7.81%` | `57.03%` | `58.59%` |

Conclusion:

```text
Scaling best-plan data made the Worker much better at reproducing selected
commands, but did not expand fixed-v5 gameplay gains beyond v0.
The p1 bottleneck persists and stronger fit mostly shifts draw/loss balance.
Do not spend the next run on more best-only data.
Next data/training variant should mix best plans with all accepted/improved
candidate plans and train a separate confidence/gate target, so the Worker is
used only when the command is likely to be outcome-improving.
```

## 2026-06-17 Plan-Worker Accepted/Mixed Probe

Extended `adaptive_plan_worker_supervised.py` with two additional selection
modes:

```text
--selection accepted
  keep candidate plans whose primitive action differs from the teacher/base
  action and improves outcome, or preserves outcome while improving score by
  --accepted-score-margin.

--selection mixed
  concatenate each row's best plan with accepted plans, using
  --mixed-best-weight and --accepted-weight.
```

Also added `evaluate_adaptive_policy.py --strategy-plan-worker-min-margin`,
which only applies Worker rerank bias when the Worker's legal top-1/top-2 logit
margin clears the requested threshold.

Mixed Worker v0:

```text
dataset:   runs/adaptive-plan-q-fixed-v5-worker-v3/plan-q-*.npz
selection: mixed
examples:  2369
init:      runs/adaptive-plan-worker-best-v0/generals-adaptive-plan-worker-best-v0.eqx
output:    runs/adaptive-plan-worker-mixed-v0/generals-adaptive-plan-worker-mixed-v0.eqx
weights:   best=0.5, accepted=1.0, accepted_score_margin=25
```

Offline result:

| metric | start | end |
| --- | ---: | ---: |
| loss | `3.4783` | `0.7753` |
| action accuracy | `23.0%` | `77.2%` |
| source accuracy | `35.7%` | `100.0%` |
| direction accuracy | `52.2%` | `77.5%` |

Fixed-v5 max250, 128 games/row on seed `84100`:

| setting | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `8.59%` | `7.81%` | `7.81%` | `50.00%` | `59.38%` |
| scale `0.01` | `11.72%` | `10.16%` | `10.16%` | `46.88%` | `55.47%` |
| scale `0.02` | `14.84%` | `7.81%` | `7.81%` | `46.88%` | `53.91%` |
| scale `0.02`, margin `1` | `11.72%` | `8.59%` | `8.59%` | `48.44%` | `57.03%` |
| scale `0.02`, margin `2` | `12.50%` | `10.16%` | `10.16%` | `48.44%` | `57.03%` |
| scale `0.05`, margin `2` | `8.59%` | `9.38%` | `8.59%` | `53.12%` | `57.03%` |

The two best 128-row settings failed 256-row confirmation on seed `84120`:

| setting | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `10.55%` | `13.28%` | `10.55%` | `53.12%` | `52.73%` |
| scale `0.01` | `8.59%` | `12.89%` | `8.59%` | `56.25%` | `53.12%` |
| scale `0.02`, margin `2` | `8.98%` | `12.11%` | `8.98%` | `54.69%` | `53.52%` |

Accepted-only Worker v0:

```text
dataset:   runs/adaptive-plan-q-fixed-v5-worker-v3/plan-q-*.npz
selection: accepted
examples:  749
init:      runs/adaptive-plan-worker-best-v0/generals-adaptive-plan-worker-best-v0.eqx
output:    runs/adaptive-plan-worker-accepted-v0/generals-adaptive-plan-worker-accepted-v0.eqx
weights:   accepted=1.0, accepted_score_margin=25
```

Offline result:

| metric | start | end |
| --- | ---: | ---: |
| loss | `3.9273` | `2.4577` |
| action accuracy | `17.5%` | `44.4%` |
| source accuracy | `29.3%` | `62.6%` |
| direction accuracy | `43.4%` | `59.6%` |

Fixed-v5 max250, 128 games/row on seed `84160`:

| setting | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `12.50%` | `16.41%` | `12.50%` | `62.50%` | `55.47%` |
| scale `0.01` | `12.50%` | `15.62%` | `12.50%` | `63.28%` | `57.81%` |
| scale `0.02` | `12.50%` | `16.41%` | `12.50%` | `62.50%` | `58.59%` |
| scale `0.02`, margin `1` | `14.84%` | `16.41%` | `14.84%` | `60.94%` | `56.25%` |
| scale `0.02`, margin `2` | `13.28%` | `16.41%` | `13.28%` | `63.28%` | `56.25%` |

The best accepted-only 128-row setting also failed 256-row confirmation on
seed `84180`:

| setting | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `12.89%` | `8.98%` | `8.98%` | `53.91%` | `51.17%` |
| scale `0.02`, margin `1` | `12.11%` | `8.20%` | `8.20%` | `55.86%` | `52.34%` |

Conclusion:

```text
The Worker can fit best, mixed, and accepted Plan-Q commands offline, but
offline command accuracy still does not translate into stable fixed-v5 gains.
Mixed and accepted-only Workers both produced 128-row false positives and then
regressed under 256-row confirmation.

Stop spending runs on Plan-Worker rerank-scale and confidence-margin sweeps.
The next Plan-Q route should either learn an explicit command/gate scorer on
model-generated source-target candidates, or collect stronger decisive
counterfactuals before letting Worker logits affect primitive inference.
```

## 2026-06-17 Model-Generated Plan-Q Candidates

Added model-generated candidate modes to `adaptive_plan_q_dataset.py`:

```text
--candidate-source model
--candidate-target model
```

Default behavior remains `heuristic`, which picks sources/targets from
privileged state. Model mode instead runs the checkpoint's strategy spatial
source/target heads on the current fogged observation and masks them to:

```text
source: movable owned active cells
target: active passable cells
```

This tests the previous diagnosis that privileged Plan-Q candidates were
misaligned with inference-time source/target generation.

Smoke:

```text
output: runs/adaptive-plan-q-model-candidates-smoke-v0/
envs: 4
steps: 2
plans/state: 2x2
plan_rollout_steps: 4
plan_worker_steps: 2
device: cuda:0
result: samples=8, mean_gap=0.0286, best_q=-0.0604, best_win=0.0%, best_draw=100.0%
```

Fixed-v5 warm190 model-candidate shard:

```text
output: runs/adaptive-plan-q-model-candidates-v0/
model:  runs/adaptive-plan-q-replacement-gate-v1/generals-adaptive-plan-q-replacement-gate-v1.eqx
envs: 32
steps: 16
samples: 512
plans/state: 4x4
truncation: 250
warmup_steps: 190
plan_rollout_steps: 64
plan_worker_steps: 16
rollouts/plan: 1
candidate_source: model
candidate_target: model
device: cuda:0
```

Shard result:

```text
mean_gap=0.3931
best_q=0.1190
best_win=19.3%
best_draw=77.9%
```

This is stronger than the earlier heuristic-candidate v3 shard0
(`mean_gap=0.4211`, `best_q=-0.0027`, `best_win=14.5%`,
`best_draw=78.7%`). The important signal is not the exact small-shard win
rate; it is that model-generated candidates are not weaker than privileged
heuristic candidates and have positive average best Q on this seed.

Trained `adaptive-plan-q-model-candidate-gate-v0` on this shard:

```text
dataset: runs/adaptive-plan-q-model-candidates-v0/plan-q-00000.npz
init:    runs/adaptive-plan-q-replacement-gate-v1/generals-adaptive-plan-q-replacement-gate-v1.eqx
output:  runs/adaptive-plan-q-model-candidate-gate-v0/generals-adaptive-plan-q-model-candidate-gate-v0.eqx
losses:  action_q=0.25, action_q_mse=0.10, replacement_gate=1.0
scope:   strategy-heads only
epochs:  80
```

Offline:

| metric | start | end |
| --- | ---: | ---: |
| total loss | `4.2487` | `3.2029` |
| action-Q rank loss | `7.8267` | `5.4970` |
| action-Q MSE | `4.7735` | `2.4241` |
| action-Q candidate acc | `16.3%` | `15.9%` |
| replacement acc | `39.4%` | `44.3%` |
| accepted fraction | `16.2%` | `16.2%` |
| pair fraction | `44.5%` | `44.5%` |
| replacement q margin | `-1.416` | `-1.078` |

Fixed-v5 max250, 128 games/row on seed `84280`:

| setting | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `11.72%` | `10.94%` | `10.94%` | `57.81%` | `57.03%` |
| candidate gate threshold `1`, policy margin `4` | `13.28%` | `8.59%` | `8.59%` | `60.16%` | `57.81%` |
| candidate gate threshold `0` | `5.47%` | `8.59%` | `5.47%` | `49.22%` | `52.34%` |

Conclusion:

```text
Model-generated source/target candidates are a useful data improvement: they
match inference-time command generation and produced a stronger fixed-v5 shard
than the privileged heuristic candidates.

The current scalar action-Q replacement gate is still the wrong execution
mechanism. Training it on aligned model candidates did not fix the seat
tradeoff: threshold 1 improved p0 but hurt p1, and threshold 0 converted draws
into losses. Do not promote model-candidate-gate-v0.

Next step should keep model-generated candidates, but replace scalar action-Q
replacement with either:
1. a direct binary command-acceptance head over source/target/action features,
2. a per-seat calibrated gate threshold, or
3. a larger model-candidate dataset before fitting any gate.
```

## 2026-06-17 Binary Command Gate Probe

Added a standalone command acceptance gate:

```text
adaptive_command_gate.py
  CommandGateNetwork, a normalized 2-layer MLP.

adaptive_command_gate_supervised.py
  trains from Plan-Q shards using candidate command labels.

evaluate_adaptive_policy.py
  --strategy-command-gate-path
  --strategy-command-gate-threshold
  --strategy-command-gate-hidden-dim
```

The gate does not modify the adaptive policy checkpoint. At inference it uses
the evaluator's current source/target worker command and accepts it only when
the MLP probability clears the threshold. Features are all available at
inference:

```text
policy_logit_delta
action_q_delta
source_logit
target_logit
finish_probability
source_army_log1p
route_distance_norm
candidate_policy_logit
current_policy_logit
candidate_q
current_q
seat
```

### Model-candidate gate v0

Trained on `runs/adaptive-plan-q-model-candidates-v0/plan-q-00000.npz`:

```text
output:   runs/adaptive-command-gate-model-candidates-v0/generals-adaptive-command-gate-model-candidates-v0.eqx
examples: 2677
positive: 11.51%
epochs:   120
```

Offline result:

| metric | start | end |
| --- | ---: | ---: |
| loss | `0.6952` | `0.5143` |
| balanced accuracy | `53.4%` | `73.5%` |
| positive probability | `0.474` | `0.644` |
| negative probability | `0.472` | `0.355` |

Fixed-v5 max250, 128 games/row on seed `84320`:

| setting | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `10.94%` | `4.69%` | `4.69%` | `50.78%` | `56.25%` |
| gate `0.5` | `9.38%` | `2.34%` | `2.34%` | `57.03%` | `35.94%` |
| gate `0.6` | `6.25%` | `2.34%` | `2.34%` | `60.16%` | `50.00%` |
| gate `0.7` | `10.16%` | `3.91%` | `3.91%` | `57.81%` | `50.00%` |

Conclusion: direct binary gating is learnable offline, but this dataset's
top-k source candidates still differ from the evaluator's actual
route/army-adjusted source command, and gameplay regressed.

### Model-worker candidate v0

Added a more inference-aligned candidate mode:

```text
--candidate-source model-worker
--candidate-target model
```

`model-worker` uses the evaluator worker command's source score for the top
model target:

```text
source_logits + 0.25 * log1p(army) - 0.05 * route_distance_to_top_target
```

Collected `runs/adaptive-plan-q-model-worker-candidates-v0/`:

```text
samples=512
mean_gap=0.2947
best_q=-0.0432
best_win=9.8%
best_draw=88.9%
```

This is more inference-aligned but much weaker than the previous
model-candidate shard (`best_win=19.3%`, `best_q=0.1190`), which means the
currently executed source/target worker command is often a poor command.

Trained `adaptive-command-gate-model-worker-candidates-v0`:

```text
examples: 2663
positive: 7.29%
epochs:   100
```

Offline result:

| metric | start | end |
| --- | ---: | ---: |
| loss | `0.7249` | `0.4733` |
| balanced accuracy | `41.2%` | `78.8%` |
| positive probability | `0.475` | `0.671` |
| negative probability | `0.500` | `0.322` |

Fixed-v5 max250, 128 games/row on seed `84380`:

| setting | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `3.91%` | `10.94%` | `3.91%` | `64.84%` | `57.81%` |
| gate `0.5` | `2.34%` | `9.38%` | `2.34%` | `27.34%` | `44.53%` |
| gate `0.6` | `4.69%` | `7.81%` | `4.69%` | `35.16%` | `54.69%` |
| gate `0.7` | `7.03%` | `7.81%` | `7.03%` | `45.31%` | `50.78%` |

The apparent `0.7` signal failed 256-row confirmation on seed `84400`:

| setting | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `7.42%` | `6.64%` | `6.64%` | `55.08%` | `56.25%` |
| gate `0.7` | `5.47%` | `5.47%` | `5.47%` | `30.47%` | `54.69%` |

Conclusion:

```text
The binary command gate is a better supervision object than scalar action-Q
thresholds, but it still cannot rescue weak source/target proposals. When the
gate accepts more commands, draw often falls because losses rise.

Do not promote either command-gate checkpoint. The next useful direction is
not more gate threshold sweep; it is improving the Commander/source-target
proposal itself, likely with outcome-supervised target/source maps or a
low-rate Commander head that proposes several target/source commands before
the Worker/gate sees them.
```

### Multi-command gate probe

Extended `evaluate_adaptive_policy.py` so the command gate can score more than
one proposal:

```text
--strategy-command-gate-source-count
--strategy-command-gate-target-count
```

The evaluator takes top-k source cells from the source head and top-k target
cells from the target head, builds every source-target worker action, scores
each action with the command gate, and accepts the highest-probability command
when it clears the threshold. Defaults remain `1x1`, preserving the previous
single-command behavior.

Fixed-v5 max250, 128 games/row on seed `84420`, using
`adaptive-command-gate-model-candidates-v0`:

| setting | p0 win | p1 win | min | p0 draw | p1 draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | `7.03%` | `12.50%` | `7.03%` | `57.81%` | `57.03%` |
| gate `0.7`, candidates `2x2` | `5.47%` | `4.69%` | `4.69%` | `62.50%` | `40.62%` |
| gate `0.8`, candidates `2x2` | `6.25%` | `6.25%` | `6.25%` | `61.72%` | `50.78%` |
| gate `0.8`, candidates `4x4` | `6.25%` | `7.81%` | `6.25%` | `59.38%` | `42.19%` |

Conclusion:

```text
Top-k proposals did not fix the gate failure. More source-target candidates
reduce the single-worker source mismatch, but the current spatial heads still
rank too many harmful target/source pairs highly enough for the gate to accept
them. The next step should train the proposal maps themselves from outcome
targets, not add more post-hoc gate variants.
```

### Source/target outcome-Q map supervision

Extended `adaptive_plan_q_supervised.py` with direct source/target proposal-map
Q losses:

```text
--source-q-mse-weight
--target-q-mse-weight
--source-q-rank-weight
--target-q-rank-weight
--q-rank-temperature
--q-target-outcome-weight
```

The target is recomputed from the saved `plan_q` matrix rather than using only
the precomputed CE marginals:

```text
plan_value = (1 - outcome_weight) * plan_q
           + outcome_weight * outcome_value(loss=-1, draw=0, win=1)

source_q_target = max_target plan_value[source, target]
target_q_target = max_source plan_value[source, target]
```

`*_q_mse` regresses candidate source/target logits to these values. `*_q_rank`
uses a candidate-local softmax CE over the same values, so the loss directly
asks the proposal maps to rank the source/target candidates by outcome-Q.
Existing Plan-Q CE/action-Q behavior is unchanged when the new weights are 0.

GPU smoke 1, Q-MSE plus old CE, using
`runs/adaptive-plan-q-model-candidates-v0/plan-q-00000.npz` and warm-starting
from `adaptive-plan-q-replacement-gate-v1`:

```text
output: runs/adaptive-plan-q-source-target-q-v0/generals-adaptive-plan-q-source-target-q-v0.eqx
weights: source=0.25, target=0.25, source_q_mse=0.5, target_q_mse=0.5
outcome_weight: 0.35
epochs: 16
device: cuda:0
```

Result:

| metric | epoch 1 | epoch 16 |
| --- | ---: | ---: |
| total loss | `8.0815` | `2.6701` |
| source Q MSE / best / corr | `3.3262 / 31.6% / -0.048` | `1.0046 / 30.6% / -0.035` |
| target Q MSE / best / corr | `9.2763 / 25.8% / -0.073` | `0.6687 / 26.7% / -0.107` |

Conclusion: Q-MSE is easy to reduce, but it mostly fits scale/mean and does
not learn useful candidate ordering.

GPU smoke 2, rank+MSE:

```text
output: runs/adaptive-plan-q-source-target-rank-v0/generals-adaptive-plan-q-source-target-rank-v0.eqx
weights: source_q_mse=0.1, target_q_mse=0.1, source_q_rank=1.0, target_q_rank=1.0
outcome_weight: 0.35
rank_temperature: 0.20
epochs: 32
device: cuda:0
```

Result:

| metric | epoch 1 | epoch 32 |
| --- | ---: | ---: |
| source rank loss / best / corr | `1.7796 / 31.8% / -0.047` | `1.4211 / 24.8% / +0.002` |
| target rank loss / best / corr | `1.4453 / 25.2% / -0.079` | `1.4136 / 26.9% / -0.020` |

Conclusion: mild rank supervision still fails to separate the useful candidate
from the source/target set.

GPU smoke 3, sharp rank-only:

```text
output: runs/adaptive-plan-q-source-target-rank-sharp-v0/generals-adaptive-plan-q-source-target-rank-sharp-v0.eqx
weights: source_q_rank=1.0, target_q_rank=1.0
outcome_weight: 0.65
rank_temperature: 0.05
epochs: 64
device: cuda:0
```

Result:

| metric | epoch 1 | epoch 64 |
| --- | ---: | ---: |
| source rank loss / best / corr | `1.7562 / 32.3% / -0.047` | `1.3834 / 23.7% / +0.072` |
| target rank loss / best / corr | `1.4404 / 25.7% / -0.093` | `1.2835 / 42.6% / +0.276` |

Conclusion:

```text
Outcome-Q can train the target proposal map on the model-candidate shard.
It does not train a reliable source proposal map from the same source label.
The next source route should not reuse raw max-over-target plan values; it
should use executor-aware source labels such as model-worker/source-army/route
features, accepted-plan source positives, or a separate Worker-conditioned
source selector. Fixed-v5 gameplay smoke was skipped because the frozen v5
policy file was not present at /tmp/generals-ppo-8x8-expander-gpu-v5.eqx.
```

### Plan-pair additive ranking

Added `adaptive_plan_q_supervised.py --plan-pair-rank-weight`, which ranks the
full source-target candidate matrix with additive proposal-map scores:

```text
pair_logit[source, target] = source_logit[source] + target_logit[target]
pair_target = softmax(plan_value[source, target] / temperature)
```

This keeps old behavior unchanged when the weight is 0, but tests whether the
decomposed source/target maps can recover the best plan without a separate
Commander pair scorer.

GPU smoke 4, sharp rank-only on evaluator-aligned model-worker source
candidates:

```text
dataset: runs/adaptive-plan-q-model-worker-candidates-v0/plan-q-00000.npz
output: runs/adaptive-plan-q-source-target-rank-worker-source-v0/generals-adaptive-plan-q-source-target-rank-worker-source-v0.eqx
weights: source_q_rank=1.0, target_q_rank=1.0
outcome_weight: 0.65
rank_temperature: 0.05
epochs: 64
device: cuda:0
```

Result:

| metric | epoch 1 | epoch 64 |
| --- | ---: | ---: |
| total loss | `2.9824` | `2.6890` |
| source rank loss / best / corr | `1.5672 / 34.4% / +0.080` | `1.3679 / 33.4% / +0.127` |
| target rank loss / best / corr | `1.4152 / 29.7% / -0.028` | `1.3212 / 42.3% / +0.303` |

GPU smoke 5, pair-rank only on the same model-worker shard:

```text
output: runs/adaptive-plan-q-pair-rank-worker-source-v0/generals-adaptive-plan-q-pair-rank-worker-source-v0.eqx
weights: plan_pair_rank=1.0
outcome_weight: 0.65
rank_temperature: 0.05
epochs: 64
device: cuda:0
```

Result:

| metric | epoch 1 | epoch 64 |
| --- | ---: | ---: |
| total loss | `2.9690` | `2.6773` |
| pair rank loss / pair best / source best / target best / corr | `2.9690 / 8.7% / 35.8% / 28.0% / +0.013` | `2.6773 / 15.3% / 33.8% / 43.9% / +0.179` |

GPU smoke 6, independent rank plus pair-rank:

```text
output: runs/adaptive-plan-q-pair-rank-combo-worker-source-v0/generals-adaptive-plan-q-pair-rank-combo-worker-source-v0.eqx
weights: source_q_rank=0.5, target_q_rank=1.0, plan_pair_rank=1.0
outcome_weight: 0.65
rank_temperature: 0.05
epochs: 64
device: cuda:0
```

Result:

| metric | epoch 1 | epoch 64 |
| --- | ---: | ---: |
| total loss | `5.1621` | `4.6850` |
| source rank loss / best / corr | `1.5548 / 36.2% / +0.089` | `1.3673 / 33.6% / +0.123` |
| target rank loss / best / corr | `1.4148 / 29.5% / -0.068` | `1.3214 / 41.8% / +0.307` |
| pair rank loss / pair best / source best / target best / corr | `2.9699 / 7.2% / 36.2% / 26.5% / +0.009` | `2.6799 / 15.9% / 33.6% / 43.7% / +0.171` |

Conclusion:

```text
Model-worker source candidates improve source ranking from the raw
model-candidate shard, but source remains much weaker than target.

Additive source+target pair ranking is above the 1/16 random baseline
(15-16% pair top1), so the Plan-Q signal is present. It is still too weak to
be a promotion path, and combining independent rank losses does not materially
improve it.

Next route should not continue spatial/rerank-scale sweeps. Use an explicit
plan-pair/Commander scorer, or make source selection target-conditioned with
Worker-conditioned route, army, safety, and accepted-plan features.
```

### Explicit plan-pair scorer

Added two files:

```text
examples/_experimental/ppo/adaptive_plan_pair_scorer.py
examples/_experimental/ppo/adaptive_plan_pair_supervised.py
```

The scorer is a normalized MLP over one source-target plan candidate. It uses
only inference-time features:

```text
policy/action-Q delta
source/target logits
finish probability
source/target army
route distance
candidate/current policy logits and Q
source/target coordinates and deltas
grid size, turn, seat
source cell observation channels
target cell observation channels
```

The supervised trainer builds row-wise `16` pair examples from Plan-Q shards
and trains a rank CE target from:

```text
plan_value = (1 - outcome_weight) * plan_q
           + outcome_weight * outcome_value(loss=-1, draw=0, win=1)
```

It now saves both the final checkpoint and a `.best.eqx` checkpoint selected by
validation pair top-1 accuracy.

Same-split additive baseline on
`runs/adaptive-plan-q-model-worker-candidates-v0/plan-q-00000.npz`, with the
v1 train/validation split:

```text
train: loss 3.0071, pair 8.66%, source 38.18%, target 23.46%, corr -0.002
val:   loss 2.9355, pair 4.19%, source 28.70%, target 26.18%, corr +0.026
```

GPU run v1, worker-source shard, gap-weighted:

```text
output: runs/adaptive-plan-pair-scorer-v1/
dataset: runs/adaptive-plan-q-model-worker-candidates-v0/plan-q-00000.npz
rows: 512
train/val: 410 / 102
hidden_dim: 128
epochs: 128
device: cuda:0
```

Best validation:

```text
epoch: 62
loss: 2.7398
pair top1: 23.3%
source top1: 32.9%
target top1: 43.7%
corr: +0.154
best checkpoint: runs/adaptive-plan-pair-scorer-v1/generals-adaptive-plan-pair-scorer-v1.best.eqx
```

GPU run v2, mixed model + model-worker shards:

```text
output: runs/adaptive-plan-pair-scorer-v2/
datasets:
  runs/adaptive-plan-q-model-worker-candidates-v0/plan-q-00000.npz
  runs/adaptive-plan-q-model-candidates-v0/plan-q-00000.npz
rows: 1024
train/val: 819 / 205
hidden_dim: 128
epochs: 128
device: cuda:0
```

Best validation:

```text
epoch: 20
loss: 2.7819
pair top1: 12.9%
source top1: 23.6%
target top1: 33.4%
corr: +0.092
```

GPU run worker-small-v0, worker-source shard, no gap weighting:

```text
output: runs/adaptive-plan-pair-scorer-worker-small-v0/
dataset: runs/adaptive-plan-q-model-worker-candidates-v0/plan-q-00000.npz
rows: 512
train/val: 410 / 102
hidden_dim: 64
epochs: 64
gap_weighting: false
device: cuda:0
```

Best validation:

```text
epoch: 11
loss: 2.7561
pair top1: 13.7%
source top1: 25.5%
target top1: 36.3%
corr: +0.087
```

Conclusion:

```text
The explicit pair scorer can learn non-additive plan interactions on the
worker-source shard: v1 validation pair top1 is 23.3%, far above the same-split
additive baseline of 4.19%.

The signal is still data-sensitive. Mixing model and model-worker candidate
families degrades pair ranking, and removing gap weighting removes most of the
benefit. This is not ready for fixed-v5 gameplay promotion.

Next step should be a larger same-family model-worker Plan-Q shard or a
target-conditioned source selector. Do not connect this scorer to inference
until validation remains stable across larger shards/seeds.
```

### Larger model-worker Plan-Q and gap-filtered pair scoring

Collected four same-family model-worker Plan-Q shards with the replacement-gate
checkpoint as the candidate generator and base policy:

```text
output: runs/adaptive-plan-q-model-worker-candidates-v1/
states: 2048
plans/state: 16
warmup_steps: 190
plan_worker_steps: 16
plan_rollout_steps: 64
opponent: fixed 8x8 v5 sample
device: cuda:0
```

Shard summary:

```text
shard 0000: mean_gap 0.3906, best_q +0.0299, best_win 17.0%, best_draw 76.4%
shard 0001: mean_gap 0.2903, best_q -0.0481, best_win  9.6%, best_draw 85.4%
shard 0002: mean_gap 0.2607, best_q -0.0477, best_win  7.0%, best_draw 92.0%
shard 0003: mean_gap 0.2312, best_q -0.0349, best_win  6.8%, best_draw 92.0%
```

Aggregate:

```text
all rows:      rows 2048, best_win 10.1%, best_draw 86.4%, mean_gap 0.293, mean_q -0.025
gap >= 0.25:  rows  779, best_win 26.4%, best_draw 73.6%, mean_gap 0.612, mean_q +0.205
gap >= 0.50:  rows  374, best_win 46.3%, best_draw 53.7%, mean_gap 0.877, mean_q +0.456
```

Same-split additive baseline on the full v1 dataset:

```text
train: loss 3.0149, pair 7.91%, source 31.48%, target 28.63%, corr +0.003
val:   loss 2.9254, pair 9.34%, source 29.44%, target 28.17%, corr +0.034
```

Added `adaptive_plan_pair_supervised.py --min-plan-gap` to filter low-margin
rows before train/validation split. Full and filtered scorer runs:

```text
worker-v3 full:
  rows 2048
  best val epoch 70
  loss 2.7538
  pair top1 11.3%
  source top1 25.6%
  target top1 40.7%
  corr +0.071

worker-gap025-v0:
  min_plan_gap 0.25
  rows 779
  best val epoch 32
  loss 2.7241
  pair top1 16.8%
  source top1 31.7%
  target top1 45.3%
  corr +0.104

worker-gap05-v0:
  min_plan_gap 0.50
  rows 374
  best val epoch 24
  loss 2.7568
  pair top1 22.2%
  source top1 37.1%
  target top1 41.3%
  corr +0.085
```

Conclusion:

```text
The larger model-worker dataset did not confirm the first 512-row pair-scorer
result across all rows. Full worker-v3 only beats the additive baseline weakly
on validation pair top1: 11.3% vs 9.34%.

The useful signal is concentrated in high-gap states. Filtering at 0.25 raises
pair top1 to 16.8%, and filtering at 0.50 reaches 22.2% on a smaller validation
set. These rows also have much higher best-plan win rates.

Next Commander-scorer data should either collect more high-gap model-worker
rows, train a separate decisive-plan scorer, or downsample low-gap draw-heavy
states. Do not train the evaluator-facing scorer directly on unfiltered
draw-heavy mixed rows.
```

### High-gap Plan-Q save filter

Added save-time filtering to `adaptive_plan_q_dataset.py`:

```text
--min-plan-gap <gap>
--require-best-plan-win
```

The filter runs after each scored rollout is flattened and before the shard is
written. It does not change plan scoring, behavior rollout, candidate
generation, or model checkpoints. It only controls which rows are saved to the
ignored `runs/` dataset shard.

Metadata now records:

```text
min_plan_gap
require_best_plan_win
num_samples_before_filter
num_samples_dropped
```

If a shard has no kept rows, the script prints a skipped-shard line and
continues. This keeps high-threshold collection runs from writing empty NPZ
files that later break validation splits.

Recommended next collection probe:

```text
output: runs/adaptive-plan-q-model-worker-highgap-v0/
candidate_source: model-worker
candidate_target: model
min_plan_gap: 0.25
plan_worker_steps: 16
plan_rollout_steps: 64
warmup_steps: 190
```

Use `--require-best-plan-win` only for a smaller decisive-plan dataset or a
positive-class Commander scorer; it will discard many rows.

### High-gap model-worker Plan-Q v0

Collected a filtered high-gap model-worker shard set:

```text
output: runs/adaptive-plan-q-model-worker-highgap-v0/
seed: 84800
filter: min_plan_gap 0.25
states scored: 2048
states kept: 818
plans/state: 16
warmup_steps: 190
plan_worker_steps: 16
plan_rollout_steps: 64
device: cuda:0
```

Shard results:

```text
0000: kept 289/512, mean_gap 0.7051, best_q +0.3176, best_win 29.1%, best_draw 70.9%
0001: kept 215/512, mean_gap 0.6548, best_q +0.3352, best_win 33.0%, best_draw 67.0%
0002: kept 154/512, mean_gap 0.6641, best_q +0.3843, best_win 34.4%, best_draw 65.6%
0003: kept 160/512, mean_gap 0.6821, best_q +0.3799, best_win 35.0%, best_draw 65.0%
```

Aggregate:

```text
rows: 818
best-plan wins/draws/losses: 264 / 554 / 0
best_win: 32.3%
best_draw: 67.7%
mean_best_q: +0.3470
mean_gap: 0.6797
gap quantiles: [0.2507, 0.3271, 0.5027, 0.9924, 1.3004, 1.5794, 1.8750]
```

Trained explicit pair scorer:

```text
output: runs/adaptive-plan-pair-scorer-highgap-v0/
rows: 818
train/val: 654 / 164
hidden_dim: 128
gap_weighting: true
min_plan_gap in trainer: 0.0
```

Same split additive baseline:

```text
train: pair 10.7%, source 35.6%, target 24.5%, corr -0.019
val:   pair 15.2%, source 37.3%, target 28.0%, corr -0.016
```

Best validation:

```text
epoch: 92
loss: 2.7456
pair top1: 18.0%
source top1: 30.7%
target top1: 38.3%
corr: +0.127
```

Also trained a harder `min_plan_gap=0.5` scorer on the same saved rows:

```text
output: runs/adaptive-plan-pair-scorer-highgap-gap05-v0/
rows: 410
train/val: 328 / 82
best validation epoch: 62
pair top1: 15.3%
source top1: 30.7%
target top1: 36.2%
corr: +0.110
same-split additive val pair top1: 15.2%
```

Conclusion:

```text
Save-time high-gap filtering works: it raises best-plan win density to 32.3%
and gives the explicit pair scorer a real but modest edge over additive
source+target on the same validation split: 18.0% vs 15.2%.

The 0.5 filter is too small at this scale and collapses to additive-baseline
pair top1. The next useful route is more min_gap=0.25 data, not harder
filtering or evaluator integration.
```

### High-gap v1 scaling and turn-filter diagnosis

Scaled `min_plan_gap=0.25` collection to 16 shards:

```text
output: runs/adaptive-plan-q-model-worker-highgap-v1/
seed: 84900
states scored: 8192
states kept: 3415
best_win: 23.4%
best_draw: 76.6%
mean_best_q: +0.2094
mean_gap: 0.6243
gap quantiles: [0.2500, 0.3274, 0.4863, 0.8206, 1.2520, 1.3948, 1.8340]
```

Distribution comparison:

```text
highgap-v0:
  rows: 818
  time mean/median: 127.3 / 128
  p0 rows/win/gap/q/time: 344 / 26.5% / 0.633 / +0.279 / 133.3
  p1 rows/win/gap/q/time: 474 / 36.5% / 0.714 / +0.396 / 123.0

highgap-v1:
  rows: 3415
  time mean/median: 103.2 / 96
  p0 rows/win/gap/q/time: 1598 / 25.0% / 0.635 / +0.178 / 94.8
  p1 rows/win/gap/q/time: 1817 / 22.0% / 0.615 / +0.237 / 110.5
```

Pair-scorer results:

```text
highgap-v1:
  rows: 3415
  train/val: 2732 / 683
  same-split additive val pair top1: 5.0%, corr -0.014
  best scorer val pair top1: 8.2%, source 22.4%, target 29.5%, corr +0.058

highgap-v1-gap05:
  rows: 1656
  train/val: 1325 / 331
  same-split additive val pair top1: 7.1%, corr -0.033
  best scorer val pair top1: 11.3%, source 27.7%, target 30.6%, corr +0.084
```

Conclusion:

```text
More rows alone did not fix pair scoring. The v1 collection drifted earlier in
the game and had much weaker p1 winning-plan density than v0. The pair scorer
still beats additive, but the absolute signal is too weak for evaluator
integration.

Added `adaptive_plan_q_dataset.py --min-save-turn` and `--max-save-turn` so the
next dataset can target mid/late high-gap states directly. Next probe should use
min_gap=0.25 plus min_save_turn around 100 or 120; do not keep scaling raw
0.25 high-gap rows.
```

Added top-k reporting to `adaptive_plan_pair_supervised.py`:

```text
pair@1
pair@2
pair@4
```

The best checkpoint is still selected by validation pair@1. The new metrics are
diagnostic only, but they are important because the mid100 run improved
correlation more than argmax top1; a Commander can still use a top-k scorer if
later rollout or a Worker head resolves the final choice.

Recomputed top-k metrics for the best highgap-v0 scorer:

```text
checkpoint: runs/adaptive-plan-pair-scorer-highgap-v0/generals-adaptive-plan-pair-scorer-highgap-v0.best.eqx
train: pair@1 13.7%, pair@2 25.7%, pair@4 41.7%, source 34.0%, target 36.0%, corr +0.176
val:   pair@1 18.0%, pair@2 23.8%, pair@4 33.5%, source 30.7%, target 38.3%, corr +0.127
```

This is not strong enough for direct evaluator integration, but it is better
than a pure argmax read suggests. A future Commander probe should treat the
scorer as a top-k shortlist, not as a single-plan selector.

### Search teacher strategy shards

`adaptive_strategy_dataset.py` now supports `--teacher-kind search` for adaptive
checkpoints. The collector keeps the adaptive policy logits as the KL teacher,
but executes the best top-k action under short rollout search as the saved
teacher action. This is intended for midgame decisive trajectory imitation:
instead of training a separate Plan-Q scorer/gate stack, generate contact-heavy
winning windows from a stronger search-controlled policy and train the U-Net
main policy with KL plus finish/outcome/belief/intent losses.

The search path is compatible with current 35-channel U-Net checkpoints
(`scoreboard-history + fog-memory`). A GPU smoke shard succeeded with:

```text
teacher: adaptive-unet-ppo-v4
input: 35 channels, scoreboard history, fog memory
search: top_k=2, rollout_steps=2, rollouts_per_action=1
output: runs/adaptive-strategy-search-smoke/search-smoke-00000.npz
samples: 4/4
device: cuda:0
```

Next data run should collect terminal-window decisive rows, not full raw
episodes:

```text
filters:
  min_save_turn >= 80
  require_contact
  min_visible_enemy_cells >= 1
  require_win_or_finish_within_250
  terminal_window <= 120
```

Start with U-Net v4 + rollout-search against Expander and fixed v5 on 8x8; keep
draw-heavy contact shards separate for anti-draw supervision.

### Midgame search imitation v0-v3

Collected rollout-search decisive shards under
`runs/adaptive-strategy-search-decisive-v0/`:

```text
unet-v4-search-expander-win:     5,414 rows
unet-v4-search-v5-win:           3,930 rows
unet-v4-search-v5-draw:          2,904 rows
unet-v4-search-expander-win-b:  11,139 rows
unet-v4-search-v5-win-b:         6,280 rows
```

All winning shards use `turn>=80`, contact, visible enemy, win-or-finish250, and
terminal-window<=120 filters. The draw shard uses the same contact/time/window
filters with `draw_only`.

Training/eval summary:

```text
v0:
  data: first search wins + v5 draw
  weights: KL=1.0, action CE=0.3, lr=1e-5
  result: overfit/regressed
  Expander 64-row min: 64.06%
  fixed-v5 max250 64-row min: 9.38%

v1:
  data: v4-expander-balanced + first search wins, no draw shard
  weights: KL=4.0, action CE=0.15, lr=5e-6
  result: best candidate, not promotion
  Expander 256-row min: 75.78%
  same-seed v4 base 256-row min: 74.22%
  Expander 512-row min: 74.61%
  same-seed v4 base 512-row min: 70.51%
  fixed-v5 max250 64-row min: 10.94%

v2:
  data: v1 data + v5 draw shard
  action CE mode: non-draw
  result: 64-row positive, 256-row regression
  Expander 64-row min: 79.69%
  Expander 256-row min: 70.31%
  same-seed v4 base 256-row min: 72.27%
  fixed-v5 max250 64-row min: 7.81%

v3:
  data: v4-expander-balanced + all search win shards
  weights: KL=5.0, action CE=0.12, lr=4e-6
  result: more search action rows regressed weak rows
  Expander 256-row min: 70.31%
```

Current conclusion: rollout-search winning trajectories are useful, but direct
action imitation has a narrow stability window. The best run, v1, improves the
same-seed 512-row Expander min over v4 base by about 4pp, but still misses the
75% promotion line and does not improve fixed-v5 max250. Do not continue by
simply adding more search-action rows or increasing action CE. Next attempts
should either keep v1 as a conservative supervised base and collect more diverse
winning windows, or move search supervision into value/finish/intent targets
with action CE staying weak.

### PPO from midgame search v1

Added teacher-template flags to `train_adaptive.py` so `--teacher-kl-weight` can
load complex adaptive checkpoints with per-size HL-Gauss value heads, outcome
head, and strategy auxiliary heads. A direct load smoke confirmed that
`adaptive-midgame-search-imitation-v1` can be loaded as a 35-channel U-Net
teacher with `(8,12,16)` value heads, 128 categorical value bins, outcome head,
strategy aux, and 3 finish logits.

PPO probes from v1:

```text
oom probe:
  96 env x 256 steps, minibatch 4096
  result: GPU OOM during train minibatch compilation

small probe:
  48 env x 128 steps, minibatch 2048
  result: too much compile/memory pressure, interrupted

tiny probe:
  24 env x 64 steps, minibatch 512
  result: runnable but too few episodes
  Expander 64-row min: 67.19%

mb256 probe:
  48 env x 128 steps, minibatch 256
  lr=3e-7, terminal reward, mixed seats, top-advantage stratified, EMA
  rollout: iter20 episodes=22, wins=16
  Expander 64-row min: 76.56%
  Expander 256-row min: 69.53%

kl probe:
  same as mb256, minibatch 128, teacher KL=0.1 to v1
  complex teacher loaded with per-size HL-Gauss/outcome/strategy aux flags
  rollout: iter20 episodes=22, wins=16
  Expander 64-row min: 67.19%
```

Conclusion: direct PPO from v1 is not currently a promotion route. The runnable
PPO configurations have too few decisive episodes and shift weak rows badly; the
KL-anchored version still regressed at 64 rows. Keep the new teacher-loader
support, but do not continue this path without either larger-memory microbatch
work, richer rollout signal, or a better value/finish/search target.

### High-gap search trajectory imitation

Added search-score diagnostics to `adaptive_strategy_dataset.py --teacher-kind
search`:

```text
search_candidate_indices
search_prior_scores
search_scores
search_outcomes
search_best_position
search_best_score
search_mean_score
search_score_gap
search_best_outcome
```

`search_score_gap` is now best score minus the second valid prior candidate.
This matters because early pass-only states can include invalid top-k candidates
with very negative prior logits; using the raw mean made those states look like
fake high-gap examples. A CUDA smoke confirmed pass-only opening rows now have
gap `0.0`.

The collector also accepts:

```text
--min-search-score-gap
--require-search-best-win
--teacher-strategy-aux
--teacher-strategy-spatial-aux
--teacher-strategy-finish-outputs
```

Collected high-gap search decisive data from `adaptive-midgame-search-imitation-v1`
against Expander:

```text
output: runs/adaptive-strategy-search-highgap-v1/
teacher: adaptive-midgame-search-imitation-v1
filters:
  turn >= 80
  contact
  visible_enemy_cells >= 1
  win_or_finish250
  terminal_window <= 120
  search_score_gap >= 0.25
budget:
  32 envs
  4 shards x 256 steps
  top_k=4, rollout_steps=8, rollouts/action=1
rows:
  shard0 289 / 8192
  shard1 682 / 8192
  shard2 587 / 8192
  shard3 736 / 8192
  total 2294
```

The first single-shard diagnostic had healthy labels:

```text
all kept rows: outcome win
finish_within_250: 100%
finish_within_100: 98.0%
search_best_outcome: 76 win / 130 draw
gap median: 1.37
gap p75: ~1012
steps_to_terminal median: 23
```

This confirms the data filter is selecting real decisive midgame/terminal
windows. However, direct main-policy trajectory imitation still produced a
seat-specific regression.

Training probes from `adaptive-midgame-search-imitation-v1`:

```text
v0:
  data: highgap-v1 only
  weights: KL=1.0, action CE=0.30, finish=0.50, outcome=0.40, belief=0.25, intent=0.20
  lr=3e-6
  Expander 64-row seed85280 min: 62.50%
  weak row: 8p1 = 62.50%
  same-seed v1 baseline min: 73.44%

v1:
  data: highgap-v1 only
  weights: KL=4.0, action CE=0.08, same aux weights
  lr=2e-6
  Expander 64-row seed85280 min: 65.62%
  weak row: 8p1 = 65.62%

v2-balanced:
  data: highgap-v1 only
  balance: size-seat, 1230 rows
  weights: KL=4.0, action CE=0.08
  Expander 64-row seed85280 min: 64.06%
  weak row: 8p1 = 64.06%

v3-mixed:
  data: v4-expander-balanced + highgap-v1
  cap: 512 rows/shard
  balance: size-seat, 1260 rows
  weights: KL=4.0, action CE=0.08
  lr=1e-6
  Expander 64-row seed85280 min: 62.50%
  weak row: 8p1 = 62.50%
```

Conclusion:

```text
High-gap search trajectory filtering is useful infrastructure.
The current direct action-CE route is not promotion-worthy.
Even balanced and mixed broad-anchor variants damage 8x8 p1.
```

The next use of these shards should move their signal away from primitive action
CE and into value/finish/search-target supervision:

```text
1. train finish/outcome/belief heads or value targets from high-gap rows;
2. use high-gap rows as weighted auxiliary replay, not as dominant action CE;
3. if policy is updated, require per-row or per-stratum KL/behavior preservation
   and confirm 8p1 before any 256-row promotion run.
```

### Search-Q rank head from high-gap shards

`adaptive_strategy_supervised.py` now consumes the rollout-search top-k fields
saved by `adaptive_strategy_dataset.py --teacher-kind search`:

```text
search_candidate_indices
search_prior_scores
search_scores
search_score_gap
```

The new `--search-q-rank-weight` loss trains the strategy action-Q head on a
soft top-k ranking target derived from `search_scores / --search-q-temperature`.
Rows only contribute when they have at least two valid prior candidates and a
positive `search_score_gap`. This keeps high-gap search supervision away from
the main policy logits and from primitive action CE.

GPU probe:

```text
run: runs/adaptive-midgame-search-q-rank-v0/
data: adaptive-strategy-search-highgap-v1, balanced by size-seat
init: adaptive-midgame-search-imitation-v1
update_scope: strategy-heads
losses: search_q_rank=1.0 only
epochs: 20
result:
  SQ loss: 3.8067 -> 2.7584
  SQ top1: ~27.5% -> ~24.2%
```

Gameplay probes on seed `85280`:

```text
scale=0, 64-row:
  Expander min: 73.44%

q-rerank scale=0.01, 64-row:
  Expander min: 54.69%
  weak row: 8p1

q-rerank scale=0.001, 64-row:
  Expander min: 75.00%

scale=0, 256-row:
  Expander min: 71.09%

q-rerank scale=0.001, 256-row:
  Expander min: 71.48%
```

Conclusion: the rank loss is trainable, but raw action-Q rerank is still not a
promotion path. The 256-row improvement is noise-sized, and a modestly larger
scale collapses 8p1. Future use should normalize/gate Q bias or use these
targets for finish/value/intent supervision rather than direct logits bias.

Low-learning-rate full-policy recheck:

```text
run: runs/adaptive-midgame-decisive-imitation-v0/
data: same highgap-v1, balanced by size-seat
weights: KL=1.0, action CE=0.30, finish=0.50, outcome=0.40,
         belief=0.25, intent=0.20
lr: 1e-6
epochs: 12
Expander 64-row seed85280 min: 56.25%
weak row: 8p1
```

This reinforces the earlier high-gap imitation result: even with finish,
outcome, belief, and intent auxiliary targets, directly updating the whole U-Net
policy on narrow high-gap winning rows creates a seat-specific regression.
Freeze policy first, mix broader non-winning/draw-heavy counterexamples, or add
a much stricter per-stratum behavior-preservation gate before trying another
full-policy trajectory imitation run.

### Search-best outcome/finish head

High-gap search shards have a useful local label that trajectory outcome hides.
In `adaptive-strategy-search-highgap-v1`, the saved trajectory outcome is all
win after filtering, but `search_best_outcome` still separates local search-win
from search-draw states:

```text
search_best_outcome win rate by stratum:
  8p0: 34.6%
  8p1: 19.0%
  12p0: 21.9%
  12p1: 32.4%
  16p0: 26.1%
  16p1: 23.3%
balanced rows: 1230
```

`adaptive_strategy_supervised.py --label-source search-best` now switches only
finish/outcome labels to this local search-best outcome. Action CE weighting
continues to use the real trajectory outcome. The same change adds
`--balance-finish-labels` and `--balance-outcome-labels` to avoid the rare
search-win labels being swamped by search-draw rows.

The first probes ran in the default sandbox, which did not expose `/dev/nvidia*`
and made JAX report `CUDA_ERROR_NO_DEVICE`. An escalated host check then
confirmed the machine has an RTX 5070 Ti and JAX can use `cuda:0`; the balanced
run was repeated on GPU.

```text
run: runs/adaptive-search-best-outcome-head-v0/
labels: search-best
finish head: binary, warm-started from multi-horizon v1
losses: finish=0.5, outcome=0.5
balance: none
result:
  finish loss: 1.3691 -> 0.6268
  finish accuracy: 26.4% -> 72.0%
  outcome loss: 2.7738 -> 1.3488
  outcome accuracy: 28.6% -> 40.8%
interpretation:
  finish accuracy mostly matches the draw-majority baseline, so the unbalanced
  run is not evidence of decisive-state recognition.

run: runs/adaptive-search-best-outcome-head-v1/
labels: search-best
losses: finish=0.5, outcome=0.5
balance: finish labels + outcome labels
device: default sandbox CPU
result:
  finish loss: 0.9918 -> 0.6982
  finish accuracy: 26.2% -> 44.7%
  outcome loss: 1.9116 -> 1.0781
  outcome accuracy: 28.8% -> 39.6%
  evaluator load smoke: passed with --strategy-finish-outputs 2

run: runs/adaptive-search-best-outcome-head-gpu-v0/
labels: search-best
losses: finish=0.5, outcome=0.5
balance: finish labels + outcome labels
device: cuda:0
result:
  finish loss: 0.9930 -> 0.7009
  finish accuracy: 27.3% -> 44.3%
  outcome loss: 1.9216 -> 1.1157
  outcome accuracy: 29.5% -> 38.3%
```

Conclusion: the data path is now correct and the outcome head learns some
search-best structure, but binary finish from frozen last-layer heads is still
weak. The next promotion-oriented version should either unfreeze a small shared
value/strategy bottleneck with a KL-anchored policy freeze, or collect broader
search-best labels including negative/draw-heavy non-winning rows before using
finish gates in gameplay.

## 2026-06-19 Search-best value bottleneck probe

Implemented the middle update scope that the previous probe requested:

```text
adaptive_strategy_supervised.py:
  --update-scope strategy-value-heads

Effect:
  freeze trunk, policy head, action-Q head, and policy logits
  update strategy auxiliary heads
  update optional outcome head
  update shared pooled value bottleneck: value_linear1
```

This is deliberately not a full policy update. The goal is to check whether
finish/outcome supervision needs one shared pooled representation layer before
touching action logits.

Single high-gap shard, GPU:

```text
run: runs/adaptive-search-best-bottleneck-gpu-v0/
datasets:
  runs/adaptive-strategy-search-highgap-v1/*.npz
labels: search-best
balance: size-seat, finish labels, outcome labels
update scope: strategy-value-heads
device: cuda:0
result:
  finish loss: 0.9681 -> 0.6909
  finish accuracy: 52.1%
  outcome loss: 1.7825 -> 0.7212
  outcome accuracy: 53.9%
  evaluator load smoke: passed with --strategy-finish-outputs 2
```

Then collected a broader contact/high-gap search-best dataset without requiring
the whole trajectory to finish as a win:

```text
run: runs/adaptive-strategy-search-contact-highgap-v0/
teacher: adaptive-midgame-search-imitation-v1 + rollout-search
filters:
  min_save_turn = 80
  require_contact = true
  min_visible_enemy_cells = 1
  min_search_score_gap = 0.25
  no require-win / no require-finish
collection:
  rows kept: 4909
  shard rows: 941, 1585, 1272, 1111
  episode outcomes among finished episodes: 125 wins, 1 draw
search_best_outcome:
  draw: 4098
  win: 811
search_best win rate by stratum:
  8p0: 16.6%
  8p1: 14.2%
  12p0: 17.3%
  12p1: 18.3%
  16p0: 16.8%
  16p1: 16.0%
```

Mixed old high-gap + broader contact data, GPU:

```text
run: runs/adaptive-search-best-bottleneck-gpu-v1/
datasets:
  runs/adaptive-strategy-search-highgap-v1/*.npz
  runs/adaptive-strategy-search-contact-highgap-v0/*.npz
labels: search-best
balance: size-seat, finish labels, outcome labels
samples after size-seat balance: 5724
update scope: strategy-value-heads
device: cuda:0
result:
  finish loss: 0.8175 -> 0.6658
  finish accuracy: 21.5% -> 62.6%
  outcome loss: 1.2538 -> 0.6680
  outcome accuracy: 38.6% -> 59.8%
```

Interpretation: unfreezing the pooled value bottleneck is useful. The frozen
last-layer search-best probe plateaued around outcome accuracy `38%`; the
value-bottleneck mixed run reaches `59.8%` while keeping primitive policy logits
unchanged. The broader contact dataset is also healthier as representation data:
it has more rows and more local draw/negative labels, even though search-best
wins are rarer than in the decisive-only high-gap shard.

Conclusion: this is a representation checkpoint, not a promotion checkpoint.
The next gameplay-oriented step should consume these calibrated finish/outcome
signals in a gated Commander/finish probe, or collect fixed-v5 max250 search-best
contact rows so the same bottleneck can learn the 8x8-vs-v5 short-finish
problem directly. Do not run another direct action-CE or spatial rerank scale
sweep from this checkpoint.

## 2026-06-19 Fixed-v5 search-best contact data

Next we targeted the unresolved 8x8 fixed-v5 `max250` gate directly. The first
attempt used the previous best fixed-v5 imitation checkpoint as the search prior:

```text
runs/adaptive-fixed-v5-imitation-v5/ckpts/generals-adaptive-fixed-v5-imitation-v5-iter-000030.eqx
```

That checkpoint no longer deserializes under the current U-Net template: the
saved value-head leaf has old `(128, 64)` shape while the current template built
from `channels=64,96,128,64` expects `(64, 128)`. Rather than spend this turn on
legacy template compatibility, the data collection used the current-loadable
search-imitation U-Net:

```text
teacher prior:
  runs/adaptive-midgame-search-imitation-v1/generals-adaptive-midgame-search-imitation-v1.eqx
opponent:
  /home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v5.eqx
mode:
  fixed-v5 sample opponent, max250 truncation
search:
  top_k=4, rollout_steps=16, rollouts/action=1
filters:
  turn >= 80
  visible contact
  visible_enemy_cells >= 1
  search_score_gap >= 0.25
```

A small probe confirmed the label is useful before scaling:

```text
run: runs/adaptive-strategy-search-fixed-v5-probe-v0/
rows: 99
trajectory outcome: loss 83, win 16
search_best_outcome: draw 70, win 29
turn range: 80..128, median 98
```

Scaled GPU collection:

```text
run: runs/adaptive-strategy-search-fixed-v5-contact-highgap-v0/
raw: 4 shards x 32 envs x 260 steps = 33280 rows
kept rows by shard: 1707, 2144, 1900, 1774
total kept rows: 7525
trajectory outcome:
  loss: 3366
  draw: 2997
  win: 1162
search_best_outcome:
  loss: 1
  draw: 6578
  win: 946
search_best_outcome by seat:
  p0: draw 3318, win 437
  p1: loss 1, draw 3260, win 509
turn min / median / p90 / max:
  80 / 147 / 227 / 250
visible enemy count:
  mean 7.04, median 7
```

This is the first fixed-v5 search-best shard with a meaningful number of local
win labels. It is still draw-dominant, which is exactly the short-finish regime
we need to model.

Trained a fixed-v5-specific bottleneck representation checkpoint from the
previous Expander/search-best bottleneck checkpoint:

```text
run: runs/adaptive-search-best-bottleneck-fixed-v5-v0/
init:
  runs/adaptive-search-best-bottleneck-gpu-v1/generals-adaptive-search-best-bottleneck-gpu-v1.eqx
dataset:
  runs/adaptive-strategy-search-fixed-v5-contact-highgap-v0/*.npz
label_source:
  search-best
balance:
  size-seat, finish labels, outcome labels
update scope:
  strategy-value-heads
device:
  cuda:0
result:
  finish loss: 0.6865 -> 0.6641
  finish accuracy: 51.6% -> 57.8%
  outcome loss: 0.7071 -> 0.7107
  outcome accuracy: 50.4% -> 58.8%
  intent accuracy: 56.6% -> 60.3%
  belief loss: 0.1151 -> 0.0865
```

The outcome loss is noisy because the minibatch label-balancing changes the
effective class prior, but the accuracy and auxiliary losses improve. A GPU
evaluator load smoke against fixed v5 passed with `--strategy-finish-outputs 2`;
it used only one game/seat and is not a strength measurement.

Conclusion: fixed-v5 contact search-best data now exists and is viable for
finish/outcome representation learning. This checkpoint still does not change
primitive policy logits, so it is not a promotion candidate. The next useful
promotion-oriented step is to use these fixed-v5 search-best rows to train a
gate/candidate model that can decide when to invoke a command/worker correction,
or to collect the same fixed-v5 data with a longer `search_rollout_steps` budget
to increase the local win-label density. Do not spend the next round on
fixed-v5 action CE or spatial rerank scale sweeps.

## 2026-06-19 Fixed-v5 search-Q value probe

Added candidate-outcome value regression to `adaptive_strategy_supervised.py`:

```text
new args:
  --search-q-value-weight
  --search-q-score-scale
  --search-q-outcome-score-weight
target:
  search_outcomes loss/draw/win -> -1/0/+1
optional tie-break:
  outcome_score_weight * tanh(search_score / score_scale)
scope:
  strategy action-Q head only
```

The loader now reads `search_outcomes` from search-teacher shards and falls back
to invalid labels for older rank-only shards, so old rank probes still load.

First, a rank-only fixed-v5 probe trained from the fixed-v5 bottleneck
checkpoint:

```text
run:
  runs/adaptive-fixed-v5-search-q-rank-v0/
init:
  runs/adaptive-search-best-bottleneck-fixed-v5-v0/generals-adaptive-search-best-bottleneck-fixed-v5-v0.eqx
dataset:
  runs/adaptive-strategy-search-fixed-v5-contact-highgap-v0/*.npz
loss:
  --search-q-rank-weight 1.0
  --search-q-temperature 50.0
result:
  search-Q rank loss: 3.8671 -> 1.4467
  search-Q rank top1: 24.8% -> 26.1%
```

Fixed-v5 `max250` replacement probes were not promotion-worthy:

```text
128 games/seat, seed 86220:
  off:                 p0 10.94%, p1  7.81%, min  7.81%
  qreplace thr=1,m=4:  p0 12.50%, p1  8.59%, min  8.59%
  qrerank 0.001:       p0 10.94%, p1  7.81%, min  7.81%

256 games/seat, seed 86240:
  off:                 p0 10.55%, p1 10.55%, min 10.55%
  qreplace thr=1,m=4:  p0 10.55%, p1 11.33%, min 10.55%
```

Then trained direct candidate outcome value regression:

```text
run:
  runs/adaptive-fixed-v5-search-q-value-v0/
init:
  runs/adaptive-search-best-bottleneck-fixed-v5-v0/generals-adaptive-search-best-bottleneck-fixed-v5-v0.eqx
dataset:
  runs/adaptive-strategy-search-fixed-v5-contact-highgap-v0/*.npz
loss:
  --search-q-value-weight 1.0
  --search-q-score-scale 1000.0
  --search-q-outcome-score-weight 0.0
result:
  search-Q value loss: 14.0065 -> 0.3222
  search-Q value top1: 22.3% -> 27.5%
```

GPU fixed-v5 `max250` probes with the value checkpoint:

```text
128 games/seat, seed 86280:
  off:
    p0 16/50/62, win 12.50%, draw 48.44%
    p1 12/39/77, win  9.38%, draw 60.16%
    min 9.38%

  qreplace threshold=0.25, policy_margin=4:
    p0 11/55/62, win  8.59%, draw 48.44%
    p1  7/41/80, win  5.47%, draw 62.50%
    min 5.47%

  qreplace threshold=1.0, policy_margin=4:
    p0 16/49/63, win 12.50%, draw 49.22%
    p1 10/43/75, win  7.81%, draw 58.59%
    min 7.81%
```

Conclusion: both rank and absolute candidate-outcome Q losses are learnable, but
they do not provide a reliable primitive action replacement signal. The result
matches the earlier spatial rerank failures: a one-step action gate is too short
a path for the midgame finish signal. Stop Q-rerank/Q-replace threshold scans.
The next mainline should be Midgame Decisive Trajectory Imitation:

```text
collect:
  v5 + rollout-search winning/contact trajectories
  U-Net base + rollout-search winning/contact trajectories
  Plan-Q oracle best-plan leads-to-win windows

filter:
  turn >= 80 or 100
  visible contact
  high search/plan gap
  trajectory win or finish_within_250

train:
  main U-Net policy with KL anchor + small action CE
  finish/outcome/belief/intent as primary auxiliary signal
```

This should train the policy on the whole gather/attack/finish chain rather than
asking an undercalibrated action-Q head to override individual primitive moves.

## 2026-06-19 Midgame decisive trajectory imitation v0-v2

We then moved directly into Midgame Decisive Trajectory Imitation on GPU. The
first shard used strict fixed-v5 `max250` filters:

```text
run:
  runs/adaptive-midgame-decisive-fixed-v5-v0/
teacher/search prior:
  runs/adaptive-midgame-search-imitation-v1/generals-adaptive-midgame-search-imitation-v1.eqx
opponent:
  /home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v5.eqx
search:
  top_k=4
  rollout_steps=32
  rollouts/action=1
filters:
  turn >= 80
  visible contact
  visible_enemy_cells >= 1
  trajectory win or finish_within_250
  terminal_window <= 120
  search_gap >= 0.25
  search_best_outcome == win
kept rows:
  104, 75, 98, 210, 89, 107 = 683 raw
  656 after size-seat balancing
```

This is the cleanest fixed-v5 decisive window shard so far: all saved samples
are contact-window wins/finishes, and all have a high-gap search-best winning
candidate. A direct policy-coupled imitation run:

```text
run:
  runs/adaptive-midgame-decisive-imitation-v0/
init:
  runs/adaptive-midgame-search-imitation-v1/generals-adaptive-midgame-search-imitation-v1.eqx
update:
  all weights
loss:
  policy_kl=1.0
  action_ce=0.30
  finish=0.50
  outcome=0.40
  belief=0.25
  intent=0.20
lr:
  1e-5
result:
  KL: 0.0004 -> 0.0547
  action CE: 3.0670 -> 2.3711
  finish accuracy: 73.0% -> 94.2%
  outcome accuracy: 100.0% positive-only, not meaningful
```

Fixed-v5 `max250`, 128 games/seat, seed `86360`:

```text
init adaptive-midgame-search-imitation-v1:
  p0 15/40/73, win 11.72%, draw 57.03%
  p1 20/44/64, win 15.62%, draw 50.00%
  min 11.72%

v0 decisive-only imitation:
  p0 14/40/74, win 10.94%, draw 57.81%
  p1 18/55/55, win 14.06%, draw 42.97%
  min 10.94%
```

Interpretation: v0 did alter finish behavior, especially p1 draw rate, but it
converted too many draws into losses.

Next we collected explicit draw-heavy contrast rows:

```text
run:
  runs/adaptive-midgame-draw-fixed-v5-v0/
same teacher/opponent/search settings
filters:
  turn >= 80
  visible contact
  visible_enemy_cells >= 1
  draw_only
  terminal_window <= 120
  search_gap >= 0.25
kept rows:
  650, 480, 584, 941, 666, 432 = 3753 draw rows
```

Training v1 mixed decisive and draw-heavy rows, using action CE only on winning
rows while balancing finish/outcome labels:

```text
run:
  runs/adaptive-midgame-decisive-imitation-v1/
datasets:
  decisive rows + draw-heavy rows
action_ce_mode:
  wins
loss:
  policy_kl=1.0
  action_ce=0.30
  finish=0.50 balanced
  outcome=0.40 balanced
  belief=0.25
  intent=0.20
result:
  KL: 0.0003 -> 0.0458
  action CE on winning rows: 3.1413 -> 2.4479
  outcome accuracy: 15.3% -> 75.0%
```

Fixed-v5 `max250`, 128 games/seat, seed `86360`:

```text
v1 mixed decisive/draw:
  p0 14/55/59, win 10.94%, draw 46.09%
  p1 10/45/73, win  7.81%, draw 57.03%
  min 7.81%
```

The draw contrast did teach outcome classification, but the policy update still
translated anti-draw pressure into extra losses.

Finally we tried a conservative v2 to test whether v1 was simply too aggressive:

```text
run:
  runs/adaptive-midgame-decisive-imitation-v2/
same data as v1
loss:
  policy_kl=5.0
  action_ce=0.10 on wins only
  finish/outcome/belief/intent same as v1
lr:
  3e-6
result:
  KL: 0.0000 -> 0.0018
  outcome accuracy: 15.1% -> 61.3%
```

Fixed-v5 `max250`, 128 games/seat, seed `86360`:

```text
v2 conservative:
  p0 14/46/68, win 10.94%, draw 53.12%
  p1 13/40/75, win 10.16%, draw 58.59%
  min 10.16%
```

Conclusion: the midgame decisive path is correctly wired and GPU-fast, but this
first data slice is not sufficient for promotion. The decisive shard is too
small and too win-only; the draw-heavy shard gives useful negatives but does not
identify safe winning actions. Do not tune v0-v2 loss weights further. The next
useful iteration is data quality:

```text
1. collect larger A1/A2 winning trajectories, not just terminal-window states
2. save first_contact->terminal and turn 80->180 gather/attack transitions
3. include search-success rollout actions for several steps after the chosen action
4. keep draw-heavy rows for outcome/finish heads only
5. keep primitive policy CE restricted to rows where search rollout actually wins
```

The key failure mode is no longer infrastructure or GPU availability; it is that
one-step terminal-window imitation does not yet encode the full
gather-attack-finish chain.

## 2026-06-19 Midgame contact search-win action gate v3

The v0-v2 failure suggested the next change should be data gating, not another
loss sweep. Added:

```text
adaptive_strategy_supervised.py --action-ce-weight-mode search-best-win
```

This keeps policy KL and auxiliary losses on every loaded row, but only applies
primitive action CE when a search-teacher shard has:

```text
search_best_outcome == win
```

The purpose is to train on broader midgame/contact trajectories while avoiding
action CE from draw/loss states. A focused unit test now verifies that loader
`action_weight` follows `search_best_outcome`, independent of the full-trajectory
outcome.

Collected a broader fixed-v5 `max250` midgame/contact shard:

```text
run:
  runs/adaptive-midgame-contact-searchwin-fixed-v5-v0/
teacher/search prior:
  runs/adaptive-midgame-search-imitation-v1/generals-adaptive-midgame-search-imitation-v1.eqx
opponent:
  /home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v5.eqx
search:
  top_k=4
  rollout_steps=32
  rollouts/action=1
filters:
  turn >= 80
  visible contact
  visible_enemy_cells >= 1
  outcome_known
  search_gap >= 0.25
kept rows by shard:
  2251, 1950, 2054, 2144, 1993, 1778, 2114, 2031
total rows:
  16315
trajectory outcome:
  loss 5365, draw 8347, win 2603
search_best_outcome:
  loss 12, draw 14152, win 2151
search_best_win action rows:
  13.18%
```

Trained a policy-coupled U-Net checkpoint:

```text
run:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/
init:
  runs/adaptive-midgame-search-imitation-v1/generals-adaptive-midgame-search-imitation-v1.eqx
update:
  all weights
loss:
  policy_kl=2.0
  action_ce=0.20
  action_ce_weight_mode=search-best-win
  finish=0.50 balanced
  outcome=0.40 balanced
  belief=0.25
  intent=0.20
lr:
  5e-6
result:
  KL: 0.0003 -> 0.0171
  action CE on search-best-win rows: 3.2851 -> 2.7896
  action row weight: ~0.132
  finish accuracy: 39.9% -> 49.8%
  outcome accuracy: 16.3% -> 53.4%
  belief loss: 0.1142 -> 0.0991
```

Fixed-v5 `max250`, 128 games/seat, seed `86600`:

```text
init:
  p0 13/41/74, win 10.16%, draw 57.81%
  p1  9/52/67, win  7.03%, draw 52.34%
  min 7.03%

v3:
  p0 13/40/75, win 10.16%, draw 58.59%
  p1 16/43/69, win 12.50%, draw 53.91%
  min 10.16%
```

Fixed-v5 `max250`, 256 games/seat, seed `86640`:

```text
init:
  p0 25/95/136, win  9.77%, draw 53.12%
  p1 30/89/137, win 11.72%, draw 53.52%
  min 9.77%

v3:
  p0 41/96/119, win 16.02%, draw 46.48%
  p1 29/95/132, win 11.33%, draw 51.56%
  min 11.33%
```

Expander adaptive 8/12/16, 128 games/row, seed `86680`:

```text
init min:
  75.00%
v3 min:
  75.78%

row details:
  8p0  75.00% -> 75.78%
  8p1  78.12% -> 77.34%
  12p0 85.94% -> 82.81%
  12p1 83.59% -> 82.81%
  16p0 78.91% -> 82.03%
  16p1 75.00% -> 77.34%
```

Conclusion: this is the first fixed-v5 midgame imitation variant in this round
with a positive 256-row fixed-v5 signal and no immediate Expander regression.
It is still far below the fixed-v5 promotion target, so do not promote yet. The
useful next step is to scale this exact data recipe, not tune thresholds:

```text
1. collect 3-5x more midgame/contact/high-gap rows with the same filters
2. keep action_ce_weight_mode=search-best-win
3. add 12/16 Expander contact rows to protect adaptive rows during full update
4. re-evaluate fixed-v5 max250 at 512 games/seat only if 256-row stays positive
```

This is a better direction than terminal-window-only decisive imitation because
it preserves draw/loss rows as negative strategic context while keeping primitive
action imitation tied to local search rollouts that actually win.

## 2026-06-19: Scaled Search-Best-Win / Decisive-Only Probes

Expanded the same midgame/contact/high-gap recipe and added 12/16 Expander
protection data. All artifacts were written under ignored `runs/`, not cache
directories.

Additional fixed-v5 `max250` search-best-win data:

```text
run:
  runs/adaptive-midgame-contact-searchwin-fixed-v5-v1/
filters:
  turn >= 80
  visible contact
  visible_enemy_cells >= 1
  outcome_known
  search_gap >= 0.25
kept rows:
  28474
trajectory outcome:
  loss 8554, draw 14679, win 5241
search_best_outcome:
  loss 15, draw 24796, win 3663
search_best_win action rows:
  12.86%
groups:
  8p0 14428, 8p1 14046
```

12/16 Expander protection data:

```text
run:
  runs/adaptive-midgame-contact-searchwin-expander-12-16-v0/
filters:
  turn >= 80
  visible contact
  visible_enemy_cells >= 1
  outcome_known
  search_gap >= 0.25
kept rows:
  36137
trajectory outcome:
  loss 2320, draw 2351, win 31466
search_best_outcome:
  loss 2, draw 26003, win 10132
search_best_win action rows:
  28.04%
groups:
  12p0 8083, 12p1 8721, 16p0 10312, 16p1 9021
```

Filtered decisive rows for positive-only trajectory imitation:

```text
run:
  runs/adaptive-midgame-contact-searchwin-decisive-filter-v0/
filter:
  search_best_outcome == win
kept rows:
  fixed-v5-v0: 2151
  fixed-v5-v1: 3663
  expander-12-16-v0: 10132
groups:
  8p0 2926, 8p1 2888
  12p0 2134, 12p1 2593
  16p0 3016, 16p1 2389
```

Main-policy probes from the v3 safe checkpoint:

```text
v4 mixed broad data:
  init: adaptive-midgame-search-imitation-v1
  data: fixed-v5-v0 + fixed-v5-v1 + expander-12-16-v0
  label_source: trajectory
  action_ce_weight_mode: search-best-win
  final KL: 0.0203
  fixed-v5 max250 256 seed86640:
    p0 30/105/121, win 11.72%, draw 47.27%
    p1 34/89/133, win 13.28%, draw 51.95%
    min 11.72%
  Expander 8/12/16 128 seed86680:
    8p0 78.12%, 8p1 78.91%
    12p0 79.69%, 12p1 84.38%
    16p0 73.44%, 16p1 76.56%
    min 73.44%
  verdict:
    fixed-v5 small positive, but 16p0 Expander regression; do not promote.

v5-lite conservative mixed:
  init: adaptive-midgame-contact-searchwin-imitation-v3
  policy_kl=4.0, action_ce=0.10, lr=2e-6
  final KL: 0.0055
  fixed-v5 max250 256 seed86640:
    p0 28/98/130, win 10.94%, draw 50.78%
    p1 24/86/146, win  9.38%, draw 57.03%
    min 9.38%
  Expander 8/12/16 128 seed86680:
    8p0 73.44%, 8p1 79.69%
    12p0 83.59%, 12p1 83.59%
    16p0 83.59%, 16p1 78.91%
    min 73.44%
  verdict:
    conservative broad update loses fixed-v5 and 8p0 Expander; do not promote.

v6 search-best labels:
  init: adaptive-midgame-contact-searchwin-imitation-v3
  label_source: search-best
  action_ce_weight_mode: search-best-win
  final KL: 0.0053
  fixed-v5 max250 256 seed86640:
    p0 29/103/124, win 11.33%, draw 48.44%
    p1 22/88/146,  win  8.59%, draw 57.03%
    min 8.59%
  verdict:
    full search-best labels are still draw-heavy; no Expander eval needed.

v7 broad + decisive oversample:
  init: adaptive-midgame-contact-searchwin-imitation-v3
  data: broad shards plus filtered decisive duplicate shards
  label_source: search-best
  action row weight: 0.368
  final KL: 0.0187
  fixed-v5 max250 256 seed86640:
    p0 34/92/130, win 13.28%, draw 50.78%
    p1 29/88/139, win 11.33%, draw 54.30%
    min 11.33%
  Expander 8/12/16 128 seed86680:
    8p0 72.66%, 8p1 75.78%
    12p0 83.59%, 12p1 85.16%
    16p0 79.69%, 16p1 78.91%
    min 72.66%
  verdict:
    decisive oversampling changes behavior but regresses 8p0; do not promote.

v8 decisive-only:
  init: adaptive-midgame-contact-searchwin-imitation-v3
  data: filtered decisive rows only
  label_source: search-best
  action_ce=0.80, policy_kl=2.0
  action row weight: 1.000
  final KL: 0.0784
  fixed-v5 max250 256 seed86640:
    p0 34/106/116, win 13.28%, draw 45.31%
    p1 28/86/142,  win 10.94%, draw 55.47%
    min 10.94%
  verdict:
    strong decisive-only action imitation lowers draw for p0 but does not raise
    min win rate. Do not run Expander promotion eval.
```

Current best safe checkpoint from this line remains:

```text
runs/adaptive-midgame-contact-searchwin-imitation-v3/generals-adaptive-midgame-contact-searchwin-imitation-v3.eqx
```

Interpretation:

```text
1. Scaling broad search-best-win contact data did not solve fixed-v5.
2. Using search_best_outcome for finish/outcome labels is too draw-heavy.
3. Positive-only decisive imitation is learnable but still cannot compress the
   rollout-search policy into one primitive action head.
4. The remaining gap is execution-conditioned planning, not another CE weight.
```

Next decision: stop action-CE imitation sweeps on these shards. Use the
decisive filtered data as supervision for a plan-conditioned Worker or gated
plan executor:

```text
source/target proposal from existing spatial heads
plan-conditioned Worker action head
gate by finish/draw confidence
mix with base policy rather than overwriting primitive logits globally
```

## 2026-06-19: Belief-Main-Stack Strategy Worker Probe

The decisive-only action imitation results above suggested that primitive CE is
not enough, so the next probe moved the same decisive rows into a learned
target-conditioned Worker.

Code changes:

```text
adaptive_plan_worker_supervised.py:
  added --dataset-format strategy
  strategy format reads:
    obs
    legal_mask
    active
    source_heatmap
    target_heatmap
    teacher_action_index
  command planes:
    source_one_hot from source_heatmap argmax
    target_one_hot from target_heatmap argmax
    route_potential to target

evaluate_adaptive_policy.py:
  added --strategy-plan-worker-command-source
  spatial:
    use strategy-spatial source/target heads
  belief-main-stack:
    use belief enemy_general_logits as target
    use main-stack/route heuristic for source
    requires --strategy-aux only, not --strategy-spatial-aux
```

The first loader version incorrectly filtered strategy rows by applying a
4-direction legal mask directly to 8-plane full/half action indices. That kept
only `2137` examples. The fix expands the 4-direction legal mask to full+half
8 action planes before checking `teacher_action_index`.

Corrected Worker training:

```text
run:
  runs/adaptive-strategy-decisive-worker-v1/
data:
  runs/adaptive-midgame-contact-searchwin-decisive-filter-v0/
format:
  strategy
examples:
  15941
network:
  U-Net 64,96,128,64
input:
  38 channels = 35 adaptive/fog/history + 3 command planes
loss:
  action=0.50
  source=1.00
  direction=1.00
epochs:
  60
```

Offline training improved executor labels:

```text
loss:
  6.8461 -> 3.5161
action accuracy:
  7.0% -> 29.3%
source accuracy:
  24.9% -> 67.7%
direction accuracy:
  31.0% -> 41.6%
```

Fixed-v5 `max250`, 128 games/seat, seed `87060`, safe v3 policy:

```text
off:
  p0 15/43/70, win 11.72%, draw 54.69%
  p1 11/40/77, win  8.59%, draw 60.16%
  min 8.59%

belief-main-stack worker, scale 0.02:
  p0 16/43/69, win 12.50%, draw 53.91%
  p1 11/35/82, win  8.59%, draw 64.06%
  min 8.59%

belief-main-stack worker, scale 0.05:
  p0 12/40/76, win  9.38%, draw 59.38%
  p1  7/45/76, win  5.47%, draw 59.38%
  min 5.47%

belief-main-stack worker, scale 0.02, worker margin >= 1.0:
  p0 16/43/69, win 12.50%, draw 53.91%
  p1 10/39/79, win  7.81%, draw 61.72%
  min 7.81%
```

Conclusion:

```text
The strategy-shard Worker is trainable and the belief-main-stack command path
lets us use safe non-spatial checkpoints. Online fixed-v5 results are still not
promotion-worthy: low scale only shifts p0 slightly and does not fix p1; higher
scale hurts both seats; simple worker-margin gating does not recover min win
rate.
```

Do not continue scale/margin scans here. The useful next step is a learned gate
or scorer trained on whether the Worker actually improves the base action in
rollout, not more unconditional Worker logit bias.

## 2026-06-19: Belief-Main-Stack Worker Gate Probe

The next probe implemented a learned Plan-Worker gate rather than another
logit-scale sweep.

Code changes:

```text
adaptive_command_gate_supervised.py:
  added --dataset-format strategy-worker
  loads a feature U-Net and a learned Plan-Worker
  builds the existing 12-feature command-gate vector for the Worker top-1 action
  positive label:
    Worker top-1 == rollout-search teacher action
    and decisive row is search_best_outcome == win
    optionally include finish_within_250 rows

evaluate_adaptive_policy.py:
  added --strategy-plan-worker-gate-path
  added --strategy-plan-worker-gate-threshold
  added --strategy-plan-worker-gate-hidden-dim
  lets the gate replace the current primitive action with the Worker top-1
  supports 3-output finish heads by using sigmoid(last finish logit)
```

GPU training command used the safe v3 feature model and the decisive strategy
Worker:

```text
feature model:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/generals-adaptive-midgame-contact-searchwin-imitation-v3.eqx
worker:
  runs/adaptive-strategy-decisive-worker-v1/generals-adaptive-strategy-decisive-worker-v1.eqx
datasets:
  runs/adaptive-midgame-contact-searchwin-fixed-v5-v0/*.npz
  runs/adaptive-midgame-contact-searchwin-fixed-v5-v1/*.npz
output:
  runs/adaptive-strategy-worker-gate-v1/generals-adaptive-strategy-worker-gate-v1.eqx
```

Training distribution:

```text
device:
  cuda:0
rows:
  44789
worker changed base greedy action:
  17311
worker matched rollout-search teacher:
  3756
decisive rows:
  10771
training examples:
  17311
positive fraction:
  5.52%
```

Training fit was weak but nonzero:

```text
epoch 40:
  loss 0.6253
  weighted accuracy 63.5%
  P+ 0.561
  P- 0.439
```

Fixed-v5 `max250`, 128 games/seat, seed `87060`, safe v3 policy:

```text
no gate:
  p0 15/43/70, win 11.72%, draw 54.69%
  p1 11/40/77, win  8.59%, draw 60.16%
  min 8.59%

Plan-Worker gate threshold 0.55:
  p0 11/52/65, win  8.59%, draw 50.78%
  p1  8/45/75, win  6.25%, draw 58.59%
  min 6.25%

Plan-Worker gate threshold 0.65:
  p0 14/44/70, win 10.94%, draw 54.69%
  p1 10/38/80, win  7.81%, draw 62.50%
  min 7.81%
```

Conclusion:

```text
The learned gate is wired correctly and can train from strategy shards, but the
offline proxy label is too weak for promotion. The 0.55 gate reduces p0 draw but
also increases losses; the safer 0.65 gate still misses the no-gate p1 baseline.
Do not continue threshold scans or proxy-gate training from these labels.
```

Next decision:

```text
Return to direct Midgame Decisive Trajectory Imitation. The repeated failure
pattern is now consistent:
  spatial rerank scale fails
  worker logit bias fails
  worker margin gate fails
  offline proxy worker gate fails

The strategy signal should be pushed into the U-Net main policy via decisive
trajectory supervision, not inserted as an inference-time primitive action
override.
```

## 2026-06-19: Safe-v3 Search-Win Data Refresh and Domain Balance

The previous fixed-v5 search-win shards were useful, but their search prior was
not the current safe v3 checkpoint. This round recollected midgame contact rows
with safe v3 as the rollout-search prior, then tested whether the signal should
enter the main U-Net policy.

New fixed-v5 data:

```text
run:
  runs/adaptive-midgame-contact-searchwin-fixed-v5-safev3-v0/
teacher/search prior:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/generals-adaptive-midgame-contact-searchwin-imitation-v3.eqx
opponent:
  /home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v5.eqx
search:
  top_k=4
  rollout_steps=32
  rollouts/action=1
filters:
  turn >= 80
  visible contact
  visible_enemy_cells >= 1
  outcome_known
  search_gap >= 0.25
rows:
  62024
trajectory outcome:
  loss 20157
  draw 31659
  win  10208
search_best_outcome:
  loss 19
  draw 53815
  win  8190
groups:
  8p0 30561, search-win 4170, trajectory-win 5485
  8p1 31463, search-win 4020, trajectory-win 4723
```

Fixed-v5-only policy-coupled training:

```text
run:
  runs/adaptive-midgame-contact-searchwin-safev3-imitation-v0/
init:
  adaptive-midgame-contact-searchwin-imitation-v3
data:
  fixed-v5-safev3-v0 only
balance:
  size-seat
loss:
  policy_kl=3.0
  action_ce=0.15
  action_ce_weight_mode=search-best-win
  finish=0.50 balanced
  outcome=0.40 balanced
  belief=0.25
  intent=0.20
lr:
  2e-6
final:
  KL 0.0036
  action CE 2.8650 -> 2.7161
  outcome acc 47.9% -> 52.3%
```

Fixed-v5 `max250`, 128 games/seat, seed `88240`:

```text
safe v3:
  p0  9/50/69, win  7.03%, draw 53.91%
  p1 18/52/58, win 14.06%, draw 45.31%
  min 7.03%

v0 fixed-v5-only:
  p0 12/47/69, win  9.38%, draw 53.91%
  p1 17/47/64, win 13.28%, draw 50.00%
  min 9.38%
```

Expander 8/12/16, 128 games/row, seed `88260`:

```text
safe v3:
  8p0 73.44%, 8p1 74.22%
  12p0 78.12%, 12p1 80.47%
  16p0 77.34%, 16p1 85.94%
  min 73.44%

v0 fixed-v5-only:
  8p0 68.75%, 8p1 66.41%
  12p0 81.25%, 12p1 81.25%
  16p0 72.66%, 16p1 75.78%
  min 66.41%
```

Conclusion from v0:

```text
Safe-v3 fixed-v5 data has a real fixed-v5 signal, but using it alone causes
8x Expander forgetting. The issue is data-domain balance, not another CE weight.
```

New Expander protection data:

```text
run:
  runs/adaptive-midgame-contact-searchwin-expander-safev3-v0/
teacher/search prior:
  safe v3
opponent:
  Expander
grid sizes:
  8,12,16
rows:
  23361
trajectory outcome:
  loss 2573
  draw 582
  win  20206
search_best_outcome:
  loss 3
  draw 15709
  win  7649
groups:
  8p0 3035, search-win 907, trajectory-win 2254
  8p1 2943, search-win 852, trajectory-win 2432
  12p0 3838, search-win 1337, trajectory-win 2969
  12p1 3783, search-win 1302, trajectory-win 3456
  16p0 4257, search-win 1535, trajectory-win 4002
  16p1 5505, search-win 1716, trajectory-win 5093
```

Added trainer support:

```text
adaptive_strategy_supervised.py --balance-strata size-seat-domain
```

This reads each shard's JSON sidecar and creates a coarse data-domain label
from `opponent_policy_path` or `opponent`. It then balances by
`(grid_size, seat, domain)`, which prevents fixed-v5 8x rows from drowning
Expander 8x protection rows.

Mixed probes:

```text
v1:
  data: fixed-v5-safev3-v0 + expander-safev3-v0
  balance: size-seat
  samples after balance: 22698
  final KL: 0.0033
  fixed-v5 max250 seed88240:
    p0 win  7.81%, draw 57.03%
    p1 win 11.72%, draw 50.78%
    min 7.81%
  Expander seed88260:
    8p0 69.53%, 8p1 69.53%
    12p0 87.50%, 12p1 84.38%
    16p0 76.56%, 16p1 75.78%
    min 69.53%

v2:
  data: fixed-v5-safev3-v0 + expander-safev3-v0
  balance: size-seat-domain
  samples after balance: 23544
  final KL: 0.0030
  fixed-v5 max250 seed88240:
    p0 win  6.25%, draw 60.16%
    p1 win 14.84%, draw 49.22%
    min 6.25%
  Expander seed88260:
    8p0 71.09%, 8p1 73.44%
    12p0 83.59%, 12p1 81.25%
    16p0 78.91%, 16p1 72.66%
    min 71.09%
```

Conclusion:

```text
v0 proves safe-v3 fixed-v5 search-win data can improve fixed-v5 smoke, but it
forgets Expander. v1/v2 prove Expander protection data helps recover the
Expander rows, especially 8x, but current equal-domain mixing gives back too
much fixed-v5 p0. None of v0/v1/v2 is a promotion candidate.
```

Next decision:

```text
Do not sweep CE/KL or domain ratio blindly. The next useful step is to make
the supervised update aware of fixed-v5 p0 as the weak target:
  1. keep domain-balanced Expander protection,
  2. upweight fixed-v5 p0 search-win rows explicitly or collect more p0 wins,
  3. keep fixed-v5 p1 and Expander rows as anchors,
  4. consider a small policy-head-only adapter or LoRA-style delta instead of
     updating the whole U-Net trunk.
```

## 2026-06-19: Policy-Head-Only Decisive Imitation Probe

The previous safe-v3 search-win runs showed a real fixed-v5 signal, but full
U-Net updates moved Expander rows too much. I added:

```text
adaptive_strategy_supervised.py --update-scope policy-heads
```

This freezes the shared trunk and value heads, while allowing only
`policy_conv`, `pass_linear`, the strategy auxiliary heads, and the optional
outcome head to update. The goal was to test whether decisive midgame imitation
can enter the primitive policy without rewriting the U-Net representation.

Mixed-domain policy-head run:

```text
run:
  runs/adaptive-midgame-contact-searchwin-safev3-policyhead-v0/
data:
  fixed-v5-safev3-v0 + expander-safev3-v0
balance:
  size-seat-domain
lr:
  2e-6
final:
  KL 0.0001
  action CE 2.8122 -> 2.7898
  finish acc 46.6% -> 52.3%
  outcome acc 43.9% -> 46.3%
```

128-row smoke:

```text
fixed-v5 max250 seed88240:
  p0 win  7.81%, draw 54.69%
  p1 win 10.94%, draw 50.00%
  min 7.81%

Expander seed88260:
  8p0 74.22%, 8p1 71.88%
  12p0 78.91%, 12p1 87.50%
  16p0 79.69%, 16p1 75.00%
  min 71.88%
```

This did not beat the safe-v3 baseline strongly enough. Domain-balanced
Expander protection preserved more rows than full mixed update, but fixed-v5
gain was too small.

Fixed-v5-only policy-head run:

```text
run:
  runs/adaptive-midgame-contact-searchwin-safev3-policyhead-fixed-v0/
data:
  fixed-v5-safev3-v0 only
balance:
  size-seat
lr:
  2e-6
final:
  KL 0.0003
  action CE 2.9029 -> 2.8532
  outcome acc 39.7% -> 50.6%
```

128-row smoke:

```text
fixed-v5 max250 seed88240:
  p0 win  8.59%, draw 52.34%
  p1 win 15.62%, draw 47.66%
  min 8.59%

Expander seed88260:
  8p0 75.00%, 8p1 72.66%
  12p0 85.16%, 12p1 85.94%
  16p0 76.56%, 16p1 73.44%
  min 72.66%
```

256-row fixed-v5 same-seed triage:

```text
safe-v3 baseline, seed88340:
  p0 win 10.55%, draw 50.00%
  p1 win 11.33%, draw 50.39%
  min 10.55%

policyhead-fixed-v0, seed88340:
  p0 win 12.11%, draw 48.83%
  p1 win 12.50%, draw 48.05%
  min 12.11%
```

This is a real but modest positive signal: fixed-v5 improves by about 1.6pp on
the matched 256-row gate, while 128-row Expander only drops from the recent
safe-v3 smoke baseline `73.44%` to `72.66%`. It is not a promotion candidate,
but it is the cleanest evidence so far that fixed-v5 decisive imitation can be
inserted with much less Expander forgetting when the trunk is frozen.

A stronger output-head single point was negative:

```text
run:
  runs/adaptive-midgame-contact-searchwin-safev3-policyhead-fixed-v1/
change:
  lr 1e-5, same data/loss
final:
  KL 0.0023
  action CE 2.8945 -> 2.7552

fixed-v5 max250 256-row seed88340:
  p0 win 10.16%, draw 54.69%
  p1 win  8.98%, draw 48.44%
  min 8.98%
```

Conclusion:

```text
policy-head-only fixed-v5 imitation is useful only as a small adapter.
Pushing the output head harder improves offline CE but breaks seat balance.
Do not promote v0/v1. Keep safe-v3 as active base.

Next useful step:
  train a small gated/adapter delta or LoRA-style policy head that activates
  only in fixed-v5-like midgame/contact states, or collect cleaner p0 decisive
  trajectories. Avoid more global CE/KL sweeps.
```

## 2026-06-19: p0-Focused Policy-Head Static Mix Probe

The fixed-v5 safe-v3 shard was not p0-scarce:

```text
fixed-v5-safev3-v0:
  p0 rows 30561, search-win 4170, trajectory-win 5485
  p1 rows 31463, search-win 4020, trajectory-win 4723
```

So the weak p0 rows are not explained by row count alone. I derived a p0-only
fixed-v5 shard to test whether removing fixed-v5 p1 imitation pressure helps:

```text
run:
  runs/adaptive-midgame-contact-searchwin-fixed-v5-safev3-p0-v0/
source:
  fixed-v5-safev3-v0
filter:
  seat == 0
rows:
  30561
search-best win:
  4170
```

Unbalanced p0 fixed-v5 + Expander protection:

```text
run:
  runs/adaptive-midgame-contact-searchwin-safev3-policyhead-p0mix-v0/
data:
  fixed-v5-safev3-p0-v0 + expander-safev3-v0
balance:
  none
samples:
  53922
lr:
  2e-6
final:
  KL 0.0002
  action CE 2.8185 -> 2.7862
  outcome acc 42.2% -> 46.9%
```

Matched fixed-v5 `max250`, 256 games/seat, seed `88340`:

```text
safe-v3 baseline:
  p0 win 10.55%, draw 50.00%
  p1 win 11.33%, draw 50.39%
  min 10.55%

policyhead-fixed-v0:
  p0 win 12.11%, draw 48.83%
  p1 win 12.50%, draw 48.05%
  min 12.11%

p0mix-v0:
  p0 win 13.67%, draw 47.66%
  p1 win 12.50%, draw 49.61%
  min 12.50%
```

Expander 8/12/16, 128 games/row, seed `88260`:

```text
p0mix-v0:
  8p0 73.44%, 8p1 71.88%
  12p0 77.34%, 12p1 88.28%
  16p0 80.47%, 16p1 70.31%
  min 70.31%
```

The p0-focused unbalanced mix improved the fixed-v5 gate, especially p0, but
the Expander 16p1 row fell too far. This is not a promotion candidate.

Domain-balanced p0 fixed-v5 + Expander protection:

```text
run:
  runs/adaptive-midgame-contact-searchwin-safev3-policyhead-p0domain-v0/
data:
  fixed-v5-safev3-p0-v0 + expander-safev3-v0
balance:
  size-seat-domain
samples:
  20601
lr:
  2e-6
final:
  KL 0.0001
  action CE 2.7493 -> 2.7333
```

Matched fixed-v5 `max250`, 256 games/seat, seed `88340`:

```text
p0domain-v0:
  p0 win 12.50%, draw 50.78%
  p1 win 10.94%, draw 50.78%
  min 10.94%
```

Expander 8/12/16, 128 games/row, seed `88260`:

```text
p0domain-v0:
  8p0 74.22%, 8p1 71.09%
  12p0 77.34%, 12p1 82.03%
  16p0 82.81%, 16p1 78.91%
  min 71.09%
```

Conclusion:

```text
p0 fixed-v5 rows have useful signal, but static mixing cannot separate the
fixed-v5 max250 finish behavior from Expander generalization. Unbalanced mixing
gets the best fixed-v5 result so far in this subline (12.50% min) but hurts
Expander; domain balance protects some Expander rows but loses most fixed-v5
gain.

Do not continue ratio sweeps. The next step should be conditional:
  1. train a gate that detects fixed-v5-like midgame/contact finish states, or
  2. add a small adapter/delta policy head whose contribution is gated by
     finish/draw-risk or opponent/contact features.

Active base remains:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/generals-adaptive-midgame-contact-searchwin-imitation-v3.eqx
```

## 2026-06-19: Finish-Gated Policy Adapter Probe

Added evaluator support for loading a second adaptive checkpoint as a policy
adapter:

```text
evaluate_adaptive_policy.py
  --policy-adapter-path <adapter.eqx>
  --policy-adapter-scale <scale>
  --policy-adapter-finish-threshold <p>
```

The adapter uses the same network template as the base policy. At inference,
the evaluator computes:

```text
centered_delta = adapter_logits - base_logits
base_logits += scale * gate * centered_delta
```

where `centered_delta` is centered over legal actions only. With no finish
threshold, `scale=1.0` reproduces the adapter checkpoint's policy logits up to a
constant shift. With `--policy-adapter-finish-threshold`, the gate is a hard
on/off switch from the adapter strategy finish probability; multi-horizon
finish heads use the last horizon.

Smoke:

```text
base:
  safe-v3
adapter:
  runs/adaptive-midgame-contact-searchwin-safev3-policyhead-p0mix-v0/
scale:
  0.5
finish threshold:
  0.5
result:
  GPU load/JIT path works
```

Fixed-v5 `max250`, 256 games/seat, seed `88340`:

```text
safe-v3 baseline:
  min 10.55%

p0mix full policy:
  p0 win 13.67%, draw 47.66%
  p1 win 12.50%, draw 49.61%
  min 12.50%

p0mix adapter, scale=0.5:
  p0 win 10.94%, draw 50.39%
  p1 win 11.33%, draw 49.61%
  min 10.94%

p0mix adapter, scale=1.0:
  p0 win 13.67%, draw 47.66%
  p1 win 12.50%, draw 49.61%
  min 12.50%

p0mix hard finish-gated adapter, scale=1.0, threshold=0.5:
  p0 win 12.11%, draw 50.00%
  p1 win 13.28%, draw 48.05%
  min 12.11%
```

The `scale=1.0` no-gate result confirms the adapter delta implementation
matches direct p0mix policy behavior. The finish-gated version preserves most
of the fixed-v5 gain without using the adapter everywhere.

Expander 8/12/16, 128 games/row, seed `88260`:

```text
p0mix full policy:
  min 70.31%
  weak row: 16p1 70.31%

p0mix hard finish-gated adapter, scale=1.0, threshold=0.5:
  8p0 73.44%, 8p1 72.66%
  12p0 77.34%, 12p1 86.72%
  16p0 83.59%, 16p1 81.25%
  min 72.66%
```

Conclusion:

```text
finish-gated policy adapter is the first conditional adapter probe that keeps a
fixed-v5 improvement while recovering most Expander regression from the
unconditional p0mix policy.

It is still not a promotion candidate:
  fixed-v5 min 12.11% is only modestly above safe-v3 10.55%
  Expander min 72.66% remains below the recent safe-v3 smoke 73.44%

Next step:
  train a learned adapter gate on rollout replacement outcomes using features
  that already exist at inference time:
    finish probability
    outcome/draw logits
    policy delta margin
    seat
    contact/visible enemy features
  This should replace threshold sweeps.
```

## 2026-06-19: Learned Policy Adapter Gate Probe

Implemented a learned gate for the same policy-head adapter delta:

```text
adaptive_policy_adapter_gate_supervised.py
  inputs:
    base policy logits
    adapter policy logits
    adapter finish probability
    visible enemy density
    owned/enemy army log density
    active fraction
    adapter-changes-action flag
    seat

evaluate_adaptive_policy.py
  --policy-adapter-gate-path <gate.eqx>
  --policy-adapter-gate-threshold <p>
```

Training setup:

```text
base:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/generals-adaptive-midgame-contact-searchwin-imitation-v3.eqx

adapter:
  runs/adaptive-midgame-contact-searchwin-safev3-policyhead-p0mix-v0/generals-adaptive-midgame-contact-searchwin-safev3-policyhead-p0mix-v0.eqx

positive data:
  runs/adaptive-midgame-contact-searchwin-fixed-v5-safev3-p0-v0/*.npz
  positive rows require:
    path contains fixed-v5-safev3-p0
    adapter greedy top1 equals rollout-search teacher action
    search_best_outcome == win

negative/protection data:
  runs/adaptive-midgame-contact-searchwin-expander-safev3-v0/*.npz
```

Probe v0 trained only on rows where the adapter changed the base greedy top1:

```text
rows:       53,922
changed:       175
positives:       4
positive rate among examples: 2.29%
```

This fit the tiny offline label set but overgeneralized at evaluation time,
because most states where the adapter changes the stochastic distribution but
not greedy top1 were unseen by the gate.

Results:

```text
v0 learned gate, threshold 0.5

fixed-v5 max250, 256 games/seat, seed 88420:
  p0: 13.28%
  p1: 12.11%
  min: 12.11%

Expander 8/12/16, 128 games/row, seed 88440:
  8p0: 75.00%
  8p1: 69.53%
  12p0: 85.16%
  12p1: 81.25%
  16p0: 74.22%
  16p1: 75.78%
  min: 69.53%
```

Probe v1 added `--keep-unchanged-negatives` so states where base and adapter
share greedy top1 are explicitly negative:

```text
examples:   53,922
changed:       175
positives:       4
positive rate: 0.01%
final P+: ~0.154
final P-: ~0.001
```

Single calibrated evaluation at threshold `0.1`:

```text
fixed-v5 max250, 256 games/seat, seed 88480:
  p0: 11.33%
  p1:  9.38%
  min:  9.38%
```

Conclusion:

```text
Learned adapter gating is wired and GPU-trainable, but the current p0mix
adapter is too weak/sparse as a delta source:
  it changes greedy top1 on only 175 / 53,922 saved rows
  decisive positive gate labels are only 4 rows
  v0 keeps fixed-v5 signal but hurts Expander
  v1 learns to suppress the adapter but loses fixed-v5 gain

Do not sweep learned-gate thresholds. The next useful step is stronger adapter
data or a policy delta trained to expose more decisive state-conditioned
differences, then re-use this gate machinery.
```

## 2026-06-19: Domain-Filtered Action CE and Terminal Search-Win Probe

Added a loader-side action CE domain filter:

```text
adaptive_strategy_supervised.py
  --action-ce-path-contains <token>
```

When one or more tokens are provided, shards whose path does not contain any
token keep contributing policy KL, finish/outcome, belief, intent, and other
auxiliary losses, but their primitive action CE weight is zeroed. This lets
fixed-v5 decisive shards teach actions while Expander shards act as protection
data.

Before training, an adapter-delta inspection showed why the previous learned
gate was data-starved:

```text
p0mix-v0:
  total rows: 53,922
  greedy top1 changed: 175
  fixed-v5 changed rows: 90
  fixed-v5 changed rows with search-best win: 4

p0mix-v1, stronger action CE:
  greedy top1 changed: 933
  fixed-v5 changed rows: 397
  fixed-v5 changed rows with search-best win: 20
  expander changed rows with search-best win: 50
```

The stronger adapter created more deltas, but too many were in the Expander
protection domain, which explains why a static or learned gate could not isolate
fixed-v5 gains.

Domain-filtered policy-head probe:

```text
run:
  runs/adaptive-midgame-contact-searchwin-safev3-policyhead-p0fixedaction-v0/
data:
  fixed-v5-safev3-p0-v0 + expander-safev3-v0
action CE:
  search-best-win rows only
  path contains fixed-v5-safev3-p0
update:
  policy-heads
final:
  KL 0.0018
  ActW 0.077
```

Results:

```text
fixed-v5 max250, 128 games/seat, seed 88620:
  p0 10.94%
  p1  6.25%
  min  6.25%

Expander 8/12/16, 64 games/row, seed 88640:
  8p0 81.25%, 8p1 70.31%
  12p0 76.56%, 12p1 82.81%
  16p0 84.38%, 16p1 70.31%
  min 70.31%
```

Domain-filtered full-trunk p0 probe:

```text
run:
  runs/adaptive-midgame-contact-searchwin-safev3-main-p0fixedaction-v0/
data:
  fixed-v5-safev3-p0-v0 + expander-safev3-v0
balance:
  size-seat-domain
action CE:
  search-best-win rows only
  path contains fixed-v5-safev3-p0
update:
  all weights
final:
  KL 0.0056
  ActW 0.019
```

Results:

```text
fixed-v5 max250, 128 games/seat, seed 88720:
  p0 13.28%
  p1 10.16%
  min 10.16%

Expander 8/12/16, 64 games/row, seed 88740:
  8p0 82.81%, 8p1 65.62%
  12p0 82.81%, 12p1 89.06%
  16p0 76.56%, 16p1 82.81%
  min 65.62%
```

Domain-filtered full-trunk two-seat probe:

```text
run:
  runs/adaptive-midgame-contact-searchwin-safev3-main-fixedaction-v0/
data:
  fixed-v5-safev3-v0 + expander-safev3-v0
balance:
  size-seat-domain
action CE:
  search-best-win rows only
  path contains fixed-v5-safev3-v0
update:
  all weights
final:
  KL 0.0047
  ActW 0.033
```

Results:

```text
fixed-v5 max250, 128 games/seat, seed 88820:
  p0 10.16%
  p1 14.84%
  min 10.16%

Expander 8/12/16, 64 games/row, seed 88840:
  8p0 70.31%, 8p1 64.06%
  12p0 81.25%, 12p1 84.38%
  16p0 75.00%, 16p1 78.12%
  min 64.06%
```

Collected a stricter terminal/search-win fixed-v5 shard:

```text
run:
  runs/adaptive-midgame-terminal-searchwin-fixed-v5-safev3-v0/
teacher/search prior:
  adaptive-midgame-contact-searchwin-imitation-v3
opponent:
  fixed v5 sample
filters:
  turn >= 80
  contact
  visible_enemy_cells >= 1
  outcome_known
  terminal_window <= 120
  search_gap >= 0.25
  search_best_win
rows:
  1,722
search_best_outcome:
  win 1,722
trajectory outcome by seat:
  p0 loss/draw/win 238/231/551
  p1 loss/draw/win 120/165/417
```

Terminal winning-trajectory full-trunk probe:

```text
run:
  runs/adaptive-midgame-terminal-searchwin-safev3-main-terminalwin-v0/
data:
  terminal-searchwin fixed-v5 + expander-safev3-v0
balance:
  size-seat-domain
action CE:
  trajectory win rows only
  path contains terminal-searchwin
update:
  all weights
final:
  KL 0.0061
  ActW 0.141
```

Results:

```text
fixed-v5 max250, 128 games/seat, seed 89020:
  p0  8.59%
  p1 11.72%
  min  8.59%

Expander 8/12/16, 64 games/row, seed 89040:
  8p0 76.56%, 8p1 62.50%
  12p0 76.56%, 12p1 76.56%
  16p0 82.81%, 16p1 76.56%
  min 62.50%
```

Conclusion:

```text
The new domain filter works and is useful for controlled probes, but the
gameplay evidence is now consistent:
  primitive action CE remains too blunt, even when restricted by domain,
  seat, terminal window, search-best-win, or true trajectory wins.

The losses move, but fixed-v5 max250 does not break past the old 10%-12% band,
and 8x Expander player-1 is repeatedly the first row to collapse.

Stop global primitive CE imitation on these shards. The next useful path is to
use the terminal/search-win data as non-primitive supervision: finish/outcome
calibration, plan/target Q, or a plan-conditioned Worker trained/evaluated
before it is mixed into the base policy.
```

## 2026-06-19: Search-Best Aux-Only and Winning-Trajectory Worker Probe

The previous probe showed that primitive action CE is too blunt even when
domain-filtered. This probe moved the same signal into non-action heads and an
isolated Worker.

Binary search-best calibrator, no policy update:

```text
run:
  runs/adaptive-midgame-searchbest-binary-calibrator-v0/
data:
  fixed-v5-safev3-v0
  terminal-searchwin fixed-v5
  expander-safev3-v0
label source:
  search-best
finish head:
  binary, initialized from multi-horizon safe v3
update:
  strategy-value-heads
final:
  finish acc 56.7%
  outcome acc 47.2%
```

This is too weak to use as a gate. Letting the same labels update the trunk
under a high policy-KL anchor did not fix it:

```text
run:
  runs/adaptive-midgame-searchbest-auxonly-main-v0/
update:
  all weights
policy KL:
  8.0
action CE:
  0.0
final:
  KL 0.0009
  finish acc 51.1%
  outcome acc 53.5%

fixed-v5 max250, 128 games/seat, seed 89220:
  p0  7.81%
  p1 15.62%
  min  7.81%

Expander 8/12/16, 64 games/row, seed 89240:
  8p0 70.31%, 8p1 64.06%
  12p0 79.69%, 12p1 82.81%
  16p0 78.12%, 16p1 79.69%
  min 64.06%
```

Conclusion: even aux-only full-trunk search-best training is not safe enough;
the KL can remain numerically small while 8x Expander player-1 collapses.

Terminal/search-win Worker:

```text
run:
  runs/adaptive-terminal-searchwin-worker-v0/
data:
  terminal-searchwin fixed-v5, 1,721 examples after legal filtering
format:
  strategy source/target heatmaps + teacher action
final offline:
  action acc 38.8%
  source acc 70.7%
  direction acc 53.3%
```

Gameplay against fixed-v5 max250, safe-v3 base, 128 games/seat, seed 89320:

```text
base off:
  p0 14.06%
  p1  7.81%
  min  7.81%

terminal worker, belief-main-stack, scale 0.02:
  p0 14.84%
  p1  5.47%
  min  5.47%

terminal worker, belief-main-stack, scale 0.02, margin >= 1.0:
  p0 13.28%
  p1  5.47%
  min  5.47%
```

This Worker learned offline labels but hurt the weak p1 row, so the issue was
not just terminal data size.

Collected true search-controlled winning trajectories:

```text
run:
  runs/adaptive-midgame-searchwin-trajectory-fixed-v5-safev3-v0/
teacher/search prior:
  adaptive-midgame-contact-searchwin-imitation-v3
opponent:
  fixed v5 sample
filters:
  turn >= 50
  contact
  visible_enemy_cells >= 1
  outcome_known
  trajectory win
  terminal_window <= 160
rows:
  25,385 raw saved rows
  25,355 Worker examples after legal filtering
```

Winning-trajectory Worker:

```text
run:
  runs/adaptive-searchwin-trajectory-worker-v0/
final offline:
  action acc 37.8%
  source acc 67.8%
  direction acc 51.7%
```

Gameplay against fixed-v5 max250, safe-v3 base, 128 games/seat, seed 89520:

```text
base off:
  p0 11.72%
  p1  7.81%
  min  7.81%

winning-trajectory worker, belief-main-stack, scale 0.02:
  p0 11.72%
  p1  7.81%
  min  7.81%

winning-trajectory worker, belief-main-stack, scale 0.05:
  p0 12.50%
  p1  7.81%
  min  7.81%
```

Conclusion:

```text
True winning trajectories are valuable data: they produce a stable Worker and
avoid the p1 collapse seen in the tiny terminal-only Worker. But the current
belief-main-stack command source does not release that value; the Worker is
mostly a no-op at safe scale and only nudges p0 at higher scale.

Do not continue Worker scale scans. The next useful step is better command
selection, not a stronger executor: train a source/target or command gate on
true winning-trajectory / Plan-Q counterfactuals, then use the Worker only when
the command itself is high-confidence.
```

## 2026-06-19 Winning-Trajectory Worker Gate v0-v1

We then trained a learned gate for the true winning-trajectory Worker instead
of continuing scale scans.

First gate:

```text
run:
  runs/adaptive-searchwin-trajectory-worker-gate-v0/
data:
  runs/adaptive-midgame-searchwin-trajectory-fixed-v5-safev3-v0/
feature model:
  adaptive-midgame-contact-searchwin-imitation-v3
worker:
  adaptive-searchwin-trajectory-worker-v0
dataset format:
  strategy-worker
command source:
  belief-main-stack
examples:
  5,872 changed Worker actions
positive:
  11.14%
final:
  acc 63.2%
  P+ 0.541
  P- 0.459
```

Gameplay, fixed-v5 max250, 128 games/seat, seed 89520:

```text
base off:
  p0 11.72%
  p1  7.81%
  min  7.81%

gate v0, threshold 0.5:
  p0 15.62%
  p1 10.94%
  min 10.94%
```

Confirmation at 256 games/seat, seed 89620:

```text
base off:
  p0  8.98%
  p1  8.59%
  min  8.59%

gate v0, threshold 0.5:
  p0  9.38%
  p1 10.16%
  min  9.38%
```

This is the first Worker-gate path with a confirmed positive fixed-v5 signal,
but it was unsafe when applied to all sizes:

```text
Expander 8/12/16, 64 games/row, seed 89640:
  gate v0 all sizes min 40.62%
  weak row: 16p1 40.62%
```

Root cause: the Worker and gate were trained only on 8x8 fixed-v5
winning-trajectory rows, so applying the same primitive replacement to 12/16
created an out-of-domain inference-time override. Added:

```text
evaluate_adaptive_policy.py:
  --strategy-plan-worker-max-grid-size
```

When positive, both Plan-Worker rerank bias and gated replacement are disabled
for boards larger than the configured grid size.

Expander protection with `--strategy-plan-worker-max-grid-size 8`:

```text
Expander 8/12/16, 64 games/row, seed 89640:
  8p0 67.19%
  8p1 67.19%
  12p0 78.12%
  12p1 78.12%
  16p0 85.94%
  16p1 71.88%
  min 67.19%

same-seed base:
  min 65.62%
```

Second gate:

```text
run:
  runs/adaptive-searchwin-trajectory-worker-gate-v1/
change:
  --allow-nondecisive-worker-positives
reason:
  the whole shard contains true search-controlled winning trajectories, so a
  Worker action matching the search teacher is useful even when the local
  search_best_outcome field is not win.
positive:
  32.48%
final:
  acc 56.3%
  P+ 0.511
  P- 0.496
```

Gameplay with `--strategy-plan-worker-max-grid-size 8`:

```text
fixed-v5 max250, 128 games/seat, seed 89520:
  gate v1 threshold 0.5:
    p0 16.41%
    p1 13.28%
    min 13.28%

fixed-v5 max250, 256 games/seat, seed 89620:
  gate v1 threshold 0.5:
    p0 11.72%
    p1 11.33%
    min 11.33%

Expander 8/12/16, 64 games/row, seed 89640:
  p0/p1 rows:
    8x8  67.19% / 76.56%
    12x12 78.12% / 78.12%
    16x16 85.94% / 71.88%
  min 67.19%
```

Operational note: in the current Codex sandbox, JAX evaluator commands can see
`CUDA_ERROR_NO_DEVICE` and fall back to a very slow path. Running the evaluator
with GPU access outside the sandbox reports `Device: cuda:0` and completes the
same fixed-v5 128/256-row checks in seconds. Keep `UV_CACHE_DIR=/tmp/uv-cache`
and store generated checkpoints under `runs/`.

Conclusion:

```text
The winning-trajectory Worker gate has a real but still small fixed-v5 signal:
256-row min improves from 8.59% to 11.33% on the checked seed, without harming
12/16 Expander when max_grid_size=8 is enforced.

Do not promote it as a general adaptive policy. The next useful iteration is
to collect 12/16-compatible winning-trajectory Worker/gate data or train the
gate on explicit replacement outcome/Q labels, then use max_grid_size guards
until each size has in-domain evidence.
```

## 2026-06-19 - Strategy-worker row filters and grid-range gates

Code changes:

```text
adaptive_plan_worker_supervised.py:
  added strategy-row filters:
    --require-outcome-win
    --require-search-best-win
    --require-finish-within-250
  prints kept / total rows and stores the counts in dataset stats.

adaptive_command_gate_supervised.py:
  added strategy-worker filters:
    --filter-outcome-win
    --filter-search-best-win
    --filter-finish-within-250

adaptive_command_gate.py / evaluate_adaptive_policy.py:
  added active_area_fraction to command-gate features.
  evaluator reads sidecar feature_names, so old 12-feature gates and new
  13-feature gates both load.

evaluate_adaptive_policy.py:
  added --strategy-plan-worker-min-grid-size.
  min/max grid guards now define a size range for both Plan-Worker rerank bias
  and gated replacement.
```

Mixed 8/12/16 Worker/gate:

```text
worker:
  runs/adaptive-searchwin-trajectory-worker-mixed-v1/
  data:
    fixed-v5 winning trajectories
    Expander 12/16 winning trajectories
  filters:
    outcome_win + finish_within_250
  kept:
    53,422 / 61,522 rows
  final:
    Act 34.7%
    Src 67.5%
    Dir 48.7%

gate v1:
  runs/adaptive-searchwin-trajectory-worker-gate-mixed-v1/
  changed examples 17,044
  positive 32.01%
  final acc 53.5%

gate v2:
  runs/adaptive-searchwin-trajectory-worker-gate-mixed-v2/
  same data, plus active_area_fraction feature
  final acc 54.2%
```

Mixed gameplay:

```text
fixed-v5 max250, 128 games/seat, seed 89520:
  mixed v1 max12:
    p0 14.84%
    p1 11.72%
    min 11.72%
  mixed v2 max12:
    p0 19.53%
    p1 11.72%
    min 11.72%

Expander 8/12/16, 128 games/row, seed 89900:
  mixed v2 max12:
    min 71.09%
  same-seed base:
    min 72.66%
```

The active-area feature slightly improves the offline gate and some fixed-v5
p0 behavior, but it does not solve seat/size transfer. Mixed v1/v2 should not
be promoted.

8x8-only gate recheck:

```text
gate:
  runs/adaptive-searchwin-trajectory-worker-gate-v1/

Expander 8/12/16, max_grid_size=8:
  128 games/row, seed 89900:
    min 74.22%
    same-seed base 72.66%

  256 games/row, seed 90000:
    8p0 74.61%
    8p1 69.14%
    12p0 80.08%
    12p1 79.30%
    16p0 77.34%
    16p1 75.00%
    min 69.14%

same 256 seed base:
  8p0 72.27%
  8p1 73.05%
  12p0 80.08%
  12p1 79.30%
  16p0 77.34%
  16p1 75.00%
  min 72.27%
```

The 8x8-only gate is a fixed-v5 diagnostic, not an Expander promotion. It can
flip 8p0/8p1 under larger evaluation and should stay behind a size guard.

Protect Worker/gate with Expander rows:

```text
worker:
  runs/adaptive-searchwin-trajectory-worker-protect-v2/
  data:
    fixed-v5 winning trajectories
    Expander all-size protection trajectories
    Expander 12/16 trajectories
  filters:
    outcome_win + finish_within_250
  kept before sampling:
    73,624 / 84,883 rows
  trained examples:
    60,000
  final:
    Act 33.7%
    Src 67.2%
    Dir 47.7%

gate:
  runs/adaptive-searchwin-trajectory-worker-gate-protect-v3/
  kept:
    73,657 / 84,883 rows
  changed examples:
    25,317
  positive:
    30.63%
  final acc:
    53.7%
  feature_dim:
    13
```

Protect gameplay without min-grid guard:

```text
Expander 8/12/16, 128 games/row, seed 89900, max_grid_size=16:
  8p0 78.12%
  8p1 71.09%
  12p0 82.81%
  12p1 82.81%
  16p0 82.03%
  16p1 73.44%
  min 71.09%

same-seed base:
  min 72.66%
```

Protect gameplay with grid-range guard:

```text
Expander 8/12/16, 128 games/row, seed 89900:
  --strategy-plan-worker-min-grid-size 12
  --strategy-plan-worker-max-grid-size 16

  8p0 72.66%
  8p1 75.00%
  12p0 82.81%
  12p1 82.81%
  16p0 82.03%
  16p1 73.44%
  min 72.66%

Expander 8/12/16, 256 games/row, seed 90000:
  --strategy-plan-worker-min-grid-size 12
  --strategy-plan-worker-max-grid-size 16

  8p0 72.27%
  8p1 73.05%
  12p0 80.08%
  12p1 80.08%
  16p0 77.73%
  16p1 76.17%
  min 72.27%

same 256 seed base:
  8p0 72.27%
  8p1 73.05%
  12p0 80.08%
  12p1 79.30%
  16p0 77.34%
  16p1 75.00%
  min 72.27%
```

Conclusion:

```text
The protect Worker/gate learns useful 12/16 replacements, but still perturbs
8x8 enough to lose the six-row minimum unless a min-grid guard disables it on
8x8. With min=12,max=16, 8x8 exactly returns to base and the larger rows gain
small same-seed improvements, but the overall minimum is still base-limited by
8p0.

This is not a promotion. It is evidence that Plan-Worker inference needs
explicit domain guards and better outcome/Q labels. The next useful step is
not another static threshold sweep; collect high-gap midgame decisive rows and
train source/target outcome-Q or trajectory imitation so the main U-Net policy
learns the decisive behavior directly.
```

## 2026-06-20 - Training-time row filters and policy-head decisive imitation

Added training-time row filters to `adaptive_strategy_supervised.py`:

```text
--min-row-turn
--max-row-turn
--require-contact
--min-visible-enemy-cells
--min-visible-enemy-density
--require-outcome-win
--require-outcome-draw
--require-outcome-nonwin
--require-search-best-win
--require-finish-within-250
--require-win-or-finish-within-250
--min-search-score-gap
```

The filters apply before `--max-samples-per-shard`, so broad saved shards can
be reused for focused midgame/contact/high-gap probes without writing derived
NPZ files. The loader keeps its default dict return for tests and returns
filter stats only when the CLI asks for them.

Policy-head decisive imitation probe:

```text
run:
  runs/adaptive-midgame-decisive-policyhead-v0/

init:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/

data:
  runs/adaptive-midgame-contact-searchwin-fixed-v5-v1/
  runs/adaptive-midgame-draw-fixed-v5-v0/
  runs/adaptive-midgame-contact-searchwin-expander-12-16-v0/
  runs/adaptive-midgame-contact-searchwin-expander-safev3-v0/

row filters:
  time >= 80
  contact
  visible_enemy_cells >= 1
  search_gap >= 0.25

sampling:
  max_samples_per_shard = 2500
  balance_strata = size-seat

training:
  update_scope = policy-heads
  policy_kl = 3.0
  action_ce = 0.20
  action_ce_weight_mode = search-best-win
  action_ce_path_contains:
    adaptive-midgame-contact-searchwin-fixed-v5-v1
    adaptive-midgame-contact-searchwin-expander
  finish = 0.50, balanced, multi-horizon
  outcome = 0.40, balanced
  belief = 0.25
  intent = 0.20
  lr = 2e-5
  epochs = 8
```

Training result:

```text
samples:
  kept 91,725 / 91,725 rows
  sampled 62,227 rows
  balanced to 37,242 rows

epoch 1 -> 8:
  loss     1.5786 -> 1.5184
  intent   56.8%  -> 75.9%
  finish   62.6%  -> 64.4%
  belief   0.0642 -> 0.0596
  outcome  37.2%  -> 35.8%
  KL       0.0195 -> 0.0171
  ActCE    2.7742 -> 2.7816
  ActAcc   29.1%  -> 29.0%
  ActW     0.241
```

Gameplay:

```text
fixed-v5 max250, 128 games/seat, seed 86640:
  base v3:
    p0 12.50%
    p1 10.94%
    min 10.94%

  policyhead-v0:
    p0 13.28%
    p1  8.59%
    min  8.59%

Expander 8/12/16, 128 games/row, seed 86680:
  base v3:
    8p0 75.78%
    8p1 77.34%
    12p0 82.81%
    12p1 82.81%
    16p0 82.03%
    16p1 77.34%
    min 75.78%

  policyhead-v0:
    8p0 74.22%
    8p1 75.78%
    12p0 85.16%
    12p1 82.03%
    16p0 77.34%
    16p1 75.78%
    min 74.22%
```

Conclusion:

```text
The training-time filters work and make the probe reproducible, but freezing
the trunk and updating only the primitive policy head does not solve the
midgame decisive imitation problem. It learns auxiliary labels while barely
moving action CE, then loses fixed-v5 p1 and Expander 8p0/16p0.

Do not promote `adaptive-midgame-decisive-policyhead-v0`. This rules out the
safe-looking policy-head-only version of midgame decisive imitation. The next
useful direction is not a CE/LR sweep: use high-gap decisive rows to train
source/target outcome-Q or a target-conditioned executor with explicit
replacement outcome labels, while keeping the safe v3 policy as the active base.
```

## 2026-06-20 High-Gap Plan-Q Source/Target Gate Probe

Added `adaptive_plan_q_supervised.py --strategy-finish-outputs` and
`--init-strategy-finish-outputs` so Plan-Q supervision can load multi-horizon
strategy checkpoints such as safe v3:

```text
base:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/

dataset:
  runs/adaptive-plan-q-model-worker-highgap-mid100-v0/*.npz

source/target Q output:
  runs/adaptive-plan-q-source-target-highgap-mid100-v0/

gate output:
  runs/adaptive-command-gate-highgap-mid100-v0/
```

The source/target Q trainer used frozen strategy heads only:

```text
source_q_rank_weight = 0.5
target_q_rank_weight = 1.0
plan_pair_rank_weight = 1.0
q_target_outcome_weight = 0.75
q_rank_temperature = 0.05
gap_weighting = true
epochs = 96
```

Final offline metrics stayed weak:

```text
source Q rank accuracy: 23.7%
target Q rank accuracy: 25.8%
pair top1 accuracy:      7.0%
pair score corr:        +0.021
mean plan gap:           0.6197
```

The command gate was trained on explicit replacement labels from the same
high-gap Plan-Q shards:

```text
examples: 23753
positive: 8.42%
hidden_dim: 64
source x target candidates: 4 x 4
include_noncomparable_negatives: true

epoch 80:
  loss 0.5821
  weighted accuracy 69.2%
  P+ 0.589
  P- 0.409
```

Gameplay against fixed v5 sample, max250, 128 games/seat, seed 86640:

```text
feature model, gate off:
  p0 15/49/64, win 11.72%, draw 50.00%
  p1 14/50/64, win 10.94%, draw 50.00%
  min 10.94%

command gate threshold 0.5, 4x4 candidates:
  p0 0/65/63, win 0.00%, draw 49.22%
  p1 0/53/75, win 0.00%, draw 58.59%
  min 0.00%
```

Conclusion:

```text
Do not promote `adaptive-plan-q-source-target-highgap-mid100-v0` or
`adaptive-command-gate-highgap-mid100-v0`.

This closes the current inference-time source/target replacement route. The
high-gap shard has real outcome contrast, and the gate can fit some offline
labels, but the learned source/target proposal is too weak and replacement
control collapses fixed-v5 gameplay. Do not sweep gate thresholds here. The next
useful route should train the U-Net main policy/representation from midgame
decisive trajectories or train a plan-conditioned executor only after command
quality is independently stronger.
```

## 2026-06-20 Midgame Decisive Representation Probe v0

After the command-gate collapse, tested the fastest direct alternative: update
the U-Net main representation from midgame/contact/high-gap decisive rows while
keeping the safe v3 policy close with a strong KL anchor.

```text
run:
  runs/adaptive-midgame-decisive-repr-v0/

init:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/

data:
  runs/adaptive-midgame-contact-searchwin-fixed-v5-v1/
  runs/adaptive-midgame-draw-fixed-v5-v0/
  runs/adaptive-midgame-contact-searchwin-expander-12-16-v0/
  runs/adaptive-midgame-contact-searchwin-expander-safev3-v0/

filters:
  turn >= 80
  contact
  visible_enemy_cells >= 1
  search_gap >= 0.25

sampling:
  max_samples_per_shard = 2000
  balance_strata = size-seat-domain
  samples = 8,712 after balancing

training:
  update_scope = all
  policy_kl = 10.0
  action_ce = 0.05
  action_ce_weight_mode = search-best-win
  finish = 0.50, balanced, multi-horizon
  outcome = 0.40, balanced, label_source=search-best
  belief = 0.25
  intent = 0.20
  lr = 3e-6
  epochs = 4
```

Training moved intent/outcome but not finish:

```text
epoch 1 -> 4:
  loss     1.3018 -> 1.2199
  intent   52.1%  -> 59.5%
  finish   46.5%  -> 46.6%
  outcome  48.2%  -> 52.9%
  belief   0.0740 -> 0.0733
  KL       0.0143 -> 0.0107
  ActCE    2.7298 -> 2.7974
  ActAcc   29.3%  -> 29.2%
```

Gameplay:

```text
fixed-v5 max250, 128 games/seat, seed 86640:
  base v3 / feature-control:
    p0 11.72%
    p1 10.94%
    min 10.94%

  repr-v0:
    p0  7.81%
    p1  9.38%
    min  7.81%

Expander 8/12/16, 64 games/row, seed 86680:
  repr-v0:
    8p0  71.88%
    8p1  82.81%
    12p0 75.00%
    12p1 85.94%
    16p0 81.25%
    16p1 73.44%
    min 71.88%
```

Conclusion:

```text
Do not promote `adaptive-midgame-decisive-repr-v0`.

This result rules out the conservative low-LR all-trunk version of midgame
decisive imitation on the current mixed filtered data. Even with low action CE
and high KL, the model does not improve finishability and loses fixed-v5 p0.
The useful next step is data-quality, not a LR/epoch sweep: collect or isolate
true search-controlled winning windows with stronger finish labels, especially
states where the base draws but the teacher/search wins, then train the main
policy on that contrast.
```

## 2026-06-20 Draw/Search-Win Contrast Filters and Probe

Added two composable row filters to `adaptive_strategy_supervised.py`:

```text
--require-outcome-draw
--require-outcome-nonwin
```

They use known trajectory outcome labels and can be combined with
`--require-search-best-win` to isolate states where the recorded policy failed
to win, but rollout-search found a winning continuation. A read-only shard count
after the standard high-gap/contact filters found:

```text
fixed-v5 v1:
  draw + search_best_win:    1,331
  nonwin + search_best_win:  1,819

fixed-v5 draw v0:
  draw + search_best_win:      337

Expander 12/16:
  draw + search_best_win:      430
  nonwin + search_best_win:    628

Expander safev3:
  draw + search_best_win:      136
  nonwin + search_best_win:    606
```

Focused representation probe:

```text
run:
  runs/adaptive-midgame-drawsearch-repr-v0/

filters:
  turn >= 80
  contact
  visible_enemy_cells >= 1
  search_gap >= 0.25
  outcome = draw
  search_best = win

training:
  update_scope = all
  policy_kl = 10.0
  action_ce = 0.10
  action_ce_weight_mode = search-best-win
  finish = 0.50, balanced, multi-horizon
  outcome = 0.40, balanced, label_source=search-best
  belief = 0.25
  intent = 0.20
  lr = 2e-6
  epochs = 8
```

Training data and metrics:

```text
row filters:
  kept 2,234 / 91,725 rows
  sampled 2,234 rows
  size-seat-domain balance -> 312 samples

epoch 1 -> 8:
  loss     2.1317 -> 2.0599
  intent   78.9%  -> 79.3%
  finish   42.3%  -> 43.5%
  outcome   4.3%  ->  8.2%
  KL       0.0186 -> 0.0174
  ActCE    3.2210 -> 3.3478
  ActAcc   26.2%  -> 25.8%
```

Gameplay against fixed v5 sample, max250, 128 games/seat, seed 86640:

```text
base v3 / feature-control:
  p0 11.72%
  p1 10.94%
  min 10.94%

drawsearch-repr-v0:
  p0 11.72%
  p1  9.38%
  min  9.38%
```

Conclusion:

```text
Do not promote `adaptive-midgame-drawsearch-repr-v0`.

The new filters are useful and verified, but this particular training recipe is
data-starved after size-seat-domain balancing. The model barely learns the
search-win outcome label and loses fixed-v5 p1. Do not sweep lr/epochs here.
Next useful experiment: either collect more draw/search-win rows, train without
domain balancing but with explicit Expander KL/protection rows, or use these
contrast rows as a small weighted component inside a larger decisive trajectory
dataset.
```

### Draw/Search-Win Oversample Probe

Added oversampling balance modes:

```text
--balance-strata size-seat-oversample
--balance-strata size-seat-domain-oversample
```

The existing `size-seat` and `size-seat-domain` modes still downsample to the
smallest stratum. The new modes repeat smaller strata up to the largest stratum
count, which keeps strict contrast filters from throwing away most rows.

Oversampled representation probe:

```text
run:
  runs/adaptive-midgame-drawsearch-oversample-repr-v0/

same filters and loss as drawsearch-repr-v0, except:
  balance_strata = size-seat-domain-oversample
  seed = 90500
```

Training data and metrics:

```text
row filters:
  kept 2,234 / 91,725 rows
  sampled 2,234 rows
  size-seat-domain-oversample -> 5,520 samples

epoch 1 -> 8:
  loss     2.0589 -> 1.2267
  intent   77.7%  -> 48.5%
  finish   44.0%  -> 72.2%
  outcome   8.8%  -> 77.7%
  belief   0.0664 -> 0.0648
  KL       0.0179 -> 0.0116
  ActCE    3.3438 -> 3.4977
  ActAcc   24.3%  -> 24.3%
```

Gameplay against fixed v5 sample, max250, 128 games/seat, seed 86640:

```text
base v3 / feature-control:
  p0 11.72%
  p1 10.94%
  min 10.94%

drawsearch-oversample-repr-v0:
  p0  8.59%
  p1  9.38%
  min  8.59%
```

Conclusion:

```text
The oversampling mode is useful: it fixes the data-retention problem and lets
the finish/outcome heads learn the intended all-search-win contrast. But using
those rows to update the full U-Net policy still hurts fixed-v5 gameplay.

Do not promote `adaptive-midgame-drawsearch-oversample-repr-v0`. Do not sweep
lr/epochs on this same objective. The next useful use of these contrast rows is
as auxiliary/gating data with policy logits frozen, or as a small weighted slice
inside a much larger successful-trajectory policy update rather than as the main
policy-imitation objective.
```

## 2026-06-20 Policy Adapter Feature-Gate Split

The draw/search-win contrast model learned finish/outcome labels, but its policy
was harmful. Added an adapter-gate split so the harmful policy does not have to
be used:

```text
adaptive_policy_adapter_gate_supervised.py:
  --feature-model-path

evaluate_adaptive_policy.py:
  --policy-adapter-feature-model-path
```

The base policy and adapter policy remain separate:

```text
base policy:
  safe v3

adapter delta:
  adaptive-midgame-contact-searchwin-safev3-policyhead-p0mix-v0

feature model:
  adaptive-midgame-drawsearch-oversample-repr-v0
```

Adapter-gate features now append outcome probabilities:

```text
old 12-feature gate:
  finish probability only

new 14-feature gate:
  finish probability
  draw probability
  win probability
```

Evaluator compatibility is preserved by slicing the feature vector to the
feature count saved in the gate sidecar.

Hard finish gate, fixed-v5 max250, 128 games/seat, seed 86640:

```text
base v3:
  p0 11.72%
  p1 10.94%
  min 10.94%

p0mix adapter, adapter's own finish head threshold 0.5:
  p0 12.50%
  p1 10.16%
  min 10.16%

p0mix adapter, draw/search feature model threshold 0.5:
  p0 14.84%
  p1  9.38%
  min  9.38%
```

Learned gate with the 14-feature split:

```text
run:
  runs/adaptive-policy-adapter-gate-feature-v0/

datasets:
  fixed-v5 v1
  Expander safev3

labels:
  positive path contains fixed-v5-v1
  adapter greedy top1 matches teacher action
  search_best_outcome == win
  unchanged actions kept as negatives

examples:
  rows 51,835
  changed 187
  teacher_match 61
  positives ~0.01%

final:
  P+ 0.040
  P- 0.002
```

Gameplay with learned gate threshold 0.5, fixed-v5 max250, 128 games/seat,
seed 86640:

```text
p0 12.50%
p1 10.94%
min 10.94%
```

Conclusion:

```text
Do not promote `adaptive-policy-adapter-gate-feature-v0`.

The feature-model split is useful infrastructure, but the p0mix adapter changes
too few greedy actions to train a meaningful replacement-outcome gate. The
contrast-calibrated finish/outcome features do not fix the sparse-positive
problem. Close this p0mix adapter-gate branch unless a future adapter produces
many more candidate changes with clear positive labels.
```

## 2026-06-20 Midgame Search-Win Trajectory Policy Probe

Goal:

```text
Test whether higher-quality midgame/search-controlled winning trajectories can
update the U-Net main policy directly, instead of acting as source/target or
Worker inference-time overrides.
```

Training data:

```text
winning trajectory:
  runs/adaptive-midgame-searchwin-trajectory-fixed-v5-safev3-v0/
  rows: 25,385
  outcome win: 25,385
  turn>=80: 18,688
  search_best win: 8,789

terminal search-win:
  runs/adaptive-midgame-terminal-searchwin-fixed-v5-safev3-v0/
  rows: 1,722
  search_best win: 1,722
  median turn: 145

draw contrast:
  runs/adaptive-midgame-draw-fixed-v5-v0/

Expander protection:
  runs/adaptive-midgame-contact-searchwin-expander-safev3-v0/
  rows: 23,361
  sizes: 8/12/16
```

Run:

```text
runs/adaptive-midgame-searchwin-trajectory-imitation-v0/

init:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/generals-adaptive-midgame-contact-searchwin-imitation-v3.eqx

loader filters:
  time>=80
  contact
  visible_enemy_cells>=1
  search_gap>=0.25

balancing:
  size-seat-oversample

loss:
  intent 0.20
  finish 0.50
  belief 0.25
  outcome 0.40
  policy KL 6.0
  action CE 0.12

action CE:
  trajectory wins only
  only searchwin-trajectory and terminal-searchwin paths

update:
  all U-Net weights
  lr 3e-6
  14 epochs
```

Training result:

```text
samples after balance: 64,998
action weight mean: 0.170
policy KL: 0.0016 -> 0.0031
intent acc: 62.4% -> 82.0%
finish acc: 66.6% -> 66.8%
outcome acc: 34.5% -> 48.2%
action acc: 31.6% -> 31.7%
```

Fixed-v5 max250, 128 games/seat, seed 90720:

```text
v0 direct policy:
  p0  7.81%
  p1 17.19%
  min 7.81%
```

Expander adaptive, 128 games/row, seed 90740:

```text
8p0  71.09%
8p1  68.75%
12p0 81.25%
12p1 81.25%
16p0 77.34%
16p1 73.44%
min  68.75%
```

Conclusion:

```text
Do not promote `adaptive-midgame-searchwin-trajectory-imitation-v0`.

Even with conservative KL, direct all-trunk imitation from midgame winning
trajectories causes broad Expander regression. The auxiliary heads learn, but
the main policy does not gain stable fixed-v5 strength. This is stronger
evidence that direct trajectory CE/KL insertion is not the right way to compress
search-controlled wins into the primitive policy.
```

Adapter-gate diagnostic using v0 as the adapter:

```text
run:
  runs/adaptive-policy-adapter-gate-searchwin-traj-v0/

base:
  safe v3

adapter:
  adaptive-midgame-searchwin-trajectory-imitation-v0

gate examples:
  rows 112,492
  changed actions 2,033
  teacher matches 577
  positives 0.13%

final gate:
  P+ 0.511
  P- 0.010
  Pmean 0.011
```

Fixed-v5 max250, 256 games/seat, seed 86640:

```text
safe v3 historical:
  min about 10.94%-11.33%

v0 adapter + learned gate threshold 0.5:
  p0 16.80%
  p1 10.55%
  min 10.55%

v0 adapter + learned gate threshold 0.1:
  p0 16.80%
  p1 10.55%
  min 10.55%
```

Interpretation:

```text
The stronger v0 adapter increases changed-action examples relative to p0mix, but
the positive rate is still only 0.13%. The learned gate is effectively too sparse
to produce a promotion-level improvement, and lower threshold does not change the
result. Close this direct-policy/delta branch for now.
```

Next technical direction:

```text
Stop direct all-trunk trajectory imitation and policy-adapter gating from this
data. Use the trajectory/search-win data for calibrated finish/value targets or
move to controlled PPO from safe v3, where actual rollout outcome supplies the
credit signal. If revisiting offline data, prefer Q/value/finish calibration or
advantage-labeled action selection over primitive CE on trajectory actions.
```

## 2026-06-20 Controlled Fixed-v5 PPO Probe

Motivation:

```text
Direct all-trunk offline imitation regressed Expander hard. Test whether actual
rollout outcome credit, with long rollout + top-advantage + EMA + KL anchor, can
move fixed-v5 strength without destroying the base distribution.
```

Implementation note:

```text
train_adaptive.py now has explicit output-template flags:
  --strategy-aux
  --strategy-spatial-aux
  --strategy-finish-outputs

The old init-only flags still describe the warm-start checkpoint:
  --init-strategy-aux
  --init-strategy-spatial-aux
  --init-strategy-finish-outputs

Passing only init flags reads and discards strategy aux heads. Passing both
output and init flags preserves strategy heads through PPO.
```

Run v0:

```text
runs/adaptive-ppo-fixed-v5-controlled-v0/

init:
  safe v3

opponent:
  fixed 8x8 v5 sampled policy

training:
  num_envs 64
  learner_player mixed
  rollout 256
  iterations 20
  epochs 1
  minibatch 2048
  reward terminal only
  terminal_reward_scale 1.0
  gamma 1.0
  gae_lambda 0.90
  top_advantage_fraction 0.25
  top_advantage_mode stratified
  ema_decay 0.999
  eval_ema
  teacher_kl_weight 0.2 to safe v3
  outcome_aux_weight 0.05
  lr 1e-6
```

Training:

```text
compile emitted a nonfatal 4.28GiB allocator warning
iter 1:  episodes 66, wins 8, draws 35
iter 10: episodes 79, wins 5, draws 40
iter 20: episodes 83, wins 8, draws 46
```

Evaluation v0:

```text
fixed-v5 max250, 256 games/seat, seed 86640:
  p0 16.02%
  p1 11.33%
  min 11.33%

Expander adaptive, 128 games/row, seed 86680:
  8p0  75.78%
  8p1  77.34%
  12p0 82.81%
  12p1 82.81%
  16p0 81.25%
  16p1 77.34%
  min 75.78%
```

Run v1 continuation:

```text
runs/adaptive-ppo-fixed-v5-controlled-v1/

init:
  v0

training:
  same settings
  iterations 60
```

Evaluation v1:

```text
fixed-v5 max250, 256 games/seat, seed 86640:
  p0 14.06%
  p1 11.33%
  min 11.33%

Expander adaptive, 128 games/row, seed 86680:
  8p0  75.78%
  8p1  77.34%
  12p0 81.25%
  12p1 78.91%
  16p0 78.12%
  16p1 75.00%
  min 75.00%
```

Conclusion:

```text
Controlled PPO is safer than offline all-trunk imitation: v0 preserved Expander
min at the safe-v3 level and matched/slightly improved fixed-v5 256-row min.
However, longer continuation did not improve fixed-v5 and started to erode
12/16 Expander rows. Do not promote v1.

Keep v0 as a controlled-PPO diagnostic checkpoint, not the active base. The next
PPO attempt should either:
  1. start again from safe v3 with strategy_aux preserved and a slightly larger
     fixed-v5 signal batch, or
  2. train against a mixed opponent curriculum that includes Expander protection
     in the rollout itself, not only via KL.
```

## 2026-06-20 Mixed-Opponent And Value Calibration Follow-up

Corrected schema note:

```text
safe v3 can load with current strategy heads if the checkpoint template includes
both outcome and strategy heads:
  --outcome-aux-weight / --outcome-head
  --init-outcome-head
  --strategy-aux
  --init-strategy-aux
  --strategy-finish-outputs 3

The earlier apparent strategy schema mismatch was caused by omitting the outcome
head, which shifted Equinox leaf order and made the outcome layer appear where
the strategy intent layer was expected.
```

Mixed-opponent PPO with full safe-v3 schema:

```text
run:
  runs/adaptive-ppo-mixed-opponent-safev3-full-v0/

init/teacher:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/

opponent curriculum:
  8x8 rows: fixed v5 with p=0.5, otherwise Expander fallback
  12/16 rows: Expander fallback

training:
  num_envs 32
  rollout 128
  iterations 30
  sparse terminal reward
  top_advantage_fraction 0.25 stratified
  EMA 0.999
  outcome_aux_weight 0.2
  strategy heads preserved
```

Results:

```text
fixed-v5 max250, 256 games/seat, seed 86640:
  p0 16.41%
  p1 11.33%
  min 11.33%

Expander adaptive, 128 games/row, seed 86680:
  8p0  74.22%
  8p1  76.56%
  12p0 83.59%
  12p1 84.38%
  16p0 77.34%
  16p1 78.12%
  min 74.22%
```

Value-calibration implementation:

```text
adaptive_strategy_supervised.py:
  added --value-target-weight

target:
  selected outcome label -> value target
  loss/draw/win -> -1/0/+1

with --value-loss hl-gauss:
  train PPO value logits with HL-Gauss CE

with --update-scope strategy-value-heads:
  keep trunk/policy/action logits frozen
  train strategy heads, outcome head, value_linear1, and value heads
```

Value calibration run:

```text
run:
  runs/adaptive-valuecal-trajectory-v0/

data:
  fixed-v5 searchwin trajectory
  terminal searchwin fixed-v5
  fixed-v5 draw contrast
  fixed-v5 contact/searchwin
  Expander safe-v3 protection

filters:
  turn >= 80
  contact
  visible_enemy_cells >= 1

sample cap:
  max_samples_per_shard 2048
  max_samples 50000

training:
  update_scope strategy-value-heads
  no action CE
  label_source trajectory
  finish 0.50
  outcome 0.25
  value 0.50
  belief 0.20
  intent 0.10
  lr 5e-5
  epochs 8
```

Training signal:

```text
samples: 45,866
value loss: 5.2276 -> 2.9189
value MAE:  0.673  -> 0.609
outcome acc: 41.8% -> 44.6%
finish acc:  59.4% -> 60.0%
intent acc:  55.6% -> 58.4%
```

Value-calibrated mixed-opponent PPO:

```text
run:
  runs/adaptive-ppo-valuecal-mixed-v0/

init/teacher:
  runs/adaptive-valuecal-trajectory-v0/

same mixed-opponent PPO recipe as safev3-full-v0
```

Results:

```text
fixed-v5 max250, 256 games/seat, seed 86640:
  p0 15.23%
  p1 11.33%
  min 11.33%

Expander adaptive, 128 games/row, seed 86680:
  8p0  74.22%
  8p1  76.56%
  12p0 82.03%
  12p1 82.03%
  16p0 84.38%
  16p1 76.56%
  min 74.22%
```

Conclusion:

```text
Preserving full strategy/outcome heads and calibrating value from trajectory
labels did not move the fixed-v5 short-time gate beyond 11.33% min. The value
loss learns, but the PPO update still does not discover decisive execution.

Do not promote:
  adaptive-ppo-mixed-opponent-safev3-full-v0
  adaptive-valuecal-trajectory-v0
  adaptive-ppo-valuecal-mixed-v0

Next direction should not be longer plain PPO or value-only pretraining. The
remaining gap is execution/control: source-target outcome-Q, plan-conditioned
Worker with stronger positive gates, or advantage-labeled action selection from
counterfactual plans.
```

### 2026-06-20: Fixed-v5 Short-Gate Follow-up

This round tested whether the latest safe-v3 checkpoint can be pushed through
the fixed-v5 `max250` gate by more direct midgame imitation or 8x8-only
best-response PPO. All runs used GPU and wrote artifacts under ignored `runs/`.

Direct midgame decisive trajectory imitation from safe-v3:

```text
run:
  runs/adaptive-midgame-decisive-trajectory-imitation-v1/

init:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/

data:
  adaptive-midgame-searchwin-trajectory-fixed-v5-safev3-v0
  adaptive-midgame-contact-searchwin-fixed-v5-safev3-v0
  adaptive-midgame-contact-searchwin-expander-safev3-v0
  adaptive-midgame-draw-fixed-v5-v0

filters:
  turn >= 80
  contact
  visible_enemy_cells >= 1

training:
  update_scope all
  policy_kl 1.0
  action_ce 0.30, wins only
  finish 0.50, outcome 0.40, belief 0.25, intent 0.20
```

Results:

```text
fixed-v5 max250, 128 games/seat, seed 91340:
  min 10.16%
  max draw 51.56%

Expander adaptive, 128 games/row, seed 91360:
  min 65.62%
```

Conclusion: full-trunk trajectory imitation moved the model but damaged the
adaptive Expander policy, especially 16x16 player 1. Do not promote.

Policy-head-only variants:

```text
runs/adaptive-midgame-decisive-trajectory-policyhead-v2/
  update_scope policy-heads
  policy_kl 4.0
  action_ce 0.10, wins only

fixed-v5 max250:
  128 games/seat, seed 91400: min 11.72%
  256 games/seat, seed 91440: min 8.20%

Expander adaptive:
  128 games/row, seed 91420: min 75.00%

runs/adaptive-midgame-searchbest-action-policyhead-v3/
  update_scope policy-heads
  policy_kl 4.0
  action_ce 0.20, search_best_outcome=win only

fixed-v5 max250:
  128 games/seat, seed 91480: min 12.50%
  256 games/seat, seed 91500: min 7.03%
```

Conclusion: policy-head-only updates can preserve Expander, but the fixed-v5
gains are 128-row false positives. The current search-win shards already store
search-best actions as `teacher_action_index` in nearly all rows
(`96.7%` to `99.98%` match depending on shard), so the failure is not caused by
using raw trajectory actions.

High-gap Plan-Q Worker probe:

```text
run:
  runs/adaptive-plan-worker-highgap-mid100-v0/

data:
  runs/adaptive-plan-q-model-worker-highgap-mid100-v0/*.npz

training:
  selection best
  target-conditioned Worker
  20 epochs

offline:
  final action accuracy 10.0%
  final useful accuracy 10.0%

eval with:
  adaptive-plan-q-source-target-highgap-mid100-v0
  strategy_plan_worker_rerank_scale 0.02

fixed-v5 max250, 128 games/seat, seed 91540:
  min 6.25%
  max draw 54.69%
```

Conclusion: the high-gap Worker labels are still too weak/noisy for inference
replacement. Do not continue this Worker variant.

8x8-only fixed-v5 PPO best-response probes:

```text
runs/adaptive-ppo-fixed-v5-8only-br-v0/
  grid_sizes 8
  fixed-v5 opponent p=1.0
  mixed learner seats
  terminal reward 20
  rollout 256
  top_advantage stratified 0.25
  EMA 0.999 saved as eval model

fixed-v5 max250, 128 games/seat, seed 91620:
  min 8.59%

Expander adaptive, 128 games/row, seed 91640:
  min 67.19%

runs/adaptive-ppo-fixed-v5-8only-br-last-v1/
  same, but save last iterate and lr 5e-6

fixed-v5 max250, 128 games/seat, seed 91700:
  min 8.59%
  max draw 61.72%

Expander adaptive, 128 games/row, seed 91720:
  min 73.44%

runs/adaptive-ppo-fixed-v5-8only-drawpenalty-v0/
  no top-advantage filtering
  truncation_reward_scale 5.0

fixed-v5 max250, 128 games/seat, seed 91780:
  min 9.38%
  max draw 57.03%
```

Conclusion:

```text
Short 8x8 fixed-v5 best-response PPO does not currently solve the gate.
The EMA save path was not the root cause: saving last iterate did not help.
Making timeout draws negative also did not reduce draw enough or improve win rate.

Stop:
  direct midgame action imitation
  policy-head-only search-best CE
  short 8-only fixed-v5 PPO
  simple draw-penalty PPO
  high-gap Worker rerank from current labels

Next useful direction:
  build cleaner counterfactual execution labels, not more primitive CE.
  Use Plan-Q/replacement data to train an advantage-labeled action head or
  command-gated executor where positives are "changed action later wins" rather
  than "teacher/search picked this single action".
```

### 2026-06-20: Fixed-v5 Rollout-Search Strategy Teacher

This round added a true fixed-v5 rollout-search data source for strategy shards.
The previous fixed-v5 search-win shards used safe-v3 as the search prior; this
new path lets the dataset collector query a fixed 8x8 `PolicyValueNetwork`
teacher, score its top-k actions with short rollouts, and store the best action
as an adaptive padded action index.

Implementation:

```text
adaptive_strategy_dataset.py:
  teacher_kind += fixed-search
  fixed_rollout_search_candidates(...)

fixed-search behavior:
  crop padded observation to fixed teacher grid
  run fixed_policy_teacher_logits
  map top-k fixed action indices back to adaptive action indices
  execute candidates in the padded GameState
  save search_candidate_indices / scores / outcomes / gap

validation:
  py_compile passed
  smoke shard wrote 32 rows under runs/adaptive-fixed-search-dataset-smoke/
```

A1 fixed-v5 rollout-search collection:

```text
run:
  runs/adaptive-fixed-v5-searchwin-a1-v1/

teacher:
  /home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v5.eqx

opponent:
  same raw fixed-v5 policy

collection:
  num_envs 32
  shards 16
  steps/shard 256
  grid_sizes 8
  top_k 4
  rollout_steps 8
  rollouts/action 1
  turn >= 80
  contact
  visible_enemy_cells >= 1
  search_best_outcome == win
  search_score_gap >= 500

rows:
  total 2168
  p0 1263
  p1 905
  sample wins 1557
  sample draws 377
  mean search gap 1058.7
  turn range 80-250
```

Policy-head-only probes from safe-v3:

```text
runs/adaptive-fixed-v5-search-a1-policyhead-v0/
  init: adaptive-midgame-contact-searchwin-imitation-v3
  update_scope: policy-heads
  policy_kl: 1.0
  action_ce: 0.30

fixed-v5 max250, 128 games/seat:
  p0 11.72%
  p1 11.72%
  min 11.72%
  draw 50.78% / 45.31%

Expander adaptive, 128 games/row:
  min 74.22%

runs/adaptive-fixed-v5-search-a1-policyhead-v1/
  same init and data
  policy_kl: 0.30
  action_ce: 0.60

fixed-v5 max250, 128 games/seat:
  min 9.38%

Expander adaptive, 128 games/row:
  min 75.00%
```

Base-KL relabel probe:

```text
data:
  runs/adaptive-fixed-v5-searchwin-a1-v1-basekl/

change:
  replaced fixed-v5 teacher_logits with safe-v3 logits
  kept fixed-v5 rollout-search best action labels

run:
  runs/adaptive-fixed-v5-search-a1-basekl-policyhead-v2/

training:
  update_scope: policy-heads
  policy_kl: 1.0
  action_ce: 0.30

fixed-v5 max250, 128 games/seat:
  p0 12.50%
  p1 12.50%
  min 12.50%
  draw 50.00% / 57.81%

Expander adaptive, 128 games/row:
  min 70.31%
```

Attempting to start from `adaptive-fixed-v5-imitation-v5` iter 30 failed before
training because the legacy fixed-v5 imitation checkpoint stores the older value
head shape (`size_value_linear1[0].weight` on disk is `(128,64)` while the
current loader expects `(64,128)`). Revisit only if a legacy checkpoint adapter
is worth the time.

Conclusion:

```text
fixed-search is a useful data source.
It creates direct v5 + rollout-search labels without using safe-v3 as the
search prior, and the collector is now available for future shards.

Do not promote any A1 policy-head checkpoint.
Best fixed-v5 smoke was v2 at 12.50% min, but it regressed Expander to 70.31%.
The result confirms the pattern from earlier runs: primitive action CE can move
the head but does not reliably transfer rollout-search strength into gameplay.

Next useful use of fixed-search data:
  finish/value/command labels
  command-gated executor positives
  advantage-labeled changed actions
  legacy-loader experiment only if we need to fine-tune the fixed-v5 imitation
  checkpoint directly
```

### 2026-06-20: Fixed-Search Mixed Labels and Value/Q Probes

The previous `search-best` label mode could not mix fixed-search positives with
ordinary draw/loss contrast shards: rows without `search_best_outcome` received
zero finish/outcome weight. Added:

```text
adaptive_strategy_supervised.py:
  --label-source search-best-or-trajectory

behavior:
  if search_best_outcome is present:
    finish/outcome/value target comes from search-best
  else:
    finish/outcome/value target falls back to trajectory labels
```

The loader test now verifies the mixed behavior on a synthetic shard.

Fixed-search mixed-label value calibration:

```text
run:
  runs/adaptive-fixed-search-mixedlabel-valuecal-v0/

init:
  adaptive-midgame-contact-searchwin-imitation-v3

data:
  runs/adaptive-fixed-v5-searchwin-a1-v1/*.npz
  runs/adaptive-midgame-draw-fixed-v5-v0/*.npz

label_source:
  search-best-or-trajectory

balance:
  size-seat-domain

samples:
  kept 5921/5921
  balanced 5606

training:
  update_scope strategy-value-heads
  action CE 0
  finish 0.60, outcome 0.40, value 0.50
  belief 0.20, intent 0.10
```

Training signal:

```text
loss:      3.3112 -> 2.5039
finish:    50.9% -> 56.1%
outcome:   55.4% -> 62.5%
value CE:  4.8389 -> 3.3793
value MAE: 0.424 -> 0.431
```

Short controlled PPO from the value-calibrated checkpoint:

```text
failed large run:
  runs/adaptive-ppo-fixedsearch-valuecal-mixed-v0/
  192 envs, minibatch 4096
  stopped during compile after repeated GPU allocator OOM warnings

completed run:
  runs/adaptive-ppo-fixedsearch-valuecal-mixed-v1/
  96 envs
  rollout 128
  iterations 20
  minibatch 1024
  mixed fixed-v5/Expander opponent p=0.5 on 8x rows
  terminal reward 20
  top advantage 0.25 stratified
  EMA 0.999
  teacher KL 0.05 to valuecal checkpoint
```

Results:

```text
fixed-v5 max250, 128 games/seat, seed 93300:
  p0 10.94%
  p1 10.94%
  min 10.94%
  draw 56.25% / 50.00%

Expander adaptive, 128 games/row, seed 93320:
  8p0  74.22%
  8p1  82.81%
  12p0 83.59%
  12p1 81.25%
  16p0 79.69%
  16p1 76.56%
  min 74.22%
```

Fixed-search A1 Q-value probe:

```text
run:
  runs/adaptive-fixed-search-a1-qvalue-v0/

data:
  runs/adaptive-fixed-v5-searchwin-a1-v1/*.npz

training:
  update_scope strategy-heads
  search_q_value_weight 1.0
  search_q_rank_weight 0.5
  search_q_outcome_score_weight 0.1
  no policy update

offline:
  search-Q value loss 13.0544 -> 10.3694
  search-Q value accuracy stayed weak at 26.5%
```

Single conservative replacement gate:

```text
eval:
  --strategy-q-replace-threshold 0.5
  --strategy-q-replace-policy-margin 4.0

fixed-v5 max250, 128 games/seat, seed 93440:
  p0 9.38%
  p1 9.38%
  min 9.38%
```

Conclusion:

```text
The mixed label mode is useful infrastructure, but this A1 fixed-search data is
still too small/noisy to become a policy improvement through value calibration,
short PPO, or current strategy-Q replacement.

Stop:
  fixed-search A1 valuecal -> short PPO
  fixed-search A1 strategy-Q replacement
  larger minibatch/env PPO on this path unless memory is redesigned

Next useful step should change the data/control target, not the weights:
  collect changed-action counterfactual rows where base action draws/loses and
  fixed-search replacement later wins, then train a gate/executor on that
  changed-action outcome rather than all best actions.
```

### 2026-06-20: Changed-Action Plan-Q Gate Probe

The next probe targeted inference-time command replacement using model-generated
source/target candidates instead of heuristic candidates. The Plan-Q collector
first needed an explicit multi-horizon finish template:

```text
adaptive_plan_q_dataset.py:
  added --strategy-finish-outputs

reason:
  multi-horizon strategy checkpoints store three finish logits, while the
  collector previously defaulted to the older binary finish head.
```

`adaptive-midgame-contact-searchwin-imitation-v3` remains a safe policy base,
but it is not a valid source/target command candidate model because it was saved
without `strategy_spatial_aux`. The probe therefore used the policy-preserving
spatial checkpoint:

```text
base feature/policy model:
  runs/adaptive-plan-q-source-target-highgap-mid100-v0/

collector:
  candidate_source: model-worker
  candidate_target: model
  grid: 8x8 padded to 16
  warmup: 80
  truncation: 250
  plan rollout: 24
  worker steps: 3
  opponent: fixed v5 sample
```

GPU collection summary:

```text
output:
  runs/adaptive-plan-q-fixedv5-changed-mid-v1/

shard 0000:
  samples 448/512
  mean_gap 0.1119
  best_win 11.4%
  best_draw 88.6%

shard 0001:
  samples 451/512
  mean_gap 0.0638
  best_win 2.9%
  best_draw 96.9%

shard 0002:
  samples 459/512
  mean_gap 0.0668
  best_win 5.9%
  best_draw 93.9%

shard 0003:
  samples 448/512
  mean_gap 0.0593
  best_win 3.3%
  best_draw 96.7%
```

Command gate training fit the offline label distribution but the positive rate
was low:

```text
run:
  runs/adaptive-command-gate-fixedv5-changed-mid-v1/

examples:
  7805

positive:
  4.09%

epoch 60:
  loss 0.4671
  accuracy 78.9%
  P+ 0.678
  P- 0.309
  Pmean 0.325
```

Fixed-v5 max250 evaluation at the same seed:

```text
gate off:
  p0 14/50/64, win 10.94%, draw 50.00%
  p1 14/38/76, win 10.94%, draw 59.38%
  min 10.94%

gate threshold 0.6:
  p0 0/2/126, win 0.00%, draw 98.44%
  p1 0/2/126, win 0.00%, draw 98.44%
  min 0.00%
```

Conclusion:

```text
Do not promote or sweep this command-gate route.
The gate can identify comparable changed-action positives offline, but current
source/target proposal plus worker replacement stalls into almost pure draw
against fixed v5.

Next useful direction:
  train a plan-conditioned executor or command policy with explicit finish
  pressure, or collect longer executed-command trajectories where the
  replacement itself reaches terminal wins. The current single-step command gate
  is not enough.
```

### 2026-06-20: Executed-Prefix Plan-Worker Probe

The command-gate failure showed that choosing a one-step replacement is not
enough. Added an optional executed-prefix path:

```text
adaptive_plan_q_dataset.py:
  --save-worker-prefix-steps N

behavior:
  after scoring source-target plans, select the best plan and save the first N
  target-conditioned Worker observations, masks, action labels, source/target
  command cells, plan outcome, and plan Q.

adaptive_plan_worker_supervised.py:
  --dataset-format plan-q-prefix

behavior:
  flatten worker_prefix_* arrays into Worker examples.
  --require-outcome-win filters by selected plan outcome.
```

GPU smoke:

```text
run:
  runs/adaptive-plan-q-prefix-smoke-v0/

settings:
  4 envs
  2 rollout steps
  2x2 plans
  plan_rollout_steps 8
  plan_worker_steps 4
  save_worker_prefix_steps 4

result:
  shard saved with worker_prefix_obs shape (8, 4, 35, 16, 16)
  32 valid prefix steps
  trainer smoke loaded plan-q-prefix and saved a small Worker
```

Winning oracle-prefix v0:

```text
collector:
  runs/adaptive-plan-q-prefix-oracle-win-v0/
  heuristic source/target candidates
  require_best_plan_win
  warmup 80
  plan_rollout_steps 48
  plan_worker_steps 12
  save_worker_prefix_steps 12

data:
  8 winning command states
  96 valid non-pass prefix steps
  mean plan_q_gap 0.842

worker:
  runs/adaptive-plan-worker-prefix-oracle-win-v0/
  small U-Net, 32/48/64/32
  final action/useful accuracy 96.9%
```

Fixed-v5 max250, 128 games/seat:

```text
seed 93900 baseline:
  p0 4.69%
  p1 12.50%
  min 4.69%

seed 93900 prefix worker v0, scale 0.02, command belief-main-stack:
  p0 11.72%
  p1 9.38%
  min 9.38%

seed 93700 historical baseline:
  p0 10.94%
  p1 10.94%
  min 10.94%

seed 93700 prefix worker v0, scale 0.02, command belief-main-stack:
  p0 12.50%
  p1 10.94%
  min 10.94%
```

Expander smoke, 64 games/row, max500, seed 93940:

```text
baseline:
  8p0 60.94%
  8p1 62.50%
  12p0 73.44%
  12p1 70.31%
  16p0 51.56%
  16p1 48.44%
  min 48.44%

prefix worker v0 with --strategy-plan-worker-max-grid-size 8:
  8p0 73.44%
  8p1 62.50%
  12p0 73.44%
  12p1 70.31%
  16p0 51.56%
  16p1 48.44%
  min 48.44%
```

This is a weak but real positive signal: multi-step executed-prefix labels can
improve an 8x8 row without changing larger rows when range-gated.

Winning oracle-prefix v1 scaled the same data source:

```text
collector:
  runs/adaptive-plan-q-prefix-oracle-win-v1/
  16 envs
  16 rollout steps
  8 shards

data:
  186 winning command states
  2195 valid non-pass prefix steps
  mean plan_q_gap 0.746
  min plan_q_gap 0.433

worker:
  runs/adaptive-plan-worker-prefix-oracle-win-v1/
  final action/useful accuracy 98.6%
  source accuracy 98.6%
  direction accuracy 99.6%
```

But v1 did not improve gameplay:

```text
fixed-v5 max250, 128 games/seat, seed 93900:
  p0 7.03%
  p1 11.72%
  min 7.03%

fixed-v5 max250, 128 games/seat, seed 93700:
  p0 9.38%
  p1 10.16%
  min 9.38%
```

Conclusion:

```text
Keep the executed-prefix infrastructure.
Do not continue expanding oracle-only prefix data.

The small v0 signal says the failure point is partly execution, but v1 shows
that fitting oracle heuristic commands too well creates command-distribution
mismatch against inference-time belief/main-stack commands.

Next useful variant:
  collect prefix data from inference-matched commands:
    candidate_source model-worker
    candidate_target model or belief-main-stack
    save executed prefixes only when the plan later wins or sharply reduces draw
  mix a small oracle slice as regularization rather than the whole Worker target.
```

### 2026-06-20: Inference-Matched Belief Prefix Probe

Added inference-matched candidate modes to the Plan-Q collector:

```text
adaptive_plan_q_dataset.py:
  --candidate-target belief
    use strategy enemy_general_logits as target logits

  --candidate-source main-stack
    use army mass and route distance to the top target, with no source-head prior

This matches:
  evaluate_adaptive_policy.py --strategy-plan-worker-command-source belief-main-stack
```

Parser guards:

```text
--candidate-source belief:
  rejected

--candidate-target model-worker/main-stack:
  rejected

--candidate-target belief:
  requires --strategy-aux
```

GPU smoke:

```text
runs/adaptive-plan-q-prefix-belief-smoke-v0/
  4 envs
  2 rollout steps
  2x2 plans
  plan_rollout_steps 8
  plan_worker_steps 4
  save_worker_prefix_steps 4

result:
  shard saved successfully
  samples 8/8
  best_win 0.0
  best_draw 1.0
```

Inference-matched winning prefix collection:

```text
run:
  runs/adaptive-plan-q-prefix-belief-win-v0/

settings:
  candidate_source main-stack
  candidate_target belief
  require_best_plan_win
  warmup 80
  plan_rollout_steps 48
  plan_worker_steps 12
  save_worker_prefix_steps 12
  fixed-v5 sample opponent

data:
  186 winning command states
  2176 valid non-pass prefix steps
  mean plan_q_gap 0.732
  min plan_q_gap 0.426
```

Worker training:

```text
run:
  runs/adaptive-plan-worker-prefix-belief-win-v0/

architecture:
  U-Net 32,48,64,32

epoch 60:
  action/useful accuracy 97.8%
  source accuracy 97.9%
  direction accuracy 98.2%
```

Fixed-v5 max250 evaluation:

```text
seed 93900 baseline:
  p0 4.69%
  p1 12.50%
  min 4.69%

seed 93900 belief-prefix Worker, scale 0.02:
  p0 7.81%
  p1 9.38%
  min 7.81%

seed 93700 historical baseline:
  p0 10.94%
  p1 10.94%
  min 10.94%

seed 93700 belief-prefix Worker, scale 0.02:
  p0 9.38%
  p1 9.38%
  min 9.38%
```

Conclusion:

```text
The inference-matched collector works and produces enough winning prefix data.
Direct Worker imitation/rerank still does not solve fixed-v5; it improves a weak
seed only partially and regresses the stronger seed.

Stop:
  more Worker-rerank scale sweeps
  larger pure prefix imitation datasets

Next useful route:
  use prefix data as supervision for main-policy/finish gating or train a
  command policy that decides when to enter/exits a plan, rather than forcing
  centered Worker logits into every state.
```

### 2026-06-20: Prefix Main-Policy Supervision

Implemented the first direct main-policy route for executed Plan-Q prefixes.

Collector change:

```text
adaptive_plan_q_dataset.py:
  --save-worker-prefix-steps now also saves worker_prefix_teacher_logits
```

This records the warm-start policy logits for every saved prefix state, so
prefix imitation can use a true per-state KL anchor instead of reusing the
pre-plan state's logits.

Trainer change:

```text
adaptive_strategy_supervised.py:
  added --dataset-format plan-q-prefix
```

The loader flattens:

```text
worker_prefix_obs
worker_prefix_legal_mask
worker_prefix_active
worker_prefix_teacher_logits
worker_prefix_action_index
worker_prefix_plan_outcome
worker_prefix_source_index / target_index
```

into the existing strategy-supervised schema. It drops pass labels by default,
uses saved prefix logits for `policy_kl`, and uses the executed prefix action
for small `action_ce`. `--require-outcome-win` on this format means the selected
best source-target plan won in the short counterfactual rollout.

GPU smoke:

```text
runs/adaptive-plan-q-prefix-logits-smoke-v0/
  8 base rows
  32 valid prefix steps
  worker_prefix_teacher_logits shape (8, 4, 2049)

runs/adaptive-prefix-policy-smoke-v0/
  loaded plan-q-prefix format
  12 non-pass samples
  KL starts at 0.0000 against saved logits
```

High-quality prefix collection:

```text
run:
  runs/adaptive-plan-q-prefix-logits-belief-win-v0/

settings:
  candidate_source main-stack
  candidate_target belief
  require_best_plan_win
  warmup 80
  plan_rollout_steps 48
  plan_worker_steps 12
  save_worker_prefix_steps 12

data:
  194 best-plan-win states
  2328 valid non-pass prefix steps
  shard mean_gap range 0.809 -> 1.123
```

Main-policy probe:

```text
run:
  runs/adaptive-prefix-policy-belief-win-v0/

loss:
  policy_kl 1.0
  action_ce 0.3
  finish 0.1
  outcome 0.1
  update_scope policy-heads

fixed-v5 max250 128-row, seed 95300:
  init p0/p1/min: 8.59 / 10.94 / 8.59
  probe p0/p1/min: 13.28 / 13.28 / 13.28
  draw also decreased from ~55-57% to 50-55%
```

Expander smoke:

```text
Expander 8/12/16 64-row, seed 95320:
  init min 57.81
  probe min 51.56
```

Conclusion:

```text
Executed winning prefixes contain a real anti-draw / fixed-v5 signal when they
train the main policy directly. The same single-source CE update causes
multi-size Expander regression, especially 12/16 rows, so this checkpoint is not
a promotion candidate.

Do next:
  mix plan-q-prefix rows with Expander/adaptive preservation rows
  or add size/seat protected loss before further fixed-v5 prefix CE

Do not do next:
  promote the prefix CE checkpoint
  keep scaling pure 8x8 winning-prefix action CE without preservation data
```

## 2026-06-20 14:31 - V4 Adapter Size-Gate Follow-up

Implementation changes:

```text
adaptive_plan_q_dataset.py:
  added init-template flags for channels/input/value/outcome/strategy heads
  allows old checkpoints such as adaptive-unet-ppo-v4 to load into newer target templates

adaptive_strategy_supervised.py:
  added --no-strategy-aux for pure policy-delta adapters
  plan-q-prefix CE-only adapters can now keep the base policy schema

evaluate_adaptive_policy.py:
  added main-model init-template flags
  added policy-adapter min/max grid-size gates
```

GPU results:

```text
v4 heuristic Plan-Q prefix collection:
  runs/adaptive-plan-q-prefix-v4-8win-v0/
  432 best-plan-win states
  5051 non-pass prefix rows
  mean_gap about 1.16-1.19

pure v4 prefix adapter:
  runs/adaptive-prefix-policy-v4-8win-v0/
  fixed-v5 max250 256-row seed 95640:
    v4 base min 8.98%
    adapter scale=1.0 min 7.81%
    adapter scale=0.25 min 8.20%
  conclusion: heuristic best-plan prefix CE still makes the policy lose more often

v4 base + midgame trajectory v3 as 8x8-only policy adapter:
  fixed-v5 max250 512-row seed 95740:
    expanded v4 base p0/p1/min: 10.16 / 10.94 / 10.16
    adapter p0/p1/min: 11.52 / 13.67 / 11.52

  Expander 8/12/16 512-row seed 95780:
    adapter min: 73.05
    rows:
      8p0 73.24
      8p1 73.05
      12p0 82.03
      12p1 84.77
      16p0 76.95
      16p1 82.03

  same-seed Expander 256-row check:
    expanded v4 base min 71.09
    adapter min 73.83
```

Interpretation:

```text
The best current deployment-shaped composition is not a new full checkpoint.
It is:

  base:
    runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

  8x8 adapter:
    runs/adaptive-midgame-contact-searchwin-imitation-v3/
      generals-adaptive-midgame-contact-searchwin-imitation-v3.eqx

  inference:
    --policy-adapter-scale 1.0
    --policy-adapter-max-grid-size 8
```

This keeps the stronger v4 12/16 behavior and injects the existing midgame
trajectory fixed-v5 signal only on 8x8. It is still a modest gain, not a v5
breakthrough: fixed-v5 max250 moved from 10.16% to 11.52% min at 512 rows.

Do next:

```text
Use the adapter-size-gate pattern as the default diagnostic wrapper.
Collect more genuine midgame decisive trajectory data, not heuristic Plan-Q prefixes.
Train/validate adapters per size before mixing them into the base.
Run GPU evals serially; parallel JAX evals triggered CUDA allocation warnings on the 16GB card.
```

## 2026-06-20 14:46 - Legacy Fixed-v5 Adapter Loader

Implementation changes:

```text
adaptive_network.py:
  added explicit drop-mismatched legacy checkpoint loading
  matching leaves are restored
  shape-mismatched leaves are reinitialized after consuming the serialized array

evaluate_adaptive_policy.py:
  added --drop-mismatched-init-leaves
  added --policy-adapter-mode delta|blend|replace
```

Reason:

```text
runs/adaptive-fixed-v5-imitation-v5/ckpts/
  generals-adaptive-fixed-v5-imitation-v5-iter-000030.eqx

could not load into the current template because one old value-head leaf had
disk shape (128,64) while the current template expected (64,128). The policy
trunk and policy head were still useful, so the correct recovery path is to
drop only mismatched leaves under an explicit evaluation flag.
```

Loader validation:

```text
legacy iter30 direct load with --drop-mismatched-init-leaves:
  fixed-v5 max250, 128 games/seat, seed 95820
  p0 17.19%
  p1 16.41%
  min 16.41%
```

Same-seed fixed-v5 gate:

```text
base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

adapter:
  runs/adaptive-fixed-v5-imitation-v5/ckpts/
    generals-adaptive-fixed-v5-imitation-v5-iter-000030.eqx

inference:
  --policy-adapter-scale 1.0
  --policy-adapter-mode replace
  --policy-adapter-max-grid-size 8
  --drop-mismatched-init-leaves

fixed-v5 max250, 512 games/seat, seed 95920:
  v4 base:
    p0 8.79%
    p1 10.94%
    min 8.79%
    draw about 54-56%

  v4 + legacy adapter:
    p0 11.91%
    p1 16.21%
    min 11.91%
    draw about 65-69%
```

`--policy-adapter-mode delta --policy-adapter-scale 1.0` produced the same
fixed-v5 result as `replace`, which is expected: a scale-1 centered legal delta
is the adapter logits plus a legal-action constant.

Same-seed Expander gate:

```text
Expander adaptive, 512 games/row, seed 95940:

v4 base:
  8p0 71.88%
  8p1 77.15%
  12p0 85.35%
  12p1 82.23%
  16p0 78.91%
  16p1 81.45%
  min 71.88%

v4 + legacy adapter, max-grid-size 8:
  8p0 88.48%
  8p1 89.26%
  12p0 85.35%
  12p1 82.23%
  16p0 78.91%
  16p1 81.45%
  min 78.91%
```

Interpretation:

```text
The legacy fixed-v5 imitation checkpoint is currently the strongest 8x8 adapter.
It lifts the six-row Expander 512-row min from 71.88% to 78.91% by fixing the
8x8 rows, while 12/16 rows remain bit-for-bit unchanged under the size gate.

This is not a fixed-v5 max250 breakthrough. The fixed-v5 win-rate gain is real
but draw-heavy; the short-gate problem is still finishability, not just survival.
```

Current best deployment-shaped Expander wrapper:

```text
base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

8x8 adapter:
  runs/adaptive-fixed-v5-imitation-v5/ckpts/
    generals-adaptive-fixed-v5-imitation-v5-iter-000030.eqx

evaluation flags:
  --network-arch unet
  --channels 64,96,128,64
  --scoreboard-history
  --fog-memory
  --value-heads per-size
  --value-head-sizes 8,12,16
  --value-loss mse
  --init-value-heads shared
  --init-value-loss mse
  --policy-adapter-path <legacy iter30>
  --policy-adapter-scale 1.0
  --policy-adapter-mode replace
  --policy-adapter-max-grid-size 8
  --drop-mismatched-init-leaves
```

Do next:

```text
Keep this wrapper as the current Expander diagnostic baseline.
Use the legacy adapter as an 8x8 expert/protection source when training the next
midgame decisive policy, but do not spend more time on policy-adapter scale
sweeps. The next fixed-v5 push still needs decisive trajectory / finishability
supervision rather than more inference-time logit mixing.
```

## 2026-06-20 15:06 - Legacy Plan-Q Prefix Adapter Probe

Implementation changes:

```text
adaptive_strategy_supervised.py:
  added --drop-mismatched-init-leaves

train_adaptive.py:
  added --drop-mismatched-init-leaves
  added --teacher-drop-mismatched-init-leaves

adaptive_plan_q_dataset.py:
  added --drop-mismatched-init-leaves
```

These flags let the current training/data tools use legacy fixed-v5 imitation
checkpoints directly, not only the evaluator.

Negative probes:

```text
legacy search-win policy-head CE:
  init:
    adaptive-fixed-v5-imitation-v5 iter30
  data:
    adaptive-fixed-v5-searchwin-a1-v1
    adaptive-midgame-terminal-searchwin-fixed-v5-safev3-v0
  samples:
    3214 after size-seat balancing
  update:
    policy heads only
    KL 1.0
    action CE 0.25
  result:
    fixed-v5 max250 512-row seed 96040:
      trained adapter min 13.48%
      same-seed legacy adapter min 14.84%
  conclusion:
    action CE on search-best labels still degrades the legacy expert

legacy fixed-v5 PPO:
  init:
    adaptive-fixed-v5-imitation-v5 iter30
  config:
    128 envs
    mixed seats
    256 rollout
    top-advantage 0.25 stratified
    sparse terminal reward
    40 iterations
    EMA saved
  training signal:
    rollout learner wins were effectively zero
  result:
    fixed-v5 max250 512-row seed 96120:
      min 0.00%
  conclusion:
    sparse fixed-v5 PPO from this expert collapses into fast losses

legacy fixed-v5 PPO with teacher KL:
  config:
    teacher KL 0.05 to legacy iter30
    lr 5e-7
    30 iterations
  result:
    fixed-v5 max250 512-row seed 96260:
      min 0.00%
  conclusion:
    KL did not solve the sparse-positive PPO failure mode
```

Old checkpoint selection:

```text
fixed-v5 imitation v3/v4/v5, same seed 96160:
  v3 adapter:
    p0 16.02%
    p1 14.06%
    min 14.06%

  v4 adapter:
    p0 16.02%
    p1 13.48%
    min 13.48%

  v5 iter30 adapter:
    p0 14.84%
    p1 14.06%
    min 14.06%

conclusion:
  no older fixed-v5 imitation checkpoint clearly beats v5 iter30
```

Positive probe: legacy Plan-Q executed-prefix data.

Collection:

```text
run:
  runs/adaptive-plan-q-legacy-mainstack-fixedv5-v0/

base:
  adaptive-fixed-v5-imitation-v5 iter30

opponent:
  fixed 8x8 v5 sample

candidate source:
  main-stack

candidate target:
  heuristic

plan rollout:
  4x4 plans
  24 rollout steps
  1 rollout per plan
  8 target-conditioned worker steps
  save 8 executed prefix steps

filters:
  warmup 80
  turn >= 80
  best plan outcome win
  plan_q_gap >= 0.25

data:
  shard 0: 14/256 states kept, mean_gap 0.9932
  shard 1: 28/256 states kept, mean_gap 1.0303
  total:
    42 plan states
    312 non-pass winning prefix rows before balancing
    112 rows after size-seat balancing
```

Training:

```text
run:
  runs/adaptive-legacy-planq-prefix-policy-v0/

init:
  adaptive-fixed-v5-imitation-v5 iter30

update:
  policy heads only
  KL 1.0 to saved prefix logits
  action CE 0.3
  20 epochs
  lr 5e-5

offline:
  action CE 2.537 -> 2.476
  teacher action accuracy 50.9%
  policy KL 0.000 -> 0.0043
```

Fixed-v5 gate:

```text
fixed-v5 max250, 512 games/seat, seed 96400:
  legacy iter30 adapter:
    p0 14.06%
    p1 12.70%
    min 12.70%
    draw about 68%

  legacy Plan-Q prefix adapter:
    p0 15.04%
    p1 16.41%
    min 15.04%
    draw about 65%
```

Expander gate:

```text
Expander adaptive, 512 games/row, seed 96420:
  legacy iter30 adapter:
    8p0 85.55%
    8p1 87.89%
    12p0 83.01%
    12p1 81.45%
    16p0 83.01%
    16p1 79.69%
    min 79.69%

  legacy Plan-Q prefix adapter:
    8p0 88.09%
    8p1 87.11%
    12p0 83.01%
    12p1 81.45%
    16p0 83.01%
    16p1 79.69%
    min 79.69%
```

Interpretation:

```text
This is the first positive fixed-v5 movement in this round that does not damage
the 8/12/16 Expander gate. It is still small and draw-heavy, but the mechanism
is different from failed action CE:

  collect counterfactual source-target plans on the legacy expert distribution
  keep only high-gap best-plan-win states
  train executed prefix actions, not single-step search labels

The tiny 42-state dataset moved fixed-v5 same-seed min from 12.70% to 15.04%.
Scaling this exact data source is now more promising than PPO, search-best CE,
or checkpoint selection.
```

Current best wrapper:

```text
base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

8x8 adapter:
  runs/adaptive-legacy-planq-prefix-policy-v0/
    generals-adaptive-legacy-planq-prefix-policy-v0.eqx

flags:
  --policy-adapter-scale 1.0
  --policy-adapter-mode replace
  --policy-adapter-max-grid-size 8
```

Do next:

```text
Scale the same legacy Plan-Q prefix collection:
  more shards
  keep main-stack source
  compare heuristic vs enemy-general/belief target only after enough rows
  keep filters: turn>=80, best-plan-win, gap>=0.25

Train a v1 prefix adapter with:
  1k-5k prefix rows after balancing if available
  policy-head-only first
  no PPO

Promotion target:
  fixed-v5 max250 512-row min > 17%
  Expander 8/12/16 512-row min >= 79%
```

## 2026-06-20 15:28 - Plan-Q Prefix Scaling and Midgame Trajectory Follow-up

This round used the GPU directly (`Device: cuda:0`). The sandbox cannot expose
CUDA to JAX even though `nvidia-smi` works, so GPU training/evaluation was run
with elevated execution. All model artifacts stayed under ignored `runs/`.

Legacy Plan-Q prefix scaling:

```text
v1 data:
  runs/adaptive-plan-q-legacy-mainstack-fixedv5-v1/
  150 high-gap best-plan-win states
  1177 valid non-pass prefix steps
  seat split: p0 85, p1 65
  mean plan_q_gap: 1.039

v1 adapter:
  data: v0 + v1
  samples after size-seat balance: 1426
  update: legacy iter30 policy heads only
  fixed-v5 max250 512-row seed96620:
    v1 min 11.91%
    v0 same seed min 10.94%
    legacy same seed min 14.26%
```

The larger v1 collection did not preserve the small v0 improvement. To diagnose
label quality, `adaptive_strategy_supervised.py --dataset-format plan-q-prefix`
now supports:

```text
--min-teacher-action-logit-margin <margin>
```

For each saved prefix step it computes:

```text
teacher_logit(chosen_prefix_action) - max_teacher_logit
```

and filters rows below the threshold. With `--min-teacher-action-logit-margin
-1.0`, training became much cleaner offline:

```text
v2 margin-filtered adapter:
  samples after size-seat balance: 760
  action CE: about 0.58
  teacher action accuracy: about 75%

fixed-v5 max250 512-row seed96720:
  v2 margin adapter:
    p0 14.65%
    p1 13.28%
    min 13.28%

  v0 same seed:
    p0 16.21%
    p1 11.91%
    min 11.91%

  legacy iter30 same seed:
    p0 14.06%
    p1 13.87%
    min 13.87%
```

Interpretation:

```text
Filtering by base-policy support improves offline CE/accuracy, but it still does
not beat the same-seed legacy adapter. The useful v0 signal remains real but
fragile; scaling raw prefix rows or selecting labels by teacher-logit support is
not enough.
```

Midgame decisive trajectory follow-up:

```text
a1mix-v0:
  init: adaptive-midgame-contact-searchwin-imitation-v3
  data: A1 fixed-v5 search-win + terminal search-win + fixed-v5 draw/contact +
        Expander contact protection
  update: all U-Net weights
  action CE only on A1/terminal search-best-win paths
  fixed-v5 max250 256-row seed96820:
    p0 10.94%
    p1 11.72%
    min 10.94%

v3 adapter same-seed control:
  p0 10.55%
  p1  7.42%
  min  7.42%

a1pos-v1:
  data: A1 + terminal positives + draw contrast
  action row weight: about 50%
  fixed-v5 max250 256-row seed96820:
    min 10.16%

rescue-searchwin-policyhead-v0:
  data: trajectory non-win rows where search_best_outcome == win
  kept rows: 3296, seat-balanced to 3206
  update: policy heads only
  fixed-v5 max250 256-row seed96820:
    min 10.16%
```

Conclusion:

```text
The A1/terminal/rescue trajectory data contains a small fixed-v5 signal: a1mix
beats the same-seed v3 adapter. It still does not reach the legacy or Plan-Q v0
512-row band and does not justify promotion.

Primitive trajectory/action imitation remains the wrong compression target at
this stage. The next useful step should move the same midgame decisive data into
finish/value/command gating or a plan-conditioned executor with an explicit
"enter/exit plan" decision, not another action CE or adapter-scale sweep.
```

Current best wrapper remains:

```text
base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

8x8 adapter:
  runs/adaptive-legacy-planq-prefix-policy-v0/
    generals-adaptive-legacy-planq-prefix-policy-v0.eqx
```

## 2026-06-20 15:37 - Deployment-Shaped Policy Adapter Gate Probe

Implemented a gate-trainer compatibility change:

```text
adaptive_policy_adapter_gate_supervised.py:
  --base-outcome-head / --no-base-outcome-head
  --base-strategy-aux / --no-base-strategy-aux
  --base-strategy-spatial-aux / --no-base-strategy-spatial-aux
  --base-strategy-finish-outputs
  --drop-mismatched-init-leaves
```

This lets the gate train on the actual deployment shape:

```text
base:
  adaptive-unet-ppo-v4
  no strategy/outcome heads

adapter:
  a strategy-head checkpoint

feature model:
  adapter itself, or an explicit strategy-head model
```

Gate training:

```text
run:
  runs/adaptive-policy-adapter-gate-a1mix-v0/

base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

adapter:
  runs/adaptive-midgame-decisive-trajectory-a1mix-v0/
    generals-adaptive-midgame-decisive-trajectory-a1mix-v0.eqx

data:
  adaptive-fixed-v5-searchwin-a1-v1
  adaptive-midgame-terminal-searchwin-fixed-v5-safev3-v0

examples:
  rows 3890
  changed actions 179
  positives 52
  positive fraction 29.05%

final gate:
  weighted acc about 71.5%
  P+ 0.557
  P- 0.402
```

Fixed-v5 gate:

```text
fixed-v5 max250, 256 games/seat, seed97020:
  a1mix ungated:
    p0 10.55%
    p1 12.11%
    min 10.55%

  a1mix learned gate threshold 0.5:
    p0 13.67%
    p1 12.11%
    min 12.11%

  current best legacy Plan-Q v0 same seed:
    p0 16.41%
    p1 14.06%
    min 14.06%

fixed-v5 max250, 512 games/seat, seed97040:
  a1mix learned gate threshold 0.5:
    p0 10.94%
    p1 11.13%
    min 10.94%
```

Interpretation:

```text
The learned gate is no longer the old sparse-gate failure mode: the high-quality
A1/terminal rows produce 29% positives among changed-action examples, and the
gate improves a1mix on a same-seed 256-row fixed-v5 gate.

The 512-row confirmation still falls back to about 11% min and remains below the
legacy Plan-Q prefix wrapper. Do not sweep thresholds. The useful signal is that
deployment-shaped gating can learn where an adapter is locally search-consistent;
the missing piece is stronger candidate behavior and longer enter/exit-plan
labels, not a better threshold.
```

## 2026-06-20 15:48 - Policy Adapter Commit Probe

Implemented an evaluator-only enter-plan diagnostic:

```text
flag:
  evaluate_adaptive_policy.py --policy-adapter-commit-steps N

behavior:
  after a learned adapter gate or finish-threshold trigger,
  force the adapter for the next N policy turns
```

GPU smoke:

```text
device:
  cuda:0, RTX 5070 Ti

run:
  v4 base
  a1mix adapter
  a1mix learned gate threshold 0.5
  commit steps 4
```

Fixed-v5 max250 result:

```text
256 games/seat, seed97120:
  p0:
    wins 21 / losses 81 / draws 154
    win 8.20%
    draw 60.16%

  p1:
    wins 25 / losses 96 / draws 135
    win 9.77%
    draw 52.73%

  min:
    8.20%
```

Comparison:

```text
a1mix learned gate, no commit, 256-row seed97020:
  min 12.11%

current best legacy Plan-Q v0 same seed family:
  min 14.06%
```

Interpretation:

```text
Committed adapter forcing is worse than the non-committed learned gate. This
rules out the simplest enter-plan continuation heuristic. Keep the flag as a
diagnostic switch, but do not run a 512-row confirmation or sweep commit length.

The next useful route is still better decisive trajectory labels or a true
plan-conditioned executor, not extending a weak adapter decision for several
turns.
```

## 2026-06-20 16:00 - Legacy Plan-Q Adapter Gate Offline Probe

Tested whether the current best legacy Plan-Q prefix adapter can be gated
instead of always replacing 8x8 logits.

Setup:

```text
base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

adapter:
  runs/adaptive-legacy-planq-prefix-policy-v0/
    generals-adaptive-legacy-planq-prefix-policy-v0.eqx

feature model:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/
    generals-adaptive-midgame-contact-searchwin-imitation-v3.eqx

gate data:
  adaptive-fixed-v5-searchwin-a1-v1
  adaptive-midgame-terminal-searchwin-fixed-v5-safev3-v0
```

The legacy adapter is policy-only, so the gate used a separate feature model for
strategy finish features and treated outcome features as zero. The first attempt
with the regular per-size value-head template failed on the legacy checkpoint's
old value-head serialization; retrying with a shared value-head template loaded
the policy logits and reinitialized irrelevant value leaves.

Dataset:

```text
rows:
  3890

changed adapter actions:
  943

positives:
  275

positive fraction among changed examples:
  29.16%
```

Training:

```text
lr 1e-3, 80 epochs:
  unstable
  final P+ 0.872
  final P- 0.871

lr 1e-4, 100 epochs:
  final loss 0.7035
  final acc 46.4%
  final P+ 0.458
  final P- 0.465
```

Interpretation:

```text
The gate does not separate positive and negative legacy-adapter changed actions,
even though the positive fraction is healthy. This is a data/feature boundary
failure, not a threshold problem. Do not run gameplay eval or sweep thresholds
for this legacy adapter gate.

The current best remains the ungated v4 + legacy Plan-Q prefix adapter. Further
fixed-v5 work needs better enter/exit labels or a true plan-conditioned executor,
not a binary gate over the existing legacy delta.
```

## 2026-06-20 16:11 - Wrapper-Aligned Search Data Probe

Added adapter-aware search teacher support to the strategy dataset collector.
This lets data collection use the same deployed wrapper as evaluation:

```text
base:
  adaptive-unet-ppo-v4

adapter:
  adaptive-legacy-planq-prefix-policy-v0

composition:
  mode replace
  scale 1.0
  max grid size 8
```

New collector flags:

```text
adaptive_strategy_dataset.py:
  --teacher-adapter-model-path
  --teacher-adapter-scale
  --teacher-adapter-mode delta|blend|replace
  --teacher-adapter-min-grid-size
  --teacher-adapter-max-grid-size
  --teacher-drop-mismatched-init-leaves
```

The adapter is composed into both:

```text
1. teacher_logits saved for KL anchoring
2. rollout-search prior and rollout policy
```

This fixes the previous deployment mismatch where offline search data came from
a single checkpoint but evaluation used `v4 + adapter`.

GPU collection:

```text
run:
  runs/adaptive-wrapper-searchwin-v0/

teacher:
  v4 + legacy Plan-Q adapter

opponent:
  fixed v5 sample

filters:
  turn >= 80
  contact
  search_score_gap >= 0.25
  search_best_outcome == win

search:
  top_k 4
  rollout_steps 16
  rollouts/action 1

kept:
  shard0 191 / 8192
  shard1 245 / 8192
  total 436

seat:
  p0 250
  p1 186

trajectory labels:
  win 272
  draw-risk 117

mean search gap:
  1038.24
```

Training/eval:

```text
v0, init v4 base:
  runs/adaptive-wrapper-searchwin-policy-v0/
  policy-head only
  fixed-v5 max250 256-row seed97400:
    p0 10.55%
    p1 9.77%
    min 9.77%

same-seed legacy Plan-Q baseline:
  p0 12.11%
  p1 12.89%
  min 12.11%
```

The v0 failure was expected after inspection: initializing from v4 and using
replace mode discards the legacy adapter's existing behavior.

```text
v1, init legacy Plan-Q adapter:
  runs/adaptive-wrapper-searchwin-policy-v1-legacyinit/
  policy-head only
  lr 5e-5
  policy_kl 2.0
  action_ce 0.2

fixed-v5 max250 256-row seed97400:
  p0 15.23%
  p1 14.45%
  min 14.45%

same-seed legacy baseline:
  min 12.11%

fixed-v5 max250 512-row seed97420:
  p0 14.26%
  p1 11.52%
  min 11.52%

same-seed legacy baseline:
  p0 15.23%
  p1 11.72%
  min 11.72%
```

The v1 256-row signal did not confirm at 512 rows. It is essentially equal to
legacy at larger sample size and slightly worse on p1.

```text
v2, legacy init + size-seat oversample:
  runs/adaptive-wrapper-searchwin-policy-v2-legacyinit-balanced/

fixed-v5 max250 256-row seed97400:
  p0 13.67%
  p1 12.11%
  min 12.11%
```

Interpretation:

```text
The wrapper-aligned collector is useful infrastructure and should replace
single-checkpoint search data when the deployed policy is a base+adapter wrapper.

The tiny 436-row search-best CE refinement is not enough to improve the current
legacy Plan-Q adapter. The next attempt should use the same collector but either:
  1. collect substantially more wrapper-aligned search-win data with p1 balance,
  2. train finish/outcome/belief heads from these rows instead of primitive CE,
  3. use accepted changed-action or multi-step executor labels rather than one
     search-best primitive action.

Do not promote v0/v1/v2.
```

## 2026-06-20 17:03 - Max500 Counterfactual Prefix Seat-Balanced Scaling

Changed the fixed-v5 diagnostic gate from `max250` to `max500` and added a
single-seat collection control to the Plan-Q collector:

```text
adaptive_plan_q_dataset.py:
  --learner-seat mixed|p0|p1
```

The default remains `mixed`. The new `p0`/`p1` modes are for accepted-prefix
collection where the causal filter can heavily favor one learner seat; without
this control, `adaptive_strategy_supervised.py --balance-strata size-seat`
throws away most of the useful rows.

GPU data collection used the current legacy Plan-Q prefix adapter distribution,
fixed-v5 sample opponent, max500 truncation, strict causal filtering, and 6x6
main-stack/heuristic plan candidates:

```text
base model:
  runs/adaptive-legacy-planq-prefix-policy-v0/generals-adaptive-legacy-planq-prefix-policy-v0.eqx

mixed-seat v4-c6:
  output: runs/adaptive-plan-q-legacy-prefix-accepted-max500-v4-c6/
  accepted states: 279
  prefix rows: 3284
  seat counts: p0 33, p1 246
  base win rate: 3.6%
  mean plan advantage: 1.171
  median best-plan time-to-terminal: 20

p0补数 v4-c6-p0:
  output: runs/adaptive-plan-q-legacy-prefix-accepted-max500-v4-c6-p0/
  accepted states: 415
  prefix rows: 4929
  seat counts: p0 415
  base win rate: 7.7%
  mean plan advantage: 0.952
  median best-plan time-to-terminal: 26

combined max500 prefix set:
  shards: 81
  accepted states: 759
  accepted prefix rows: 8993
  seat counts: p0 462, p1 297
  base win rate: 5.7%
  mean plan advantage: 1.026
  mean plan-Q gap: 1.123
  median best-plan time-to-terminal: 24
```

Training/eval:

```text
v4-c6balanced:
  output: runs/adaptive-legacy-planq-prefix-policy-v4-max500-c6balanced/
  init: legacy Plan-Q prefix v0
  update: policy heads only
  policy_kl: 1.0
  action_ce: 0.2
  prefix_pairwise_margin: 0.2
  epochs: 20
  lr: 3e-5
  final KL: 0.026

fixed-v5 max500, 256 games/seat, seed98460:
  p0 35.16%
  p1 42.19%
  min 35.16%
  adapter_diff ~38.9%

v4-c6balanced-conservative:
  output: runs/adaptive-legacy-planq-prefix-policy-v4-max500-c6balanced-conservative/
  policy_kl: 2.0
  action_ce: 0.05
  prefix_pairwise_margin: 0.05
  epochs: 10
  lr: 1e-5
  final KL: 0.0007

fixed-v5 max500, 256 games/seat, seed98460:
  p0 36.33%
  p1 42.19%
  min 36.33%
  adapter_diff ~38.4%

current best legacy Plan-Q v0 wrapper:
  base: runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx
  adapter: runs/adaptive-legacy-planq-prefix-policy-v0/generals-adaptive-legacy-planq-prefix-policy-v0.eqx
  mode: replace
  max_grid_size: 8

fixed-v5 max500, 256 games/seat, seed98460:
  p0 42.97%
  p1 46.09%
  min 42.97%

fixed-v5 max500, 512 games/seat, seed98800:
  p0 38.09%
  p1 41.41%
  min 38.09%
  draw: p0 12.70%, p1 9.57%
```

Interpretation:

```text
The 6x6 accepted-prefix data is much higher volume and has strong offline
counterfactual advantage, but direct policy-head imitation still regresses
gameplay. This is now evidence against simply scaling executed-prefix CE/margin
training, even with better seat balance and max500 truncation.

The current best deployment remains v4 base + legacy Plan-Q v0 adapter. For
max500, the promotion baseline should be the 512-row min 38.09%, not the noisier
256-row 42.97%.

Next direction:
  1. keep --learner-seat for data balancing,
  2. stop training v0 further from raw prefix actions,
  3. convert accepted-prefix data into enter/exit-plan or finish/outcome labels,
  4. evaluate only if a candidate clears the current max500 512-row baseline.
```

## 2026-06-20 17:20 - Prefix-Derived Policy Adapter Gate

Extended `adaptive_policy_adapter_gate_supervised.py` so policy-adapter gates can
train directly from executed Plan-Q prefix shards:

```text
new/changed:
  --dataset-format strategy|plan-q-prefix
  --min-prefix-plan-advantage
  --allow-prefix-nonwin
  --require-prefix-base-not-win
  --max-prefix-plan-time-to-terminal
  --max-prefix-step
```

For prefix shards, positive labels mean:

```text
adapter top action == executed accepted-prefix action
and adapter changes the base top action
and plan_outcome == win
and plan_advantage >= threshold
and optional base_outcome != win
and optional plan_time / prefix_step filters pass
```

This is intentionally stricter than raw prefix imitation. It asks whether the
current adapter already contains the useful accepted-prefix action, then learns
when to let that adapter replace the base policy.

Also extended `evaluate_adaptive_policy.py policy_adapter_gate_features` from
14 to 18 features by appending:

```text
scoreboard_time
scoreboard_land_advantage
scoreboard_army_advantage
contact_binary
```

Old gate sidecars remain compatible because evaluation slices by saved
`feature_names` length.

GPU training used the max500 accepted-prefix data from the previous section:

```text
base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

adapter:
  runs/adaptive-legacy-planq-prefix-policy-v0/generals-adaptive-legacy-planq-prefix-policy-v0.eqx

feature model:
  runs/adaptive-midgame-contact-searchwin-imitation-v3/
    generals-adaptive-midgame-contact-searchwin-imitation-v3.eqx

filters:
  min_prefix_plan_advantage = 0.25
  require_prefix_base_not_win = true
  max_prefix_plan_time_to_terminal = 80
  max_prefix_step = 7
```

First gate without phase/contact features:

```text
output:
  runs/adaptive-prefix-policy-adapter-gate-max500-v0/

examples:
  3085 changed-action examples
  positives: 21.72%
  valid prefix rows: 8951
  label domain: 5477
  prefix action matches: 1020

offline final:
  acc 59.7%
  P+ 0.534
  P- 0.475

fixed-v5 max500 256-row seed98460:
  threshold 0.5:
    trigger 100%
    p0 42.97%
    p1 46.09%
    min 42.97%
  threshold 0.6:
    trigger 100%
    min 42.97%
  threshold 0.8:
    trigger 100%
    min 42.97%
```

Interpretation: without phase/contact features, the gate cannot distinguish
opening from accepted-prefix midgame states. It just reproduces the ungated v0
adapter.

Phase/contact gate:

```text
output:
  runs/adaptive-prefix-policy-adapter-gate-max500-v1-phase/

offline final:
  acc 72.3%
  P+ 0.604
  P- 0.388

fixed-v5 max500 256-row seed98460:
  threshold 0.5:
    p0 39.06%
    p1 42.19%
    min 39.06%
    trigger p0 86.74%, p1 80.27%
    adapter_diff p0 27.49%, p1 22.45%
```

The phase-aware gate is technically useful: it suppresses a meaningful share of
adapter actions and learns the offline prefix label much better. Gameplay still
falls below the same-seed ungated v0 baseline (`42.97%` min), so do not promote
it or continue threshold sweeps.

Updated diagnosis:

```text
Accepted-prefix data is useful, but current v0 adapter's top action only matches
the accepted prefix on a subset of changed states. A binary gate over the old
adapter cannot synthesize missing plan execution. It can only choose when to use
the old adapter.

Next higher-yield route:
  train an explicit enter/exit plan policy or plan-conditioned Worker from these
  prefixes, then gate that executor. The gate needs phase/contact features, but
  the executor must be capable of producing the accepted prefix action.
```

### 2026-06-20 17:34 - Max500 Main-Stack Heuristic Plan-Worker Probe

User-facing gate change: fixed-v5 diagnostics should use `max500` instead of
`max250` by default. `max250` was too compressed and over-penalized otherwise
reasonable games; `max500` is now the main fixed-v5 short-game gate, with longer
horizons allowed when checking final behavior.

Implementation:

```text
evaluate_adaptive_policy.py:
  added --strategy-plan-worker-command-source main-stack-heuristic
  source: existing main-stack/route source picker
  target: public-observation heuristic over visible enemy, cities, fog structures,
          generals, and neutral cells
  aux: not required for rerank-only use; still required for learned Worker gates

README.md / docs/zh-manual.md:
  documented the new Plan-Worker command source
```

Training:

```text
output:
  runs/adaptive-plan-worker-prefix-max500-mainstack-heuristic-v0/

data:
  max500 accepted-prefix shards
  81 shards
  759 accepted plan rows
  8958 kept winning prefix examples / 9108 prefix slots

train:
  arch unet
  channels 64,96,128,64
  input 38 = 35 adaptive/fog/history + 3 command planes
  epochs 30
  lr 3e-5
  loss weights action=1, source=0.5, direction=0.5

final offline:
  action 48.5%
  source 65.2%
  direction 65.3%
```

Gameplay checks:

```text
base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

current best wrapper:
  + runs/adaptive-legacy-planq-prefix-policy-v0/
    generals-adaptive-legacy-planq-prefix-policy-v0.eqx
  + --policy-adapter-mode replace
  + --policy-adapter-max-grid-size 8

fixed-v5 max500 256-row seed98920:
  v4 + Worker only:
    p0 21.88%
    p1 21.88%
    min 21.88%

  current best wrapper:
    p0 36.72%
    p1 39.84%
    min 36.72%

  current best wrapper + Worker scale 0.02:
    p0 39.45%
    p1 41.02%
    min 39.45%

fixed-v5 max500 512-row seed98800:
  current best wrapper baseline:
    p0 38.09%
    p1 41.41%
    min 38.09%

  current best wrapper + Worker scale 0.02:
    p0 36.52%
    p1 37.50%
    min 36.52%
```

Conclusion:

```text
Do not promote the Plan-Worker combination.

The 256-row same-seed lift was a small false positive. At 512 rows, the Worker
hurts the max500 wrapper by -1.57pp min versus the existing v0 adapter baseline.
Keep max500 as the fixed-v5 gate, keep the new main-stack-heuristic command
source for future data/interface work, but do not continue Worker rerank-scale
sweeps.

Current deployment reference remains:
  v4 base + legacy Plan-Q prefix policy v0 adapter
  max500 512-row reference min: 38.09%
```

### 2026-06-20 17:54 - Online Counterfactual Search Wrapper

Rationale:

```text
The last several runs show that accepted-prefix data is real but hard to
compress into a single policy head, Worker, or gate. The next high-yield
diagnostic is to keep the counterfactual evaluator online and measure the
planner upper bound directly.
```

Implementation:

```text
evaluate_adaptive_policy.py:
  added --online-search-top-k
  added --online-search-rollout-steps
  added --online-search-rollouts-per-action
  added --online-search-min-turn
  added --online-search-require-contact
  added --online-search-min-grid-size / --online-search-max-grid-size
  added search scoring weights:
    --online-search-army-weight
    --online-search-land-weight
    --online-search-prior-weight
    --online-search-terminal-score

scope:
  supports fixed checkpoint opponents via --opponent-policy-path
  composes the base policy with the deployment policy adapter
  scores top-k primitive prior actions by short rollout against the fixed opponent
  executes the best-scoring action in the real game
```

GPU smoke:

```text
num_games=2
max_steps=10
top_k=2
rollout_steps=2
rollouts/action=1
result: ran on cuda:0
```

Main probe:

```text
base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

adapter:
  runs/adaptive-legacy-planq-prefix-policy-v0/
    generals-adaptive-legacy-planq-prefix-policy-v0.eqx

opponent:
  /home/codeboy/research/generals-bots/generals-ppo-8x8-expander-gpu-v5.eqx
  mode sample

gate:
  fixed-v5 max500

online search:
  top_k=4
  rollout_steps=16
  rollouts/action=1
  min_turn=80
  require_contact=true
  max_grid_size=8
```

Results:

```text
64 games/seat seed99020:
  baseline:
    p0 42.19%
    p1 32.81%
    min 32.81%
  online search:
    p0 53.12%
    p1 43.75%
    min 43.75%

128 games/seat seed99040:
  baseline:
    p0 32.03%
    p1 39.84%
    min 32.03%
  online search:
    p0 42.19%
    p1 50.78%
    min 42.19%

256 games/seat seed99060:
  baseline:
    p0 41.80%
    p1 38.28%
    min 38.28%
    draw 10.94% / 11.33%
  online search:
    p0 53.52%
    p1 50.39%
    min 50.39%
    draw 6.64% / 5.86%
```

Conclusion:

```text
This is the strongest fixed-v5 max500 result so far.

The online counterfactual wrapper improved same-seed 256-row min by +12.11pp
over the current best policy-adapter wrapper and pushed both seats to roughly
50% against v5 sample. It is not a pure .eqx checkpoint, but it proves the
missing capability is present in short counterfactual evaluation and not in the
current policy head.

Next:
  1. run 512-row confirmation for this exact wrapper
  2. collect online-search action traces as high-confidence DAgger data
  3. distill not only the chosen action, but search value, action margin,
     and search-enter state features
  4. add the same online-search path for Expander/heuristic opponents so 12/16
     can get an equivalent planner upper-bound diagnostic
```

### 2026-06-20 18:20 - Online Search First-Step Alignment Fix

Issue:

```text
The first 512-row max500 confirmation was weaker than the 256-row signal:

  no-search baseline seed99080:
    p0 44.14%
    p1 41.99%
    min 41.99%

  online search seed99080 before fix:
    p0 55.08%
    p1 44.73%
    min 44.73%

The p0 gain was large, but p1 barely moved. Inspection found that online
search sampled an opponent first action inside candidate scoring, while the
real environment step later sampled a different opponent action. That made
every candidate plan evaluate against a different stochastic first response
than the one actually executed.
```

Fix:

```text
evaluate_policy_opponent_batch now samples the real opponent action before
online search and passes that action into online_search_action_policy_opponent.
Candidate counterfactuals now force:

  candidate learner first action
  same opponent first action that the real step will execute

The rest of the search rollout remains stochastic as before.
```

Aligned 512-row result:

```text
fixed-v5 max500
seed: 99080
base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx
adapter:
  runs/adaptive-legacy-planq-prefix-policy-v0/
    generals-adaptive-legacy-planq-prefix-policy-v0.eqx
online search:
  top_k=4
  rollout_steps=16
  rollouts/action=1
  min_turn=80
  require_contact=true

baseline:
  p0 44.14%
  p1 41.99%
  min 41.99%
  draw 11.13% / 7.42%

aligned online search:
  p0 53.52%
  p1 51.56%
  min 51.56%
  draw 6.25% / 4.69%
```

Interpretation:

```text
This is now a promotion-grade planner signal for max500 fixed-v5:

  min gain vs same-seed baseline: +9.57pp
  both seats above 50%
  draw materially lower on both seats

The result is still a wrapper/planner, not a pure model checkpoint. The next
research move should be trace collection from the aligned online search path:
chosen action, candidate priors, candidate scores, score margin, search-enter
state, and whether the final episode converts draw/loss into win.
```

### 2026-06-20 18:47 - Online Search for Expander/Heuristic Opponents

Implementation:

```text
Extended evaluate_adaptive_policy.py online search beyond fixed checkpoint
opponents:

  online_search_action_policy_opponent:
    fixed checkpoint opponent path

  online_search_action_heuristic_opponent:
    built-in heuristic opponent path, including Expander

Both paths now use the same real first-step opponent action that the environment
will execute. The heuristic path calls opponent_action(...) inside each rollout
step, so it can be used for 8/12/16 Expander planner upper-bound diagnostics.
```

Note on legacy adapter template:

```text
The legacy Plan-Q prefix adapter was trained as an 8x gate. When evaluating
8/12/16 with that adapter loaded but size-gated to 8, use:

  --value-heads per-size
  --value-head-sizes 8

Policy logits are still valid for 12/16; value heads are not used by evaluation
actions. Without this, the loader builds a 3-head value template and the 8x
legacy adapter stream ends early.
```

128-row Expander triage:

```text
base:
  runs/adaptive-unet-ppo-v4/generals-adaptive-unet-ppo-v4.eqx

adapter:
  runs/adaptive-legacy-planq-prefix-policy-v0/
    generals-adaptive-legacy-planq-prefix-policy-v0.eqx

settings:
  grid_sizes: 8,12,16
  opponent: expander
  max_steps: 750
  policy_mode: sample
  online_search:
    top_k=4
    rollout_steps=16
    rollouts/action=1
    min_turn=80
    require_contact=true
    max_grid_size=16

seed99220 baseline:
  8p0 84.38%
  8p1 88.28%
  12p0 83.59%
  12p1 86.72%
  16p0 78.91% draw 16.41%
  16p1 78.12% draw 15.62%
  min 78.12%

seed99220 online search:
  8p0 85.94%
  8p1 91.41%
  12p0 95.31%
  12p1 87.50%
  16p0 92.19% draw 3.91%
  16p1 89.06% draw 5.47%
  min 85.94%
```

256-row Expander confirmation:

```text
seed99240 baseline:
  8p0 85.94%
  8p1 83.98%
  12p0 80.08%
  12p1 80.86%
  16p0 76.95% draw 14.45%
  16p1 77.34% draw 14.45%
  min 76.95%

seed99240 online search:
  8p0 89.45%
  8p1 89.45%
  12p0 88.67%
  12p1 89.84%
  16p0 88.28% draw 5.47%
  16p1 85.16% draw 6.25%
  min 85.16%
```

Conclusion:

```text
This is the first clear large-map planner upper-bound result:

  same-seed 256-row min gain: +8.21pp
  16x draw roughly cut from 14.45% to 5-6%
  all six rows above 85%

This does not mean the pure checkpoint has solved Expander. It means the
missing large-map behavior is accessible through short counterfactual primitive
search using the current U-Net policy as rollout policy. The next training
target should be aligned online-search distillation rather than another PPO
reward sweep:

  inputs:
    policy obs + memory + scoreboard history

  labels:
    base action
    search action
    candidate scores
    search margin
    search-enter flag
    final outcome

  objective:
    train policy/action-Q/finish heads to reproduce high-margin search choices
    only where the online search changes outcome timing or draw risk.
```
