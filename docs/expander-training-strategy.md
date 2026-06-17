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
