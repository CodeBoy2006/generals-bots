# Generals Bots

`generals-bots` 是一个基于 JAX 的 Generals.io 双人对战模拟器与 bot 实验框架。项目目标是提供可复现、可批量并行、适合强化学习研究的游戏环境。

更完整的中文手册见 [docs/zh-manual.md](docs/zh-manual.md)。超过 Expander 的训练过程与策略见 [docs/expander-training-strategy.md](docs/expander-training-strategy.md)。

## 项目概览

- `generals/core/`：核心游戏逻辑、环境包装、动作、观测、地图生成和奖励函数。
- `generals/agents/`：内置 agent，包括 `RandomAgent` 和 `ExpanderAgent`。
- `generals/gui/`：pygame 可视化和 replay GUI。
- `generals/remote/`：连接 generals.io 远程服务的客户端代码。
- `examples/`：单局、向量化、可视化示例。
- `examples/_experimental/ppo/`：实验性 PPO、行为克隆、策略评估和策略可视化工具。
- `tests/`：pytest 测试。
- `docs/`：中文手册和开发记录。

## 快速搭建

推荐使用 `uv` 按锁文件安装依赖。项目要求 Python 3.11 或更高版本。

```bash
git clone https://github.com/CodeBoy2006/generals-bots.git
cd generals-bots
uv sync --extra dev
```

确认包和 JAX 后端可用：

```bash
uv run python -c "import generals; print(generals.GeneralsEnv)"
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

CUDA 13 环境可安装 GPU extra：

```bash
uv sync --extra dev --extra cuda13
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

## 核心接口

`GeneralsEnv` 是主要环境入口。当前接口中，`reset` 会返回预生成 reset pool 和初始状态；`step` 需要显式传入该 pool。

```python
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import ExpanderAgent, RandomAgent

env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
agent_0 = RandomAgent()
agent_1 = ExpanderAgent()

key = jrandom.PRNGKey(42)
pool, state = env.reset(key)

while True:
    obs_0 = get_observation(state, 0)
    obs_1 = get_observation(state, 1)

    key, k1, k2 = jrandom.split(key, 3)
    actions = jnp.stack([
        agent_0.act(obs_0, k1),
        agent_1.act(obs_1, k2),
    ])

    timestep, state = env.step(state, actions, pool)
    if bool(timestep.terminated) or bool(timestep.truncated):
        break

print(f"Winner: {int(timestep.info.winner)}")
```

动作格式为长度 5 的整数数组：

```text
[pass, row, col, direction, split]
```

- `pass`：`1` 表示跳过，`0` 表示移动。
- `row`、`col`：源格子坐标。
- `direction`：`0=上`，`1=下`，`2=左`，`3=右`。
- `split`：`1` 表示移动一半军队，`0` 表示移动除 1 个驻军外的全部军队。

## 常用实验

单局对战：

```bash
uv run python examples/simple_example.py
```

并行环境示例：

```bash
uv run python examples/vectorized_example.py
```

pygame 可视化：

```bash
uv run python examples/visualization_example.py
```

玩家对战训练好的 PPO checkpoint：

```bash
./play-v5.command
```

观看 PPO 机器对战：

```bash
./watch-v5.command
```

`play-v5.command` 会使用 `uv run --python 3.12` 启动当前仓库根目录的
`generals-ppo-8x8-expander-gpu-v5.eqx`，默认玩家为 player 0，PPO 为
player 1，8x8 generated 地图，sample 策略，并展示 Top-3 候选动作。
开局会自动跳过双方都无法移动的初始 pass 回合，因此窗口第一帧即可点击自己的格子移动。
脚本默认启用自动 tick，每秒推进 2 回合；如果你没有提交动作，人类回合会自动 pass。
选中源格等待目标格时，自动 pass 会暂停，避免点目标前回合被跳过。
macOS 下也可以在 Finder 中双击该脚本启动。若 checkpoint 不在仓库根目录，
可设置 `MODEL_PATH=/path/to/model.eqx ./play-v5.command`。脚本默认传
`POLICY_INPUT=auto`，加载时会按 checkpoint 形状自动选择 9 通道 observation
或 18 通道 augmented-full-state；也可以用
`POLICY_INPUT=observation|full-state|augmented-full-state` 手动覆盖。

`watch-v5.command` 使用同一个 checkpoint 控制 PPO 0 和 PPO 1，默认 sample
策略、自动 tick、每秒 4 回合。可通过环境变量分别选择两个模型：

```bash
MODEL_0_PATH=generals-ppo-8x8-expander-gpu-v4.eqx \
MODEL_1_PATH=generals-ppo-8x8-expander-gpu-v5.eqx \
./watch-v5.command
```

如果某个 checkpoint 是 18 通道 augmented 输入，默认 `auto` 会从 checkpoint
形状自动为该玩家使用 `augmented-full-state`。也可以用 `MODEL_0_POLICY_INPUT`
和 `MODEL_1_POLICY_INPUT` 分别覆盖两个 AI 的输入类型。

用 GUI 运行 `v5 + rollout-search`：

```bash
SEARCH_POLICY=1 ./play-v5.command
```

这会让人机模式里的 PPO 对手使用 v5 checkpoint 作为 policy prior，并在每步对
top-k 候选动作做短 rollout 搜索。默认搜索预算为
`SEARCH_TOP_K=4`、`SEARCH_ROLLOUT_STEPS=16`、`SEARCH_ROLLOUTS_PER_ACTION=4`，
比普通 v5 明显更慢。若想先流畅观察，可降低预算：

```bash
SEARCH_POLICY=1 \
SEARCH_TOP_K=2 \
SEARCH_ROLLOUT_STEPS=8 \
SEARCH_ROLLOUTS_PER_ACTION=2 \
./play-v5.command
```

观看对战时可以只让 player 0 使用 rollout-search：

```bash
MODEL_0_SEARCH_POLICY=1 ./watch-v5.command
```

或者两边都使用搜索：

```bash
MODEL_0_SEARCH_POLICY=1 MODEL_1_SEARCH_POLICY=1 ./watch-v5.command
```

rollout-search GUI 目前只支持 9 通道 observation checkpoint，当前
`generals-ppo-8x8-expander-gpu-v5.eqx` 满足这个条件。

也可以手动指定两个 checkpoint：

```bash
uv run python examples/play_against_model.py \
  --machine-vs-machine \
  --model-0-path runs/generals-ppo-a.eqx \
  --model-1-path runs/generals-ppo-b.eqx \
  --model-0-policy-input augmented-full-state \
  --model-1-policy-input observation \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode sample \
  --opponent-policy-mode sample \
  --tick-rate 4
```

手动打开 rollout-search：

```bash
uv run python examples/play_against_model.py generals-ppo-8x8-expander-gpu-v5.eqx \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode sample \
  --policy-input observation \
  --search-policy \
  --search-top-k 4 \
  --search-rollout-steps 16 \
  --search-rollouts-per-action 4
```

手动指定 checkpoint 和参数：

```bash
uv run python examples/play_against_model.py runs/generals-ppo-8x8-generated.eqx \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode sample \
  --auto-tick \
  --tick-rate 2 \
  --human-player 0 \
  --fps 30 \
  --preview-top-k 3
```

控制方式：

- 左键点击自己的可移动格子，再点击相邻目标格移动。
- `S` 切换下一步是否移动一半军队，`P` 跳过本回合。
- 右键或 `Esc` 取消选中，终局后按 `R` 重开，`Q` 退出。
- 选中的源格会显示黄色边框，可移动目标格会显示绿色边框。
- 右侧面板会显示当前选择、split 状态和最近一次点击结果。
- 自动 tick 默认开启，会在没有人类动作时自动 pass 并推进回合；`--no-auto-tick` 可关闭，`--tick-rate` 控制每秒自动推进次数。
- 默认会在棋盘和右侧面板展示 PPO 模型下一步 Top-K 候选动作、概率和 value。
- `--preview-top-k` 可设置展示 1-5 个候选，`--no-ai-preview` 可关闭预览。
- `--policy-mode sample` 时预览显示的是采样分布，实际动作仍按概率抽样。
- `--machine-vs-machine` 会关闭人类输入流程，让两个 PPO agent 按自动 tick 对战；`--model-0-path` 和 `--model-1-path` 可分别指定两个 checkpoint。
- `--opponent-model-path` 仍可作为 `--model-1-path` 的兼容别名。

加载 checkpoint 时，`--grid-size` 必须与保存该 `.eqx` 模型时使用的网络尺寸一致。`.eqx` 属于实验产物，建议放在项目内已忽略的 `runs/` 或其他非缓存实验目录，不要提交进 Git。

4x4 PPO smoke test：

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --num-steps 64 \
  --num-iterations 10 \
  --model-path runs/generals-ppo-4x4.eqx
```

8x8 generated 地图 PPO：

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --num-steps 64 \
  --num-iterations 10 \
  --pool-size 512 \
  --model-path runs/generals-ppo-8x8-generated.eqx
```

从已有 checkpoint 继续 PPO，并使用多 epoch/minibatch 更新：

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
  --opponent expander \
  --init-model-path runs/generals-ppo-8x8-expander-gpu-v4.eqx \
  --model-path runs/generals-ppo-8x8-expander-gpu-v5.eqx
```

冻结 checkpoint 自博弈训练：

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
  --init-model-path runs/generals-ppo-current.eqx \
  --opponent-policy-path runs/generals-ppo-best-frozen.eqx \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 1.0 \
  --model-path runs/generals-ppo-selfplay-next.eqx
```

当前策略自博弈训练：

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
  --init-model-path runs/generals-ppo-current.eqx \
  --self-play-opponent \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 1.0 \
  --model-path runs/generals-ppo-current-selfplay.eqx
```

`--self-play-opponent` 会让非 learner 玩家在每次 rollout 中使用当前正在更新的同一个 policy；它不能和 `--opponent-policy-path` 或 `--opponent-policy-pool` 同时使用。`--opponent-policy-pool a.eqx,b.eqx` 会让普通 PPO 从多个同架构 frozen checkpoint 中采样对手，适合 checkpoint league best-response；可用 `--opponent-policy-pool-modes sample,greedy` 指定各自执行模式。`--checkpoint-dir`、`--checkpoint-every` 和 `--keep-checkpoints` 可周期保存并保留中间模型，便于后续 league 评估选模。`--learner-player` 可以把 learner 放在 player 0 或 player 1；`--terminal-reward-scale` 会在 decisive terminal transition 上额外加入零和胜负奖励。`--general-target-reward-scale` 会用完整状态奖励强兵靠近敌方 general 的势能变化，可配合 `--general-target-min-army` 和 `--general-target-max-distance` 控制触发条件。`--path-assignment-reward-scale` 会在 reward 内缓存 passable shortest-path 距离场，并把强兵分配到敌方 general、非己方城市或前线目标，可用 `--path-assignment-*-weight` 控制目标优先级。`--policy-input augmented-full-state` 可让 PPO learner 使用 18 通道输入，通常与 `--init-input-channels 9` 一起从 v5 这类 9 通道 checkpoint 扩展。如果候选模型和冻结对手使用不同网络容量，可用 `--channels` 和 `--opponent-channels` 分别指定四层卷积通道，例如 `--channels 64,64,64,32 --opponent-channels 32,32,32,16`。

自适应多尺寸 PPO：

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
  --checkpoint-dir runs/generals-adaptive-bc-checkpoints \
  --checkpoint-every 100 \
  --keep-checkpoints 10 \
  --model-path runs/generals-adaptive-bc-8-12-16.eqx
```

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
  --init-model-path runs/generals-adaptive-bc-8-12-16.eqx \
  --init-channels 64,64,64,32 \
  --checkpoint-dir runs/generals-adaptive-ppo-checkpoints \
  --checkpoint-every 50 \
  --keep-checkpoints 10 \
  --model-path runs/generals-adaptive-ppo-8-12-16.eqx
```

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_adaptive_policy.py runs/generals-adaptive-ppo-8-12-16.eqx \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --num-games 2048 \
  --max-steps 750 \
  --opponent expander \
  --policy-mode sample \
  --map-generator generated \
  --json-output runs/generals-adaptive-ppo-8-12-16-eval.json \
  --require-win-rate 0.90
```

`behavior_clone_adaptive.py`、`train_adaptive.py` 和 `evaluate_adaptive_policy.py` 使用固定 `--pad-to` 画布保存一个 checkpoint，并通过 `--grid-sizes` 在 8x8、12x12、16x16 等有效棋盘之间切换。评估器会自动跑每个尺寸的 player 0 和 player 1 两个座位，`--require-win-rate 0.90` 会在任一尺寸或座位未达标时返回非零退出码。当前 GUI 和固定尺寸 `evaluate_policy.py` 仍只支持普通 `PolicyValueNetwork` checkpoint，不能直接加载 adaptive checkpoint。
`behavior_clone_adaptive.py` 和 `train_adaptive.py` 都支持 `--channels` 指定 adaptive 网络容量，也支持 `--grid-size-weights` 对困难尺寸过采样；`train_adaptive.py` 还支持 `--init-channels` 从不同容量的 adaptive checkpoint 零填充扩容 warm start。`--learner-player alternate` 按 training iteration 交替更新两个座位；`--learner-player mixed` 会把总 `num_envs` 拆成 player 0 和 player 1 两半，并把两侧轨迹拼进同一个 PPO update，适合减少按轮次交替造成的座位遗忘。`--reward-mode terminal` 会关闭 dense composite reward，只保留 terminal win/loss；`--gamma`、`--gae-lambda`、`--top-advantage-fraction`、`--top-advantage-mode stratified`、`--ema-decay` 和 `--eval-ema` 用于 v3-noarch 长 rollout/EMA 训练；默认 `--top-advantage-mode global` 会在整个 rollout batch 里取最高 advantage transition，`stratified` 会按 effective size 和 learner seat 分开取样，适合诊断困难行是否被全局 top-k 饿死。`train_adaptive.py --opponent-policy-path` 可让 adaptive PPO 在单一有效尺寸上对抗固定 `PolicyValueNetwork` checkpoint；`evaluate_adaptive_policy.py --opponent-policy-path` 可用同一机制评估 adaptive checkpoint 对固定 8x8 v5 等 policy 对手；当只评估或训练单个尺寸但 checkpoint 带有多尺寸 per-size value heads 时，用 `--value-head-sizes` 和 `--init-value-head-sizes` 指定 checkpoint 内实际 value head 尺寸。`--value-loss hl-gauss --value-bins 128 --value-sigma 0.04` 会把 PPO value loss 从 scalar MSE 切换为 HL-Gauss categorical CE。`--outcome-aux-weight` 会启用 loss/draw/win 辅助头，只用同一 rollout 内已知结局的 episode segment 做监督；`--global-context` 会把 land/army/time scoreboard 标量追加为 20 通道输入，`--scoreboard-history` 进一步追加 previous scoreboard 和 one-step delta，形成 30 通道输入，并通过零初始化上下文分支 warm start 旧 15 通道 checkpoint，旧 checkpoint 通常配 `--init-input-channels 15`。`--context-residual` 会在 adaptive CNN trunk 后追加零输出 5x5 residual context branch，`--pyramid-context` 会追加 16→8→4→8→16 的零输出 U-Net-style pyramid branch，二者都适合从旧 checkpoint 保持初始行为不变后继续学习更大感受野；`--context-only-update` 可先只更新这些 context/pyramid branch，降低全量 PPO 续训导致的旧策略漂移；从已有 context/pyramid checkpoint 续训或评估时同时传对应的 `--init-context-residual`、`--init-pyramid-context`、`evaluate_adaptive_policy.py --context-residual` 或 `--pyramid-context`，如果 PPO warm start 来自带 strategy aux heads 的 checkpoint，还要传 `train_adaptive.py --init-strategy-aux` 来丢弃 aux heads 并保留共享权重。`adaptive_search_distill.py` 的 `--base-global-context` / `--base-scoreboard-history` 可让 base/search/opponent teacher 使用 20/30 通道 shared-MSE adaptive checkpoint；不传这两个参数时 teacher 仍按旧 15 通道 checkpoint 加载。distill 脚本也支持 student `--context-residual` / `--pyramid-context`，并可用 `--freeze-context-strategy-aux` 只训练 context/pyramid branch 与 strategy auxiliary heads。`adaptive_teacher_imitation.py --fixed-teacher-model-path` 可把固定 8x8 policy teacher 的 logits 映射到 adaptive padded action space，并可配 `--opponent-policy-path` 在 v5-vs-v5 分布上做 U-Net 行为克隆；`--outcome-weight-mode terminal` 会按同一 rollout 内已知 loss/draw/win 给 KL/CE 样本加权，配套 `--win-action-weight`、`--loss-action-weight`、`--draw-action-weight` 和 `--unknown-action-weight` 可用于偏向赢家轨迹或弱化超时 draw 轨迹；`--terminal-action-window` 会把动作监督进一步聚焦到下一次终局前的最后 N 步，`--p0-action-weight`/`--p1-action-weight` 可做 seat-specific reweight，`--outcome-aux-weight` 可在 imitation 中同时监督 loss/draw/win head。`--strategy-q-weight`、`--strategy-q-rank-weight`、`--strategy-intent-weight`、`--strategy-finish-weight` 和 `--strategy-belief-weight` 会启用 top-k Q、pairwise Q-rank、弱 intent、finish 和敌方 general belief 辅助头；`--strategy-q-target outcome` 会把 Q/rank target 从 shaped rollout score 改为候选动作的 replacement outcome（loss/draw/win -> `-1/0/+1`），`--strategy-q-target outcome-score` 会用胜/平/负做主排序并用 `--strategy-q-outcome-score-weight` 加少量 shaped score 作为 tie-break，默认 `score` 保持旧行为；`--strategy-q-weight-mode accepted` 只在候选动作相对 top-prior 有 outcome 改善或同 outcome 高 margin 改善时训练 Q/rank，默认 `active` 保持旧的全 active-row 辅助监督；`--strategy-q-replay-capacity` 和 `--strategy-q-replay-ratio` 会缓存 accepted Q rows 并按比例追加到后续 batch，replay 行只保留 Q/rank 权重、不会重复放大 KL/search/intent/belief loss；`--freeze-strategy-aux-only` 只更新这些 strategy auxiliary heads，适合先做辅助头预训练而不扰动 policy/trunk；从已有 strategy checkpoint 续训时用 `--init-strategy-aux`。`evaluate_adaptive_policy.py --strategy-q-rerank-scale <x>` 可以在 `--strategy-aux` checkpoint 上把 strategy-Q 输出作为 policy logits 的中心化 bias 做推理探针。`adaptive_worker_pretrain.py` 会把 target heatmap、eligible source heatmap 和 BFS route potential 追加为 18 通道 Worker 输入，用 full-state BFS/path-assignment 生成 soft action targets；`--action-loss-weight`、`--source-loss-weight` 和 `--direction-loss-weight` 可把 Worker 训练从单一 flat action CE 改成 source-cell 与 direction 的分解监督。`evaluate_worker_policy.py` 可以用 fogged observation command wrapper 单独或混合 fallback adaptive checkpoint 评估 Worker 接管，`--hybrid-mode rerank --worker-logit-scale <x>` 会把 Worker logits 作为 fallback logits 的中心化 bias；fallback 是 U-Net、fog-memory 或 scoreboard-history checkpoint 时，配套传 `--fallback-network-arch unet`、`--fallback-fog-memory`、`--fallback-scoreboard-history` 以及必要的 fallback value-head 模板参数。但这些 checkpoint 仍是 Commander/Worker 实验件，不是 `evaluate_adaptive_policy.py` promotion 策略。 如果 checkpoint 使用 categorical/per-size value head、outcome head、strategy aux、global/context residual、pyramid context、scoreboard history 或 global context，评估时也要给 `evaluate_adaptive_policy.py` 传匹配的 `--value-heads`、value-loss 模板参数、`--outcome-head`、`--strategy-aux`、`--context-residual`、`--pyramid-context`、`--global-context` 或 `--scoreboard-history`。

下一轮 adaptive PPO v3-outcome continuation 建议从当前最强候选启动；本地 16GB GPU 已验证 512 envs x 256 steps 会 OOM，先用 256 envs 和 1024 minibatch 做 256 games/row triage：

```bash
JAX_PLATFORMS=cuda TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --extra dev --extra cuda13 python examples/_experimental/ppo/train_adaptive.py 256 \
  --grid-sizes 8,12,16 \
  --grid-size-weights 8:1,12:1,16:2 \
  --pad-to 16 \
  --map-generator generated \
  --pool-size 16384 \
  --num-steps 256 \
  --num-iterations 80 \
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
  --init-model-path runs/generals-adaptive-search-distill-p1-v1-ckpts/generals-adaptive-search-distill-p1-v1-iter-000040.eqx \
  --checkpoint-dir runs/generals-adaptive-ppo-v3-outcome-ckpts \
  --checkpoint-every 10 \
  --keep-checkpoints 8 \
  --model-path runs/generals-adaptive-ppo-v3-outcome.eqx \
  --seed 68000
```

Residual GRU 记忆 PPO：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train_recurrent.py 512 \
  --grid-size 8 \
  --map-generator generated \
  --opponent-policy-path runs/generals-ppo-8x8-expander-gpu-v5.eqx \
  --init-model-path runs/generals-ppo-8x8-expander-gpu-v5.eqx \
  --hidden-size 64 \
  --freeze-base \
  --model-path runs/generals-recurrent-ppo-8x8-v5.eqx
```

`train_recurrent.py` 会在 CNN policy 上叠加 GRU hidden state 和 residual logits/value delta；`--freeze-base` 会冻结 warm-start 的 CNN，只训练记忆适配器，适合保护 v5 或行为克隆基线。用 `evaluate_recurrent_policy.py` 可评估 recurrent checkpoint；没有 `--opponent-policy-path` 时也可直接测 `--opponent expander`。

胜者轨迹辅助克隆：

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
  --init-model-path runs/generals-ppo-current.eqx \
  --opponent-policy-path runs/generals-ppo-best-frozen.eqx \
  --policy-mode sample \
  --opponent-policy-mode sample \
  --winner-source both \
  --negative-weight 0.0 \
  --model-path runs/generals-ppo-outcome-clone.eqx
```

`outcome_clone.py` 会完整 rollout 对局，并只用最终胜者视角的动作做监督样本；`--winner-source learner` 只保留 learner 赢局，`--negative-weight` 可额外压低败者动作概率。

rollout-search 强辅助评估：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/search_policy.py runs/generals-ppo-8x8-expander-gpu-v5.eqx \
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
  --rollouts-per-action 4
```

该脚本不训练新 checkpoint；它把 checkpoint 作为 policy prior，并对 top-k 候选动作做短 rollout 评分，可作为强评估策略或后续蒸馏 teacher。

保守 rollout-search 蒸馏：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/conservative_search_distill.py 128 \
  --base-model-path runs/generals-ppo-8x8-expander-gpu-v5.eqx \
  --target-mode soft \
  --num-steps 64 \
  --num-iterations 80 \
  --min-margin 1 \
  --margin-scale 4 \
  --improve-weight 0.02 \
  --kl-weight 1.0 \
  --lr 0.000001 \
  --model-path runs/generals-ppo-8x8-conservative-search.eqx
```

该脚本用固定 base checkpoint 做 rollout-search teacher，并用 KL 约束学生贴近 base。`--target-mode hard` 只在 search 最优动作明显优于 base top-prior 动作时加入小权重动作监督；`--target-mode soft` 会把 top-k rollout 分数转成软目标，避免把近似并列候选强行压成单标签。`--policy-input full-state` 会用 privileged 完整状态替换标准 observation；`--policy-input augmented-full-state` 会保留原 9 个 observation 通道，并追加 9 个 full-state 通道。augmented 模式默认使用 18 输入通道，并支持从 9 通道 checkpoint 自动扩展 conv1 权重。评估时也要给 `evaluate_policy.py` 传同名 `--policy-input`，必要时传 `--input-channels` 和 `--opponent-input-channels`。它适合继续研究 search distillation，不应把训练 loss 当成棋力指标；仍需用 `evaluate_policy.py --opponent-policy-path` 独立评估。

若要用同一门槛评估 checkpoint 或强辅助策略，可使用 `evaluate_league.py`。默认会评估所有 heuristic 的两个 seat；加 `--checkpoint-opponent v5=...:sample` 可纳入 v5 gate；加 `--search-policy` 会评估 `v5 + rollout-search` 这类强辅助推理策略，并输出最低 required pair 胜率 `league_score`。

行为克隆 warm start：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 2000 \
  --lr 0.0007 \
  --model-path runs/generals-bc-8x8-soft.eqx
```

批量评估 checkpoint：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py runs/generals-ppo-8x8-expander-gpu-v5.eqx \
  --num-games 2048 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500 \
  --opponent expander \
  --policy-mode sample \
  --policy-player 0
```

使用 `--policy-player 1` 可做镜像座位评估，避免只测 player 0 带来的出生点偏差。

评估两个 checkpoint 之间的对局时，给 `evaluate_policy.py` 传入 `--opponent-policy-path` 和 `--opponent-policy-mode`。
评估非默认容量 checkpoint 时，给候选传入 `--channels`；如果对手 checkpoint 容量不同，再传入 `--opponent-channels`。评估非默认输入通道或 privileged 输入时，使用 `--policy-input`、`--input-channels` 和 `--opponent-input-channels` 保持网络结构与 checkpoint 一致。

Adaptive strategic trunk experiments can now use a real U-Net backbone instead of the legacy shallow CNN:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/adaptive_teacher_imitation.py 128 \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --grid-size-weights 8:2,12:2,16:1 \
  --network-arch unet \
  --channels 64,96,128,64 \
  --global-context \
  --scoreboard-history \
  --fog-memory \
  --value-heads per-size \
  --value-loss hl-gauss \
  --teacher-model-path legacymodels/generals-adaptive-ppo-v3-composite-balanced-probe1.eqx \
  --teacher-global-context \
  --teacher-scoreboard-history \
  --teacher-input-channels 30 \
  --teacher-policy-mode greedy \
  --kl-weight 1.0 \
  --action-ce-weight 5.0 \
  --model-path runs/adaptive-unet-imitation/generals-adaptive-unet-imitation.eqx
```

`--fog-memory` appends explored-ever, last-seen enemy ownership, last-seen enemy army, seen city, and seen general planes. `adaptive_teacher_imitation.py` lets the legacy adaptive CNN drive mixed-seat rollouts and trains the U-Net with all-action teacher KL plus action CE. Greedy teacher action labels are much less noisy than sampled labels, while KL still preserves the teacher distribution. U-Net checkpoints must be evaluated with matching `--network-arch unet`, `--channels`, `--scoreboard-history`, `--fog-memory`, and value-head flags.

`adaptive_strategy_dataset.py` collects offline strategy-supervision shards under `runs/` without changing policy weights. Each `.npz` shard stores adaptive observations, legal masks, teacher logits/actions, outcome/terminal labels, enemy-general and hidden-enemy maps, weak intent labels, and source/target heatmaps. Example U-Net teacher collection:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/adaptive_strategy_dataset.py 16 \
  --grid-sizes 8,12,16 \
  --pad-to 16 \
  --num-steps 128 \
  --teacher-kind adaptive \
  --teacher-model-path runs/adaptive-unet-imitation-v3/generals-adaptive-unet-imitation-v3.eqx \
  --teacher-network-arch unet \
  --teacher-channels 64,96,128,64 \
  --teacher-input-channels 35 \
  --teacher-scoreboard-history \
  --teacher-value-heads per-size \
  --teacher-value-head-sizes 8,12,16 \
  --teacher-value-loss hl-gauss \
  --teacher-outcome-head \
  --scoreboard-history \
  --fog-memory \
  --opponent expander \
  --output-dir runs/adaptive-strategy-dataset-v0/unet-v3-expander
```

For midgame decisive trajectory imitation, use save-time filters so the shard keeps contact-heavy, terminal-window, or gather/attack states instead of every opening move. `--min-save-turn`, `--max-save-turn`, `--require-contact`, `--min-visible-enemy-cells`, `--terminal-window`, `--require-win`, `--require-finish-within-250`, `--require-win-or-finish-within-250`, and `--draw-only` filter rows after rollout labels are computed and before the shard is written. Separate shards are preferred for separate windows, for example terminal-120 wins and draw-heavy contact states, then mix those shards in `adaptive_strategy_supervised.py`.

`adaptive_strategy_dataset.py --teacher-kind search` uses an adaptive checkpoint as the policy prior, scores its top-k actions with short rollouts, and saves the best search action as the imitation action while still storing the prior logits for KL anchoring. It supports scoreboard-history and fog-memory U-Net checkpoints, so current 35-channel models can generate midgame decisive trajectory shards. Complex teacher checkpoints that include auxiliary heads can be loaded with `--teacher-strategy-aux`, `--teacher-strategy-spatial-aux`, and `--teacher-strategy-finish-outputs`. Keep the search budget small for smoke runs, then raise `--search-top-k`, `--search-rollout-steps`, and `--search-rollouts-per-action` for production shards.

Search teacher shards also store `search_candidate_indices`, `search_scores`, `search_outcomes`, `search_best_outcome`, and `search_score_gap`. The gap is computed as best score minus the second valid prior candidate, so pass-only opening states do not get fake high-margin labels. Use `--min-search-score-gap` and optionally `--require-search-best-win` to keep only high-confidence search decisions:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/adaptive_strategy_dataset.py 32 \
  --grid-sizes 8,12,16 \
  --teacher-kind search \
  --teacher-model-path runs/adaptive-midgame-search-imitation-v1/generals-adaptive-midgame-search-imitation-v1.eqx \
  --teacher-network-arch unet \
  --teacher-channels 64,96,128,64 \
  --teacher-input-channels 35 \
  --teacher-scoreboard-history \
  --teacher-value-heads per-size \
  --teacher-value-head-sizes 8,12,16 \
  --teacher-value-loss hl-gauss \
  --teacher-outcome-head \
  --teacher-strategy-aux \
  --teacher-strategy-finish-outputs 3 \
  --scoreboard-history \
  --fog-memory \
  --search-top-k 4 \
  --search-rollout-steps 8 \
  --search-rollouts-per-action 1 \
  --min-save-turn 80 \
  --require-contact \
  --min-visible-enemy-cells 1 \
  --require-win-or-finish-within-250 \
  --terminal-window 120 \
  --min-search-score-gap 0.25 \
  --output-dir runs/adaptive-strategy-search-highgap-v1
```

`adaptive_strategy_supervised.py` consumes those shards and trains only the frozen-base strategy heads. The first stage intentionally leaves the policy/trunk logits unchanged while learning intent, finish-within-250, and enemy-general belief:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/adaptive_strategy_supervised.py \
  --dataset 'runs/adaptive-strategy-dataset-v0/unet-v3-expander/*.npz' \
  --dataset 'runs/adaptive-strategy-dataset-v0/fixed-v5-max250/*.npz' \
  --network-arch unet \
  --channels 64,96,128,64 \
  --init-channels 64,96,128,64 \
  --input-channels 35 \
  --init-input-channels 35 \
  --global-context \
  --value-heads per-size \
  --init-value-heads per-size \
  --value-head-sizes 8,12,16 \
  --init-value-head-sizes 8,12,16 \
  --value-loss hl-gauss \
  --init-value-loss hl-gauss \
  --outcome-head \
  --init-outcome-head \
  --init-model-path runs/adaptive-unet-imitation-v3/generals-adaptive-unet-imitation-v3.eqx \
  --num-epochs 20 \
  --minibatch-size 512 \
  --intent-weight 0.2 \
  --finish-weight 0.4 \
  --belief-weight 0.3 \
  --model-path runs/adaptive-strategy-supervised-v0/generals-adaptive-strategy-supervised-v0.eqx
```

For multi-horizon finishability, use `--finish-head-mode multi-horizon`. This expands the strategy finish head to three independent BCE logits for `finish_within_50`, `finish_within_100`, and `finish_within_250`; warm-start old binary-finish checkpoints with `--init-finish-head-mode binary`. Evaluation and PPO warm-starts that need to load such checkpoints should pass `--strategy-finish-outputs 3` or `--init-strategy-finish-outputs 3` respectively.

Draw-heavy shards should not usually contribute action CE. `adaptive_strategy_supervised.py --action-ce-weight-mode non-draw` keeps policy KL, outcome, finish, belief, and intent losses on known draw samples, but removes those rows from action CE/QCE. `wins` is stricter and only applies action CE/QCE to known winning rows; `search-best-win` applies action CE only when a search-teacher shard's local `search_best_outcome` is win, which is useful for broad midgame/contact datasets where draw/loss rows should still train outcome/finish/belief but not primitive actions. `all` preserves the historical behavior.

Use `adaptive_strategy_supervised.py --balance-strata size-seat` when a filtered shard is concentrated in a few grid-size or learner-seat rows. It downsamples to equal `(grid_size, seat)` counts before shuffling, which is useful for high-gap or terminal-window imitation probes where raw row counts can hide seat tradeoffs. For mixed-domain datasets, `--balance-strata size-seat-domain` further splits rows by coarse shard domain, using sidecar metadata such as fixed policy opponent versus Expander opponent when available. This prevents fixed-v5 8x rows from drowning same-size Expander protection rows during coupled policy updates.

`adaptive_strategy_supervised.py --label-source search-best` switches only the finish/outcome labels from full-trajectory outcome to the saved rollout-search best-action outcome. This is useful on high-gap search shards where the recorded trajectory may eventually win but the local search best action still distinguishes immediate win-like states from draw-like states. Pair it with `--finish-head-mode binary --init-finish-head-mode multi-horizon` when converting a 3-logit finish checkpoint into a 2-logit finish gate. `--balance-finish-labels` and `--balance-outcome-labels` apply batch-level inverse-frequency weighting so rare search-win labels are not drowned by search-draw rows.

`adaptive_strategy_supervised.py --update-scope strategy-value-heads` is the middle setting between frozen-head probes and full policy coupling. It keeps trunk/policy/action logits frozen, but lets the strategy auxiliary heads, optional outcome head, and shared pooled value bottleneck (`value_linear1`) update. Use it for search-best finish/outcome representation work where the last-layer heads alone underfit, but a full `--update-scope all` policy update would risk seat/size regressions.

`adaptive_strategy_supervised.py --update-scope policy-heads` is a narrower policy-coupled mode for decisive imitation probes. It freezes the U-Net/CNN trunk and value heads, but updates the primitive policy output head (`policy_conv`/`pass_linear`) plus the strategy auxiliary heads and optional outcome head. This is useful when full-trunk imitation shows fixed-opponent signal but causes Expander forgetting; keep a positive `--policy-kl-weight` so the output head stays anchored to the warm-start policy.

This is a representation-learning step, not a gameplay promotion step. For ordinary trajectory labels, use `--outcome-weight 0` initially; the U-Net imitation v3 outcome head is poorly calibrated on fixed-v5 draw-heavy states, so outcome supervision should wait for better balanced shards or a fresh outcome head. Search-best labels are a separate probe because they supervise local search outcome, not the final rollout winner.

For a cautious policy-coupled follow-up, start from a checkpoint that already has trained strategy heads, cap each shard to avoid long fixed-v5 rollouts dominating the batch, and require a positive policy KL anchor:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/adaptive_strategy_supervised.py \
  --dataset 'runs/adaptive-strategy-dataset-v1/v4-expander-balanced/*.npz' \
  --dataset 'runs/adaptive-strategy-dataset-v0/fixed-v5-max250/*.npz' \
  --dataset 'runs/adaptive-strategy-dataset-v0/fixed-v5-max500/*.npz' \
  --dataset 'runs/adaptive-strategy-dataset-v0/fixed-v5-max750/*.npz' \
  --max-samples-per-shard 4096 \
  --network-arch unet \
  --channels 64,96,128,64 \
  --init-channels 64,96,128,64 \
  --input-channels 35 \
  --init-input-channels 35 \
  --global-context \
  --value-heads per-size \
  --init-value-heads per-size \
  --value-head-sizes 8,12,16 \
  --init-value-head-sizes 8,12,16 \
  --value-loss hl-gauss \
  --init-value-loss hl-gauss \
  --outcome-head \
  --init-outcome-head \
  --init-strategy-aux \
  --init-model-path runs/adaptive-strategy-heads-v4-balanced-v1/generals-adaptive-strategy-heads-v4-balanced-v1.eqx \
  --update-scope all \
  --policy-kl-weight 1.0 \
  --action-ce-weight 0.05 \
  --lr 0.0000005 \
  --model-path runs/adaptive-strategy-coupled-v4-balanced-v1/generals-adaptive-strategy-coupled-v4-balanced-v1.eqx
```

`--update-scope all` is intentionally guarded by `--policy-kl-weight > 0` because otherwise the offline strategy losses can move the trunk without an action-distribution anchor. Current coupled checkpoints are diagnostic artifacts; keep using 256/512-row promotion gates before replacing the active Expander base.

`train_adaptive.py --teacher-kl-weight` can also anchor PPO to complex adaptive checkpoints. When the teacher checkpoint has per-size HL-Gauss value heads, outcome head, or strategy auxiliary heads, pass the matching `--teacher-value-heads`, `--teacher-value-head-sizes`, `--teacher-value-loss`, `--teacher-outcome-head`, `--teacher-strategy-aux`, and `--teacher-strategy-finish-outputs` flags so the teacher tree matches the serialized checkpoint. This is useful for KL-anchoring PPO to strategy-supervised U-Net checkpoints without using the teacher for rollout actions.

To train only the strategy-Q head for inference-time reranking, keep the base policy frozen and add Q losses against teacher logits/actions:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/adaptive_strategy_supervised.py \
  --dataset 'runs/adaptive-strategy-dataset-v1/v4-expander-balanced/*.npz' \
  --dataset 'runs/adaptive-strategy-dataset-v0/fixed-v5-max250/*.npz' \
  --dataset 'runs/adaptive-strategy-dataset-v0/fixed-v5-max500/*.npz' \
  --dataset 'runs/adaptive-strategy-dataset-v0/fixed-v5-max750/*.npz' \
  --max-samples-per-shard 4096 \
  --network-arch unet \
  --channels 64,96,128,64 \
  --init-channels 64,96,128,64 \
  --input-channels 35 \
  --init-input-channels 35 \
  --global-context \
  --value-heads per-size \
  --init-value-heads per-size \
  --value-head-sizes 8,12,16 \
  --init-value-head-sizes 8,12,16 \
  --value-loss hl-gauss \
  --init-value-loss hl-gauss \
  --outcome-head \
  --init-outcome-head \
  --init-strategy-aux \
  --init-model-path runs/adaptive-strategy-heads-v4-balanced-v1/generals-adaptive-strategy-heads-v4-balanced-v1.eqx \
  --q-kl-weight 1.0 \
  --q-action-ce-weight 0.05 \
  --intent-weight 0.05 \
  --finish-weight 0.1 \
  --belief-weight 0.05 \
  --model-path runs/adaptive-strategy-q-rerank-v1/generals-adaptive-strategy-q-rerank-v1.eqx
```

Evaluate Q reranking with `evaluate_adaptive_policy.py --strategy-aux --strategy-q-rerank-scale <scale>`. Treat this as a probe: current results show the Q head can learn the offline teacher distribution, but direct all-action reranking has not passed 512-row promotion.

Search-teacher shards can also train the strategy-Q head from rollout-search top-k rankings instead of policy/action labels. `adaptive_strategy_supervised.py --search-q-rank-weight <w>` consumes `search_candidate_indices`, `search_prior_scores`, `search_scores`, and `search_score_gap`, builds a soft ranking target with `--search-q-temperature`, and only updates rows with at least two valid candidates and positive search gap. `--search-q-value-weight <w>` instead regresses candidate action-Q values to rollout-search replacement outcomes (`loss/draw/win -> -1/0/+1`), with optional shaped-score tie-breaks from `--search-q-score-scale` and `--search-q-outcome-score-weight`. This keeps high-gap rollout-search signal out of primitive policy CE:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/adaptive_strategy_supervised.py \
  --dataset 'runs/adaptive-strategy-search-highgap-v1/*.npz' \
  --balance-strata size-seat \
  --network-arch unet \
  --channels 64,96,128,64 \
  --init-channels 64,96,128,64 \
  --input-channels 35 \
  --init-input-channels 35 \
  --global-context \
  --value-heads per-size \
  --init-value-heads per-size \
  --value-head-sizes 8,12,16 \
  --init-value-head-sizes 8,12,16 \
  --value-loss hl-gauss \
  --init-value-loss hl-gauss \
  --outcome-head \
  --init-outcome-head \
  --init-strategy-aux \
  --finish-head-mode multi-horizon \
  --init-finish-head-mode multi-horizon \
  --init-model-path runs/adaptive-midgame-search-imitation-v1/generals-adaptive-midgame-search-imitation-v1.eqx \
  --update-scope strategy-heads \
  --intent-weight 0 \
  --finish-weight 0 \
  --belief-weight 0 \
  --outcome-weight 0 \
  --policy-kl-weight 0 \
  --action-ce-weight 0 \
  --search-q-rank-weight 1.0 \
  --search-q-temperature 1.0 \
  --model-path runs/adaptive-midgame-search-q-rank-v0/generals-adaptive-midgame-search-q-rank-v0.eqx
```

The first high-gap probe learned the search rank loss, but q-rerank remained a diagnostic path: scale `0.001` only moved Expander 256-row min from `71.09%` to `71.48%`, while scale `0.01` collapsed the 8x8 player-1 row at 64 games. A later fixed-v5 max250 candidate-outcome value probe reduced offline value MSE but still failed direct `--strategy-q-replace-threshold` promotion at 128 games. Do not promote Q-rerank or Q-replace from these results; use them as evidence that rollout-search should feed finish/outcome/belief or midgame trajectory imitation before primitive action replacement.

The strategy supervised trainer can also add explicit source/target spatial heads from `source_heatmap` and `target_heatmap` shard labels. Use `--strategy-spatial-aux` when creating the target checkpoint and leave `--init-strategy-spatial-aux` off when expanding an older strategy checkpoint that does not yet contain these heads:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/adaptive_strategy_supervised.py \
  --dataset 'runs/adaptive-strategy-dataset-v1/v4-expander-balanced/*.npz' \
  --dataset 'runs/adaptive-strategy-dataset-v0/fixed-v5-max250/*.npz' \
  --dataset 'runs/adaptive-strategy-dataset-v0/fixed-v5-max500/*.npz' \
  --dataset 'runs/adaptive-strategy-dataset-v0/fixed-v5-max750/*.npz' \
  --network-arch unet \
  --channels 64,96,128,64 \
  --init-channels 64,96,128,64 \
  --input-channels 35 \
  --init-input-channels 35 \
  --global-context \
  --value-heads per-size \
  --init-value-heads per-size \
  --value-loss hl-gauss \
  --init-value-loss hl-gauss \
  --outcome-head \
  --init-outcome-head \
  --init-strategy-aux \
  --strategy-spatial-aux \
  --init-model-path runs/adaptive-strategy-q-rerank-v1/generals-adaptive-strategy-q-rerank-v1.eqx \
  --source-weight 0.5 \
  --target-weight 0.5 \
  --model-path runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx
```

Evaluate explicit source/target inference bias with `evaluate_adaptive_policy.py --strategy-aux --strategy-spatial-aux --strategy-spatial-rerank-scale <scale>`. This is still a probe; the current spatial v1 run learned the labels offline but did not pass 256-row Expander or fixed-v5 promotion.

`adaptive_plan_q_dataset.py` collects source-target plan-Q shards. By default it samples candidate source cells and target cells from privileged state, forces each plan's first primitive action, rolls out the base adaptive policy briefly, and saves plan scores plus source/target score marginals. `--candidate-source model --candidate-target model` instead takes candidates from the checkpoint's strategy spatial source/target heads, masked to legal movable sources and active passable targets; use this for gate/Worker data that must match inference-time command generation. `--candidate-source model-worker --candidate-target model` uses the same route/army-adjusted source scoring as the evaluator's worker command for the top model target. `--min-plan-gap <gap>`, `--require-best-plan-win`, `--min-save-turn`, and `--max-save-turn` filter rows before shard saving, so high-gap/decisive or mid/late Commander data can be collected without keeping every draw-heavy low-margin state. This is the replacement path for direct spatial rerank scale sweeps.

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/adaptive_plan_q_dataset.py 8 \
  --grid-sizes 8 \
  --num-steps 16 \
  --warmup-steps 0 \
  --num-shards 1 \
  --pool-size 128 \
  --model-path runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx \
  --network-arch unet \
  --channels 64,96,128,64 \
  --input-channels 35 \
  --scoreboard-history \
  --fog-memory \
  --value-heads per-size \
  --value-head-sizes 8,12,16 \
  --value-loss hl-gauss \
  --outcome-head \
  --strategy-aux \
  --strategy-spatial-aux \
  --source-count 4 \
  --target-count 4 \
  --plan-rollout-steps 8 \
  --rollouts-per-plan 1 \
  --output-dir runs/adaptive-plan-q-v0 \
  --shard-prefix plan-q
```

The first smoke shard is intentionally small. It validates the data path and produces non-uniform Plan-Q marginals, but does not yet provide anti-draw outcome labels because the 8-step rollout horizon remains nonterminal. Longer fixed-v5 max250 shards are the next data step.

For fixed-v5 max250 data, use `--truncation 250` and `--warmup-steps <n>` to advance behavior games before expensive plan scoring. Counterfactual plan rollouts are truncation-aware, so wins after the max-step gate are not counted as wins.

Use `--plan-worker-steps <n>` when first-step plan forcing only separates draw from loss. This keeps the forced source-target first move, then executes `n` additional target-conditioned worker moves before handing control back to the base policy. In the fixed-v5 warm190 probe, `--plan-worker-steps 16` produced a small number of winning plan labels where the first-step-only shard had none. The default is `0` for backward-compatible first-step scoring.

Train source/target heads from Plan-Q marginals with `adaptive_plan_q_supervised.py`:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/adaptive_plan_q_supervised.py \
  --dataset 'runs/adaptive-plan-q-v0-scale10/*.npz' \
  --network-arch unet \
  --channels 64,96,128,64 \
  --init-channels 64,96,128,64 \
  --input-channels 35 \
  --init-input-channels 35 \
  --global-context \
  --value-heads per-size \
  --init-value-heads per-size \
  --value-loss hl-gauss \
  --init-value-loss hl-gauss \
  --outcome-head \
  --init-outcome-head \
  --strategy-aux \
  --init-strategy-aux \
  --strategy-spatial-aux \
  --init-strategy-spatial-aux \
  --init-model-path runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx \
  --source-weight 0.5 \
  --target-weight 0.5 \
  --model-path runs/adaptive-plan-q-supervised-v0/generals-adaptive-plan-q-supervised-v0.eqx
```

This trainer defaults to frozen trunk/policy updates and only moves the source/target strategy heads. It is a data-quality and representation probe until the Plan-Q shards include longer-horizon fixed-v5 outcome labels.

For outcome/Q proposal-map supervision, the trainer also accepts
`--source-q-mse-weight`, `--target-q-mse-weight`, `--source-q-rank-weight`,
`--target-q-rank-weight`, `--plan-pair-rank-weight`,
`--q-rank-temperature`, and `--q-target-outcome-weight`. These losses recompute
source/target targets from the saved `plan_q` matrix, optionally blending
decisive plan outcomes into loss/draw/win values, then train candidate-local
source/target value rankings instead of only the precomputed source/target CE
marginals. `--plan-pair-rank-weight` ranks the full source-target candidate
matrix with additive `source_logit + target_logit` pair scores; it is a
diagnostic for whether decomposed proposal maps can recover the best plan
without adding an explicit Commander pair scorer.

The same trainer can train the strategy action-Q head directly from plan outcomes:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/adaptive_plan_q_supervised.py \
  --dataset runs/adaptive-plan-q-fixed-v5-worker-v2/plan-q-00000.npz \
  --init-model-path runs/adaptive-strategy-spatial-v1/generals-adaptive-strategy-spatial-v1.eqx \
  --network-arch unet \
  --channels 64,96,128,64 \
  --input-channels 35 \
  --global-context \
  --value-heads per-size \
  --init-value-heads per-size \
  --value-loss hl-gauss \
  --init-value-loss hl-gauss \
  --outcome-head \
  --init-outcome-head \
  --strategy-aux \
  --init-strategy-aux \
  --strategy-spatial-aux \
  --init-strategy-spatial-aux \
  --source-weight 0 \
  --target-weight 0 \
  --action-q-weight 1.0 \
  --action-q-mse-weight 0.1 \
  --gap-weighting \
  --model-path runs/adaptive-plan-q-action-q-v1/generals-adaptive-plan-q-action-q-v1.eqx
```

Plan action labels use the standard adaptive action index, including the single global pass index. The action-Q loss aggregates duplicate plan slots onto their shared primitive action before applying ranking CE, then can be probed with `evaluate_adaptive_policy.py --strategy-q-rerank-scale <scale>`. A more conservative diagnostic is `--strategy-q-replace-threshold <q_advantage>`, which keeps the sampled/greedy policy action unless the best legal strategy-Q action clears a predicted advantage threshold; `--strategy-q-replace-policy-margin <logit_margin>` can additionally require policy support.

`adaptive_plan_q_supervised.py --replacement-gate-weight` trains the same action-Q head as a pairwise accepted-replacement gate. It compares the best source-target plan action with the teacher/base action when both appear in the candidate set, using `--replacement-score-margin` to define same-outcome improvements and `--replacement-target-margin` as the desired Q gap. This is still a diagnostic path: the first fixed-v5 shard run reduced draw under conservative replacement but did not improve 256-row min win rate.

`evaluate_adaptive_policy.py --strategy-q-replace-worker-candidate` restricts the replacement gate to the single source/target worker action produced by the spatial heads instead of taking the best Q over every legal primitive action. Use it with `--strategy-q-replace-threshold` and `--strategy-aux --strategy-spatial-aux`. Early probes did not improve fixed-v5 max250 either, so this is useful mainly for separating action-Q calibration failures from candidate-generation failures.

`adaptive_plan_worker_supervised.py` trains a separate target-conditioned Worker from Plan-Q shards or strategy shards. It appends `source_one_hot`, `target_one_hot`, and `route_potential` command planes to the saved adaptive observation, then supervises the Worker policy with the selected plan action. For Plan-Q data, `--selection best` uses the best plan per row, `--selection accepted` keeps candidate plans that improve over the teacher/base action when that action is present in the candidate set, and `--selection mixed` combines both with configurable weights. For strategy data, `--dataset-format strategy` uses each shard's `source_heatmap`, `target_heatmap`, and `teacher_action_index`; this is useful for decisive rollout-search trajectory windows. `evaluate_adaptive_policy.py --strategy-plan-worker-path <worker.eqx> --strategy-plan-worker-rerank-scale <x>` can load that Worker and use its centered logits as a small bias. `--strategy-plan-worker-min-margin <margin>` gates the bias to states where the Worker's legal top-1/top-2 logit margin clears the threshold. By default the Worker command comes from strategy-spatial source/target heads; `--strategy-plan-worker-command-source belief-main-stack` instead uses the belief enemy-general heatmap as target and chooses the source by main-stack/route heuristic, so it only requires `--strategy-aux` rather than `--strategy-spatial-aux`. The first best-plan v0 run improved fixed-v5 max250 at 256 games/row with low scale `0.02`, while keeping Expander 256-row min roughly flat, but larger best-only, mixed, accepted-only, and decisive-strategy Workers did not confirm. Treat this as a diagnostic tool, not a promotion setting.

`adaptive_command_gate_supervised.py` trains an independent binary command-acceptance MLP from Plan-Q shards. It predicts whether to replace the current policy action with a source/target worker command using only inference-time features such as policy-logit delta, action-Q delta, source/target logits, finish probability, route distance, source army, and seat. `evaluate_adaptive_policy.py --strategy-command-gate-path <gate.eqx> --strategy-command-gate-threshold <p>` loads the gate; `--strategy-command-gate-source-count` and `--strategy-command-gate-target-count` let it score a top-k source by top-k target command set. Early model, model-worker, and multi-command probes fit offline labels or expose useful candidates, but fixed-v5 confirmation still regressed, so this is a diagnosis tool until source/target proposal quality improves.

The same gate trainer also supports `--dataset-format strategy-worker`. This reads decisive strategy shards, runs a learned `--plan-worker-path`, and labels the Worker's top-1 replacement as positive only when it matches the rollout-search teacher action on decisive `search_best_outcome == win` or optional `finish_within_250` rows. `evaluate_adaptive_policy.py --strategy-plan-worker-gate-path <gate.eqx> --strategy-plan-worker-gate-threshold <p>` then lets that gate decide whether the Plan-Worker replaces the current primitive action. The first belief-main-stack proxy gate trained on GPU but did not beat the no-gate fixed-v5 max250 baseline, so treat it as negative evidence against offline proxy gating rather than a promotion setting.

`adaptive_plan_pair_supervised.py` trains an explicit source-target pair scorer from Plan-Q shards. Unlike the additive `source_logit + target_logit` probe, it applies an MLP to inference-time pair features, including policy/action-Q deltas, source/target logits, finish probability, source/target cell features, coordinates, route distance, turn, grid size, and seat. It saves both the final checkpoint and a `.best.eqx` checkpoint selected by validation pair top-1 accuracy, and reports pair@1/pair@2/pair@4 so Commander candidate quality is not judged only by argmax. `--min-plan-gap <gap>` can filter out low-margin rows before splitting, which is useful because larger model-worker shards are draw-heavy and the rank signal concentrates in high-gap states. The first worker-source gap-weighted run improved validation pair top-1 over the additive same-split baseline; a larger same-family shard only held a weak edge without filtering, while `--min-plan-gap 0.25` and `0.5` restored stronger pair ranking on fewer rows. This remains an offline Commander-scorer probe until high-gap performance is stable across larger shards/seeds.

`adaptive_plan_pair_evaluate.py` replays the same Plan-Q pair dataset split and reports additive-baseline versus explicit-scorer metrics without training. Use it to confirm pair@1/pair@2/pair@4, source/target accuracy, and score correlation before deciding whether a scorer is good enough for shortlist diagnostics. The first high-gap recheck showed useful top-k signal but weak argmax quality, so it should not be wired directly into gameplay inference.

`--plan-policy-weight` uses the same aggregated Plan-Q action target to update policy logits directly. Use it only with `--update-scope all` and a positive `--policy-kl-weight`; early probes showed it can damage seat balance even when KL anchored.

`evaluate_adaptive_policy.py` also supports a target-conditioned probe that uses the strategy enemy-general belief head to bias legal moves toward the predicted target:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/_experimental/ppo/evaluate_adaptive_policy.py \
  runs/adaptive-strategy-q-rerank-v1/generals-adaptive-strategy-q-rerank-v1.eqx \
  --grid-sizes 8,12,16 \
  --network-arch unet \
  --channels 64,96,128,64 \
  --scoreboard-history \
  --fog-memory \
  --value-heads per-size \
  --value-head-sizes 8,12,16 \
  --value-loss hl-gauss \
  --outcome-head \
  --strategy-aux \
  --strategy-target-rerank-scale 0.5
```

For explicit source/target execution probes, `evaluate_adaptive_policy.py` also supports `--strategy-worker-mix-prob <p>`. This selects a source/target plan from the spatial heads and, with probability `p`, executes one legal target-conditioned worker move instead of the sampled policy action. `--strategy-worker-policy-margin <margin>` only permits worker moves whose base-policy logit is within `margin` of the current best action, and `--strategy-worker-finish-gate` scales the mix probability by the finish head. Current fixed-v5 probes regressed, so treat this as a diagnostic path rather than a promotion setting.

`--strategy-target-finish-gate` multiplies that target bias by the finish-head probability. This is an inference-only probe and does not change checkpoint weights; current evidence says it can move wins between rows, but has not passed 256-row confirmation.

## 验证

运行完整测试：

```bash
uv run pytest
```

编译检查：

```bash
uv run python -m compileall generals examples tests
```

提交前建议至少运行：

```bash
uv run pytest
git diff --check
git status
```

## 注意事项

- 4x4 训练命令主要用于 smoke test，不适合作为策略质量结论。
- 更有意义的实验建议使用 8x8 或更大 generated 地图，并在独立 seed 上批量评估。
- `.eqx` checkpoint 属于实验产物，建议放在项目内已忽略的 `runs/` 或其他非缓存实验目录，不要提交进 Git。
- `bench.py` 和 `examples/_experimental/benchmark_performance.py` 仍含旧版环境接口痕迹，使用前应先按当前 `reset(key) -> (pool, state)` 和 `step(state, actions, pool)` 接口修复。
