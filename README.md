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

也可以手动指定两个 checkpoint：

```bash
uv run python examples/play_against_model.py \
  --machine-vs-machine \
  --model-0-path /tmp/generals-ppo-a.eqx \
  --model-1-path /tmp/generals-ppo-b.eqx \
  --model-0-policy-input augmented-full-state \
  --model-1-policy-input observation \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode sample \
  --opponent-policy-mode sample \
  --tick-rate 4
```

手动指定 checkpoint 和参数：

```bash
uv run python examples/play_against_model.py /tmp/generals-ppo-8x8-generated.eqx \
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

加载 checkpoint 时，`--grid-size` 必须与保存该 `.eqx` 模型时使用的网络尺寸一致。`.eqx` 属于实验产物，建议放在 `/tmp` 或其他实验目录，不要提交进 Git。

4x4 PPO smoke test：

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --num-steps 64 \
  --num-iterations 10 \
  --model-path /tmp/generals-ppo-4x4.eqx
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
  --model-path /tmp/generals-ppo-8x8-generated.eqx
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
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v4.eqx \
  --model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx
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
  --init-model-path /tmp/generals-ppo-current.eqx \
  --opponent-policy-path /tmp/generals-ppo-best-frozen.eqx \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 1.0 \
  --model-path /tmp/generals-ppo-selfplay-next.eqx
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
  --init-model-path /tmp/generals-ppo-current.eqx \
  --self-play-opponent \
  --opponent-policy-mode sample \
  --learner-player 0 \
  --terminal-reward-scale 1.0 \
  --model-path /tmp/generals-ppo-current-selfplay.eqx
```

`--self-play-opponent` 会让非 learner 玩家在每次 rollout 中使用当前正在更新的同一个 policy；它不能和 `--opponent-policy-path` 同时使用。`--learner-player` 可以把 learner 放在 player 0 或 player 1；`--terminal-reward-scale` 会在 decisive terminal transition 上额外加入零和胜负奖励。`--general-target-reward-scale` 会用完整状态奖励强兵靠近敌方 general 的势能变化，可配合 `--general-target-min-army` 和 `--general-target-max-distance` 控制触发条件。`--path-assignment-reward-scale` 会在 reward 内缓存 passable shortest-path 距离场，并把强兵分配到敌方 general、非己方城市或前线目标，可用 `--path-assignment-*-weight` 控制目标优先级。`--policy-input augmented-full-state` 可让 PPO learner 使用 18 通道输入，通常与 `--init-input-channels 9` 一起从 v5 这类 9 通道 checkpoint 扩展。如果候选模型和冻结对手使用不同网络容量，可用 `--channels` 和 `--opponent-channels` 分别指定四层卷积通道，例如 `--channels 64,64,64,32 --opponent-channels 32,32,32,16`。

Residual GRU 记忆 PPO：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train_recurrent.py 512 \
  --grid-size 8 \
  --map-generator generated \
  --opponent-policy-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --hidden-size 64 \
  --freeze-base \
  --model-path /tmp/generals-recurrent-ppo-8x8-v5.eqx
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
  --init-model-path /tmp/generals-ppo-current.eqx \
  --opponent-policy-path /tmp/generals-ppo-best-frozen.eqx \
  --policy-mode sample \
  --opponent-policy-mode sample \
  --winner-source both \
  --negative-weight 0.0 \
  --model-path /tmp/generals-ppo-outcome-clone.eqx
```

`outcome_clone.py` 会完整 rollout 对局，并只用最终胜者视角的动作做监督样本；`--winner-source learner` 只保留 learner 赢局，`--negative-weight` 可额外压低败者动作概率。

rollout-search 强辅助评估：

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
  --rollouts-per-action 4
```

该脚本不训练新 checkpoint；它把 checkpoint 作为 policy prior，并对 top-k 候选动作做短 rollout 评分，可作为强评估策略或后续蒸馏 teacher。

保守 rollout-search 蒸馏：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/conservative_search_distill.py 128 \
  --base-model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --target-mode soft \
  --num-steps 64 \
  --num-iterations 80 \
  --min-margin 1 \
  --margin-scale 4 \
  --improve-weight 0.02 \
  --kl-weight 1.0 \
  --lr 0.000001 \
  --model-path /tmp/generals-ppo-8x8-conservative-search.eqx
```

该脚本用固定 base checkpoint 做 rollout-search teacher，并用 KL 约束学生贴近 base。`--target-mode hard` 只在 search 最优动作明显优于 base top-prior 动作时加入小权重动作监督；`--target-mode soft` 会把 top-k rollout 分数转成软目标，避免把近似并列候选强行压成单标签。`--policy-input full-state` 会用 privileged 完整状态替换标准 observation；`--policy-input augmented-full-state` 会保留原 9 个 observation 通道，并追加 9 个 full-state 通道。augmented 模式默认使用 18 输入通道，并支持从 9 通道 checkpoint 自动扩展 conv1 权重。评估时也要给 `evaluate_policy.py` 传同名 `--policy-input`，必要时传 `--input-channels` 和 `--opponent-input-channels`。它适合继续研究 search distillation，不应把训练 loss 当成棋力指标；仍需用 `evaluate_policy.py --opponent-policy-path` 独立评估。

行为克隆 warm start：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 2000 \
  --lr 0.0007 \
  --model-path /tmp/generals-bc-8x8-soft.eqx
```

批量评估 checkpoint：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
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
- `.eqx` checkpoint 属于实验产物，建议放在 `/tmp` 或其他实验目录，不要提交进 Git。
- `bench.py` 和 `examples/_experimental/benchmark_performance.py` 仍含旧版环境接口痕迹，使用前应先按当前 `reset(key) -> (pool, state)` 和 `step(state, actions, pool)` 接口修复。
