# Generals Bots 中文手册

本文面向希望快速运行、理解和扩展本项目的开发者与强化学习实验人员。项目提供了一个基于 JAX 的 Generals.io 双人对战模拟器，重点是高吞吐、可向量化、可复现实验，以及用于训练和评估 bot 的实验脚本。

## 1. 项目定位

`generals-bots` 是一个面向 Generals.io bot 研究的 Python 包。它把游戏核心逻辑写成纯函数式、JAX 友好的形式，使同一套环境既可以单局调试，也可以用 `jax.vmap` 同时推进大量对局。

项目适合做三类工作：

- 编写规则型 agent，例如随机移动或扩张型策略。
- 批量运行环境，用于性能测试、数据采样和策略评估。
- 训练实验性强化学习策略，目前仓库内提供 PPO、行为克隆、策略评估和可视化脚本。

当前仓库的主要特点：

- 核心模拟器位于 `generals/core/`，使用 JAX array 和不可变 `NamedTuple` 状态。
- 环境包装类 `GeneralsEnv` 支持固定地图和变尺寸/填充地图，并使用预生成 state pool 做快速 auto-reset。
- 内置 `RandomAgent` 和 `ExpanderAgent` 两个基线 agent。
- `examples/` 提供单局、向量化、GUI 可视化示例。
- `examples/_experimental/ppo/` 提供实验性训练、行为克隆和批量评估工具。

## 2. 目录结构

```text
.
├── generals/                    # Python 包主体
│   ├── core/                    # JAX 游戏逻辑、环境、动作、观测、地图生成、奖励
│   ├── agents/                  # Agent 抽象类和内置策略
│   ├── gui/                     # pygame GUI 和 replay 渲染
│   ├── remote/                  # generals.io 远程客户端相关代码
│   └── assets/                  # GUI 图片和字体资源
├── examples/                    # 用户示例
│   ├── simple_example.py        # 单局对战示例
│   ├── vectorized_example.py    # vmap 并行环境示例
│   ├── visualization_example.py # pygame 可视化示例
│   └── _experimental/           # 实验性 benchmark、PPO、策略可视化
├── tests/                       # pytest 测试
├── docs/                        # 文档和开发记录
├── pyproject.toml               # 包元数据、依赖、可选 extra
├── uv.lock                      # uv 锁文件
├── requirements.txt             # 传统 pip 依赖列表
└── README.md                    # 英文快速介绍
```

## 3. 核心概念

### 3.1 地图与格子编码

游戏地图是二维整数数组。核心生成器在 `generals/core/grid.py`：

- `-2`：mountain，不可通行。
- `0`：空地。
- `1`：玩家 0 的 general。
- `2`：玩家 1 的 general。
- `40-50` 附近的正整数：city，数值表示初始守军数量，具体范围可配置。

`generate_grid(...)` 会随机放置双方 general、mountain、city，并尽量保证地图连通和双方距离约束。训练脚本也支持 `simple` 地图，即只包含两个随机 general 的空地图，适合作为极快的 smoke test。

### 3.2 GameState

`GameState` 定义在 `generals/core/game.py`，包含完整隐藏状态：

- `armies`：每个格子的军队数量。
- `ownership`：形状为 `(2, H, W)` 的玩家占领掩码。
- `ownership_neutral`：中立格子掩码。
- `generals`、`cities`、`mountains`、`passable`：地图结构掩码。
- `general_positions`：双方 general 坐标。
- `time`：当前步数。
- `winner`：未结束为 `-1`，否则为获胜玩家编号。
- `pool_idx`：auto-reset 时从 state pool 取下一局的索引。

状态是不可变 `NamedTuple`，更新时通过 `_replace` 返回新状态，适合 JAX JIT 编译和批处理。

### 3.3 Observation

`Observation` 定义在 `generals/core/observation.py`。每个玩家只能看到战争迷雾下的局部信息，包括：

- `armies`
- `generals`
- `cities`
- `mountains`
- `neutral_cells`
- `owned_cells`
- `opponent_cells`
- `fog_cells`
- `structures_in_fog`
- `owned_land_count`
- `owned_army_count`
- `opponent_land_count`
- `opponent_army_count`
- `timestep`

`Observation.as_tensor()` 会把观测转换成神经网络可用的张量。单个观测输出 `(14, H, W)`；批量观测会保留批处理和玩家维度。

### 3.4 Action

动作是长度为 5 的整数数组：

```text
[pass, row, col, direction, split]
```

字段含义：

- `pass`：`1` 表示跳过本回合，`0` 表示移动。
- `row`、`col`：源格子坐标。
- `direction`：`0=上`，`1=下`，`2=左`，`3=右`。
- `split`：`1` 表示移动一半军队，`0` 表示移动除 1 个驻军外的全部军队。

常用工具函数：

```python
from generals import create_action, compute_valid_move_mask

action = create_action(to_pass=False, row=3, col=4, direction=1, to_split=False)
mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
```

`compute_valid_move_mask` 返回形状为 `(H, W, 4)` 的布尔数组，表示从每个格子朝四个方向移动是否合法。

### 3.5 GeneralsEnv

`GeneralsEnv` 是主要环境入口，定义在 `generals/core/env.py`。当前接口要点：

```python
import jax.random as jrandom
from generals import GeneralsEnv

env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
key = jrandom.PRNGKey(42)
pool, state = env.reset(key)
timestep, state = env.step(state, actions, pool)
```

需要注意：

- `reset(key)` 返回 `(pool, init_state)`，其中 `pool` 是预生成的批量初始状态池。
- `step(state, actions, pool)` 推进一步游戏；终局或达到 `truncation` 后，会从 `pool` 自动重置。
- `actions` 的形状是 `(2, 5)`，分别对应双方玩家。
- `TimeStep` 包含 `observation`、`reward`、`terminated`、`truncated`、`info` 和 `last_state`。

`GeneralsEnv` 支持两种地图模式：

```python
# 固定尺寸
env = GeneralsEnv(grid_dims=(10, 10), truncation=500)

# 变尺寸地图，pad 到统一大小以便批处理
env = GeneralsEnv(min_grid_size=8, max_grid_size=24, pad_to=24)
```

## 4. 快速搭建环境

### 4.1 前置条件

建议环境：

- Ubuntu 24.04 x86-64。
- Python 3.11 或更高版本。
- `uv`，用于创建可复现环境并按 `uv.lock` 安装依赖。
- 可选：NVIDIA GPU 和 CUDA 13 运行环境，用于大规模训练。

如果还没有安装 `uv`，可按 uv 官方安装方式安装。安装完成后确认：

```bash
uv --version
```

### 4.2 获取代码

```bash
git clone https://github.com/CodeBoy2006/generals-bots.git
cd generals-bots
```

如果是在当前开发机的已有仓库中工作，直接进入仓库目录即可：

```bash
cd /home/codeboy/research/generals-bots
```

### 4.3 CPU 开发环境

安装运行和开发依赖：

```bash
uv sync --extra dev
```

确认包能导入：

```bash
uv run python -c "import generals; print(generals.GeneralsEnv)"
```

确认 JAX 后端：

```bash
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

在 CPU 环境中，输出通常会显示 `cpu`。

### 4.4 CUDA 13 GPU 环境

如果机器有可用 NVIDIA GPU，并希望使用 CUDA 13 版本的 JAX 插件：

```bash
uv sync --extra dev --extra cuda13
```

验证 GPU：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

期望看到 `gpu` 和 `CudaDevice(...)`。`XLA_PYTHON_CLIENT_PREALLOCATE=false` 可以避免 JAX 一启动就预占大量显存，适合与其他任务共用 GPU。

### 4.5 传统 pip 安装方式

如果不使用 `uv`，也可以用可编辑安装：

```bash
python3 -m pip install -e .
```

若要运行测试，还需要安装开发依赖：

```bash
python3 -m pip install -e ".[dev]"
```

项目内的推荐命令仍以 `uv run ...` 为准，因为它更贴近当前锁文件和实验脚本。

## 5. 基础实验

### 5.1 单局对战

运行随机策略对扩张策略的单局对战：

```bash
uv run python examples/simple_example.py
```

该脚本会：

1. 创建 `GeneralsEnv(grid_dims=(10, 10), truncation=500)`。
2. 调用 `env.reset(key)` 得到 `pool` 和初始 `state`。
3. 每步分别取两个玩家的 `Observation`。
4. 由 `RandomAgent` 和 `ExpanderAgent` 产生动作。
5. 调用 `env.step(state, actions, pool)` 直到终局或截断。

适合用于确认环境安装和核心 API 是否可用。

### 5.2 并行环境

运行向量化示例：

```bash
uv run python examples/vectorized_example.py
```

该脚本使用 `jax.vmap` 批量推进多个环境。核心思想是：

- 用一次 `env.reset(pool_key)` 生成共享 reset pool。
- 用 `jax.vmap(env.init_state)` 生成多个初始状态。
- 对 `get_observation`、agent `act` 和 `env.step` 分别做 `vmap`。

并行环境是本项目做 RL rollout 和大规模评估的基础。

### 5.3 GUI 可视化

运行 pygame 可视化：

```bash
uv run python examples/visualization_example.py
```

该示例会显示一局 `RandomAgent` 对 `ExpanderAgent` 的游戏过程。若在无显示服务器的远程机器上运行，pygame 窗口可能无法打开；此时优先运行非 GUI 示例或使用本地桌面/远程显示转发。

## 6. 性能实验

### 6.1 吞吐实验入口

当前最稳妥的吞吐验证入口是向量化示例：

```bash
uv run python examples/vectorized_example.py
```

它会在同一进程内并行运行 256 个 10x10 环境，并定期输出双方平均占地。若需要做正式吞吐 benchmark，可以基于这个脚本扩展计时代码：把多步 rollout 包进 `jax.jit`/`jax.lax.scan`，并在计时结束前同步设备结果。

性能测试应注意：

- JAX 第一次运行会触发 JIT 编译，不能把编译时间直接当成稳态吞吐。
- 对 GPU/TPU 计时要在结果上调用 `block_until_ready()`。
- 比较不同设置时，应固定地图尺寸、并行环境数量、步数和硬件后端。

### 6.2 旧 benchmark 脚本

仓库中还有 `bench.py` 和 `examples/_experimental/benchmark_performance.py`，目标是测量环境吞吐。不过它们包含旧版 `GeneralsEnv` 接口痕迹，使用前应先按当前 `reset(key) -> (pool, state)` 和 `step(state, actions, pool)` 接口修复。新实验建议先从 `examples/vectorized_example.py` 派生。

## 7. PPO 与策略实验

实验性训练代码位于：

```text
examples/_experimental/ppo/
```

如果目标是复现或继续推进“对 Expander 胜率超过 90%”的训练路线，先阅读专门指南：[Expander 对抗训练流程与策略](expander-training-strategy.md)。该指南记录了行为克隆 warm start、PPO-vs-Expander fine-tune、多 seed 镜像评估和最终 checkpoint 的验收口径。

主要脚本：

- `train.py`：基于 raw game API 的 PPO 训练路径，当前推荐作为快速实验入口。
- `train2.py`：基于 `GeneralsEnv` 包装的 PPO 训练路径。
- `behavior_clone.py`：从 Expander teacher 做行为克隆。
- `evaluate_policy.py`：批量评估保存的 `.eqx` 策略。
- `behavior_clone_adaptive.py`：训练固定 padding 画布的自适应多尺寸行为克隆 warm start。
- `train_adaptive.py`：训练一个可在多个有效棋盘尺寸上运行的 adaptive PPO checkpoint。
- `evaluate_adaptive_policy.py`：按尺寸和座位矩阵评估 adaptive checkpoint。
- `network.py`：Equinox 策略价值网络。
- `common.py`：地图生成、动作编码、策略动作选择等共享工具。

### 7.1 4x4 smoke 训练

先跑一个很小的训练，验证依赖、JAX 后端和模型保存流程：

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --num-steps 64 \
  --num-iterations 10 \
  --model-path runs/generals-ppo-4x4.eqx
```

4x4 只适合作为 smoke test，不适合做策略质量结论。

### 7.2 8x8 简单地图 PPO

```bash
uv run python examples/_experimental/ppo/train.py 64 \
  --grid-size 8 \
  --num-steps 64 \
  --num-iterations 10 \
  --model-path runs/generals-ppo-8x8-simple.eqx
```

`simple` 是默认地图生成器，只放置两个 general，训练速度快，但缺少 mountain 和 city。

### 7.3 8x8 generated 地图 PPO

更接近实际环境的实验应使用 generated 地图：

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

常用参数说明：

- `--grid-size`：方形地图尺寸。
- `--map-generator simple|generated`：空地图或带 terrain 的地图。
- `--pool-size`：预生成 reset state 数量，必须至少等于并行环境数量。
- `--truncation`：单局最大步数。
- `--mountain-density-min/max`：mountain 密度范围。
- `--num-cities-min/max`：city 数量范围。
- `--min-generals-distance`：双方 general 最小距离；未设置时训练脚本会取 `max(3, grid_size // 2)`。
- `--model-path`：Equinox `.eqx` checkpoint 保存路径。

### 7.4 GPU 训练命令模板

在 CUDA 机器上，建议显式指定后端：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 128 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --num-steps 128 \
  --num-iterations 100 \
  --pool-size 2048 \
  --model-path runs/generals-ppo-8x8-gpu.eqx
```

如果显存不足，优先降低：

- 并行环境数量，也就是位置参数 `128`。
- `--num-steps`。
- `--pool-size`。
- `--grid-size`。

### 7.5 行为克隆 warm start

仓库已有从 randomized Expander teacher 学习的行为克隆脚本，适合作为 PPO 或 self-play 的 warm start：

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

输出模型默认建议放在项目内已忽略的 `runs/` 或其他非缓存实验目录，不要直接提交 `.eqx` checkpoint。

### 7.6 checkpoint 与 current-policy 自博弈

`train.py` 支持两类自博弈。冻结 checkpoint 自博弈会用 `--opponent-policy-path` 指定非 learner 玩家；current-policy 自博弈会用 `--self-play-opponent` 让非 learner 玩家在每轮 rollout 中使用当前正在更新的同一个 policy。

冻结 checkpoint 自博弈：

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

current-policy 自博弈：

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

`--self-play-opponent` 不能和 `--opponent-policy-path` 或 `--opponent-policy-pool` 同时使用。`--opponent-policy-pool a.eqx,b.eqx` 会让普通 PPO 从多个同架构 frozen checkpoint 中采样对手，适合 checkpoint league best-response；可用 `--opponent-policy-pool-modes sample,greedy` 指定各自执行模式。`--checkpoint-dir`、`--checkpoint-every` 和 `--keep-checkpoints` 可周期保存并保留中间模型，便于后续 league 评估选模。`--learner-player 0|1` 控制被更新的 learner 座位；`--terminal-reward-scale` 会在 decisive terminal transition 上额外加入胜负奖励。`--general-target-reward-scale` 会用完整状态奖励强兵靠近敌方 general 的势能变化，可配合 `--general-target-min-army` 和 `--general-target-max-distance` 控制触发条件。`--path-assignment-reward-scale` 会在 reward 内缓存 passable shortest-path 距离场，并把强兵分配到敌方 general、非己方城市或前线目标，可用 `--path-assignment-*-weight` 控制目标优先级。`--policy-input augmented-full-state` 可让 PPO learner 使用 18 通道输入；从 v5 这类 9 通道 checkpoint 开始时通常配合 `--init-input-channels 9` 使用。冻结 opponent 更稳定，适合作为后续 checkpoint league 的基础；current-policy opponent 适合快速检验同步自博弈方向。

Residual GRU 记忆 PPO 可用 `train_recurrent.py` 训练。它在 CNN policy 上叠加 GRU hidden state 和 residual logits/value delta；`--freeze-base` 会冻结 warm-start 的 CNN，只训练记忆适配器，适合保护 v5 或行为克隆基线。对应评估入口是 `examples/_experimental/ppo/evaluate_recurrent_policy.py`。

### 7.7 单 checkpoint 自适应多尺寸 PPO

如果目标是一个 checkpoint 同时覆盖 8x8、12x12 和 16x16，可以使用 adaptive PPO 路径。它把所有局面 pad 到固定 `--pad-to` 画布，并额外输入 active-cell、padding 和尺寸坐标通道；动作空间使用同一个 `8 * pad_to * pad_to + 1` 展平空间，最后一个 logit 是全局 pass。

先做 Expander-soft warm start：

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

再对 Expander 做 PPO fine-tune，并周期保存候选：

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

验收时必须同时测每个尺寸和两个座位：

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

`evaluate_adaptive_policy.py` 输出六行结果：3 个尺寸乘以 2 个座位。`min_win_rate` 是最弱一行的总胜率；draw 不计为 win。当前 adaptive checkpoint 使用 `AdaptivePolicyValueNetwork`，不兼容固定尺寸 GUI、`evaluate_policy.py` 或普通 `PolicyValueNetwork` checkpoint。
adaptive BC 和 PPO 训练器都支持 `--channels` 指定网络容量，也支持 `--grid-size-weights` 对困难尺寸过采样；PPO 训练器还支持 `--init-channels` 从不同容量的 adaptive checkpoint 零填充扩容 warm start。`--learner-player alternate` 在同一 run 内按 iteration 交替训练两个座位；`--learner-player mixed` 会把总 `num_envs` 拆给 player 0 和 player 1，并把两侧轨迹放进同一个 PPO update，适合减少按轮次交替造成的座位遗忘。`--reward-mode terminal` 会关闭 dense composite reward，只保留 terminal win/loss；`--gamma`、`--gae-lambda`、`--top-advantage-fraction`、`--ema-decay` 和 `--eval-ema` 用于 v3-noarch 长 rollout/EMA 训练。`--value-loss hl-gauss --value-bins 128 --value-sigma 0.04` 会把 PPO value loss 从 scalar MSE 切换为 HL-Gauss categorical CE。`--outcome-aux-weight` 会启用 loss/draw/win 辅助头，只用同一 rollout 内已知结局的 episode segment 做监督；`--global-context` 会把 land/army/time scoreboard 标量追加为 20 通道输入，`--scoreboard-history` 进一步追加 previous scoreboard 和 one-step delta，形成 30 通道输入，并通过零初始化上下文分支 warm start 旧 15 通道 checkpoint，旧 checkpoint 通常配 `--init-input-channels 15`。`adaptive_search_distill.py` 也支持同样的 `--global-context` / `--scoreboard-history` student 输入扩展；`--base-global-context` / `--base-scoreboard-history` 可让 base/search/opponent teacher 使用 20/30 通道 shared-MSE adaptive checkpoint，不传时 teacher 仍按旧 15 通道 checkpoint 作为 KL anchor 和 search prior。它的 `--learner-player mixed` 会在同一个 distill update 内拼接 p0+p1 搜索标签 batch，`--freeze-legacy-weights` 则只训练新增输入路径，冻结旧 15 通道 trunk/head，`--freeze-strategy-aux-only` 只训练 strategy auxiliary heads 而不扰动 policy/trunk/global/value，`--search-value-weight` 会额外用 top-k rollout-search 分数的 bounded value target 监督 value/shared representation，`--search-outcome-weight` 会用 best search rollout 的 loss/draw-or-unfinished/win 标签监督 outcome head，`--strategy-q-weight`、`--strategy-intent-weight`、`--strategy-finish-weight` 和 `--strategy-belief-weight` 会启用 top-k Q、弱 intent、finish 和敌方 general belief 辅助头；从已有 strategy checkpoint 续训时用 `--init-strategy-aux`。`adaptive_worker_pretrain.py` 会把 target heatmap、eligible source heatmap 和 BFS route potential 加到 18 通道 Worker 输入中，用 full-state BFS/path-assignment 生成 soft action targets；`evaluate_worker_policy.py` 可以用 fogged observation command wrapper 单独评估 Worker，也可以用 fallback adaptive checkpoint 做混合接管评估，但这些 checkpoint 仍是 Commander/Worker 实验件，不能直接当作 `evaluate_adaptive_policy.py` 的 promotion 策略。 如果 checkpoint 使用 categorical/per-size value head、outcome head、strategy aux、global context 或 scoreboard history，评估时也要给 `evaluate_adaptive_policy.py` 传匹配的 `--value-heads`、value-loss 模板参数、`--outcome-head`、`--strategy-aux`、`--global-context` 或 `--scoreboard-history`。

下一轮 adaptive PPO v3-outcome continuation 建议用 GPU 从当前最强候选启动；本地 16GB GPU 已验证 512 envs x 256 steps 会 OOM，先用 256 envs 和 1024 minibatch 做 256 games/row triage：

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

### 7.8 批量评估 checkpoint

评估行为克隆或 PPO checkpoint：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py runs/generals-bc-8x8-soft.eqx \
  --num-games 2048 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500 \
  --opponent random \
  --policy-mode sample
```

关键输出：

- `Wins/Losses/Draws`
- `Win rate`
- `Decisive win rate`
- `Draw rate`
- `Mean final time`
- `Eval seconds`

实验结论应优先基于多 seed、多批次评估，而不是单次训练日志。

评估两个 PPO checkpoint 对战时：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py runs/generals-ppo-candidate.eqx \
  --opponent-policy-path runs/generals-ppo-best-frozen.eqx \
  --opponent-policy-mode sample \
  --num-games 2048 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500 \
  --policy-mode sample \
  --policy-player 0
```

### 7.9 可视化训练好的策略

可视化 `.eqx` 模型：

```bash
uv run python examples/_experimental/visualize_policy.py runs/generals-ppo-8x8-generated.eqx 10 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5
```

可视化时应保持 `--grid-size` 和地图生成参数与训练 checkpoint 兼容，否则网络尺寸或输入分布可能不匹配。

### 7.10 玩家对战训练好的策略

可以用本地 pygame 窗口和 `.eqx` PPO checkpoint 对战：

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

也可以直接观看 PPO 机器对战：

```bash
uv run python examples/play_against_model.py \
  --machine-vs-machine \
  --model-0-path runs/generals-ppo-a.eqx \
  --model-1-path runs/generals-ppo-b.eqx \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode sample \
  --opponent-policy-mode sample \
  --tick-rate 4
```

若要在 GUI 中使用 `v5 + rollout-search`，直接用脚本环境变量打开搜索：

```bash
SEARCH_POLICY=1 ./play-v5.command
```

观看机器对战时，可只让 player 0 使用搜索：

```bash
MODEL_0_SEARCH_POLICY=1 ./watch-v5.command
```

也可以让双方都使用搜索：

```bash
MODEL_0_SEARCH_POLICY=1 MODEL_1_SEARCH_POLICY=1 ./watch-v5.command
```

rollout-search 会在每个真实动作前模拟 top-k 候选动作，默认
`SEARCH_TOP_K=4`、`SEARCH_ROLLOUT_STEPS=16`、`SEARCH_ROLLOUTS_PER_ACTION=4`，
因此比普通 v5 慢很多。需要更流畅观察时，可以先降低预算：

```bash
SEARCH_POLICY=1 \
SEARCH_TOP_K=2 \
SEARCH_ROLLOUT_STEPS=8 \
SEARCH_ROLLOUTS_PER_ACTION=2 \
./play-v5.command
```

当前 GUI 搜索只支持 9 通道 observation checkpoint；v5 checkpoint 可直接使用。

控制方式：

- 左键点击自己的可移动格子作为源格，再点击相邻目标格提交移动。
- `S` 切换下一步是否 split/半兵移动。
- `P` 跳过本回合。
- 右键或 `Esc` 取消当前选中。
- 终局或达到 `--max-steps` 后按 `R` 重开，`Q` 或关闭窗口退出。
- 选中的源格会显示黄色边框，可移动目标格会显示绿色边框。
- 右侧面板会显示当前选择、split 状态和最近一次点击结果。
- 自动 tick 默认开启，没有人类动作时会自动 pass 并推进回合；`--no-auto-tick` 可关闭，`--tick-rate` 控制每秒自动推进次数。
- 默认会展示 PPO 模型的下一步 Top-K 候选动作：棋盘上标出候选源格/目标格/箭头，右侧面板列出概率和 value。
- `--preview-top-k` 可设置展示 1-5 个候选，`--no-ai-preview` 可关闭预览。
- `--policy-mode sample` 时预览显示的是动作概率分布，实际动作仍按概率抽样；`greedy` 模式通常执行概率最高的候选。
- `--machine-vs-machine` 会让两个 PPO agent 自动对战；`--model-0-path` 和 `--model-1-path` 可分别指定两个 checkpoint，`--opponent-policy-mode` 可设置第二个 agent 的策略模式。
- `--opponent-model-path` 仍可作为 `--model-1-path` 的兼容别名。

该入口只支持当前 PPO `PolicyValueNetwork` 保存出的 Equinox `.eqx` checkpoint。`--grid-size` 必须和训练/保存模型时的网络尺寸一致，否则会加载失败或在推理时因输入尺寸不匹配报错。checkpoint 通常较大且属于实验产物，建议放在项目内已忽略的 `runs/` 或专门的非缓存实验目录，不要提交进 Git。

## 8. 编写自己的 Agent

自定义 agent 需要继承 `generals.agents.agent.Agent` 并实现 `act(observation, key)`：

```python
import jax.numpy as jnp
from generals.agents.agent import Agent
from generals.core.action import compute_valid_move_mask


class FirstMoveAgent(Agent):
    def act(self, observation, key):
        mask = compute_valid_move_mask(
            observation.armies,
            observation.owned_cells,
            observation.mountains,
        )
        moves = jnp.argwhere(mask, size=mask.size, fill_value=-1)
        num_valid = jnp.sum(jnp.all(moves >= 0, axis=-1))
        move = moves[0]
        pass_action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
        move_action = jnp.array([0, move[0], move[1], move[2], 0], dtype=jnp.int32)
        return jnp.where(num_valid > 0, move_action, pass_action)
```

为了保持 JAX 兼容性，agent 的 `act` 最好：

- 使用 `jax.numpy` 而不是普通 Python list 运算。
- 避免依赖可变全局状态。
- 对随机性使用传入的 `key`，不要复用同一个 PRNG key。
- 返回固定形状的 `jnp.ndarray`。

## 9. 测试与验证

运行完整测试：

```bash
uv run pytest
```

只运行地图生成相关测试：

```bash
uv run pytest tests/test_grid_generation_performance.py
```

编译检查常用命令：

```bash
uv run python -m compileall generals examples tests
```

提交前建议至少运行：

```bash
uv run pytest
git diff --check
git status
```

若改动涉及训练脚本，还应额外跑一个很小的 smoke train，例如：

```bash
uv run python examples/_experimental/ppo/train.py 2 \
  --num-steps 2 \
  --num-iterations 1 \
  --pool-size 8 \
  --model-path runs/generals-ppo-smoke.eqx
```

## 10. 实验建议

推荐从小到大推进：

1. 先运行 `examples/simple_example.py` 确认环境可用。
2. 再运行 `examples/vectorized_example.py` 确认 JAX 批处理正常。
3. 用 4x4 PPO smoke test 检查训练脚本。
4. 切到 8x8 generated 地图，固定 terrain 参数做短训练。
5. 用 `evaluate_policy.py` 在独立 seed 上批量评估。
6. 用 `visualize_policy.py` 观察策略是否出现明显无效行为。
7. 增加并行环境数、rollout 步数、迭代次数和地图尺寸。

做严肃对比时应记录：

- Git commit。
- JAX 后端和设备。
- 地图尺寸与生成参数。
- 训练命令完整参数。
- checkpoint 路径。
- 评估命令、seed、局数和最大步数。
- 胜/负/平、总胜率、decisive win rate、draw rate。

`docs/devlogs/` 中已有若干英文实验记录，可以作为记录格式参考。

## 11. 常见问题

### 11.1 README 里的部分代码和当前接口不一致怎么办？

以仓库源码和 `examples/` 下可运行脚本为准。当前 `GeneralsEnv.reset(key)` 返回 `(pool, state)`，`GeneralsEnv.step(...)` 需要传入 `pool`。

### 11.2 为什么要有 state pool？

JAX JIT 适合静态形状和函数式数据流。预生成 state pool 后，终局 auto-reset 可以通过数组索引完成，避免在每个 step 内重新生成复杂地图，也减少 JIT 重编译风险。

### 11.3 为什么 benchmark 第一次运行慢？

第一次运行会触发 JAX/XLA 编译。性能比较应忽略 warmup，并在计时结束前同步设备结果。

### 11.4 训练模型保存在哪里？

示例命令使用 `runs/*.eqx`。这类 checkpoint 通常较大且属于实验产物，`runs/` 已被 Git 忽略，不建议提交进 Git。

### 11.5 4x4 结果能说明策略强吗？

不能。4x4 主要用于 smoke test。更有意义的实验至少应使用 8x8 generated 地图，并在独立地图上批量评估。

### 11.6 CPU 可以训练吗？

可以，但大规模训练会慢很多。CPU 更适合安装验证、小规模 smoke test、单局调试和文档示例。
