# Generals Bots

`generals-bots` 是一个基于 JAX 的 Generals.io 双人对战模拟器与 bot 研究框架。项目包含批量环境、规则 agent、PPO/BC 训练脚本、评测工具、pygame 可视化以及可互动的浏览器对局服务。

README 只保留入口信息和常用命令。完整实验路线、长命令、训练日志和参数说明请看项目文档。

## 文档入口

- 完整中文手册：[docs/zh-manual.md](docs/zh-manual.md)
- Expander/多尺寸训练路线与结果：[docs/expander-training-strategy.md](docs/expander-training-strategy.md)
- 项目状态流水账：[statusquo.md](statusquo.md)
- 早期训练与实验日志：[docs/devlogs/](docs/devlogs/)

## 项目结构

- `generals/`：核心环境、规则 bot、策略包装器和训练工具。
- `examples/`：训练、评测、可视化、Web 服务和数据生成入口。
- `tests/`：环境、agent、训练脚本和 Web 交互的回归测试。
- `docs/`：中文手册、训练策略、实验日志和分析文档。
- `runs/`：本地训练输出目录，默认不进入版本控制。

## 环境准备

推荐使用 `uv` 创建和运行项目环境：

```bash
uv sync --extra dev
```

如需 GPU/JAX 后端，请按本机 CUDA/JAX 版本单独配置。项目默认命令都通过 `uv run` 执行。

## 快速运行

规则 agent 对局：

```bash
uv run python examples/simple_example.py
```

批量环境 smoke test：

```bash
uv run python examples/vectorized_example.py
```

pygame 可视化：

```bash
uv run python examples/visualization_example.py
```

## 浏览器对局

启动 Web 服务后，在浏览器里打开终端输出的地址即可互动：

```bash
uv run python examples/play_web.py generals-ppo-8x8-expander-gpu-v5.eqx --host 127.0.0.1 --port 8765
```

Web 服务支持 8x8、16x16 等地图、Human/Model 控制切换、live stream、规则 agent、PPO checkpoint 以及 champion wrapper。复杂策略组合和完整启动参数见 [docs/zh-manual.md](docs/zh-manual.md)。

## 当前模型状态

- 固定 8x8：`generals-ppo-8x8-expander-gpu-v5.eqx` 是当前主要 checkpoint，采样评测对 Expander 已达到稳定优势，贪心评测较弱。
- 自适应多尺寸：`adaptive-unet-ppo-v4` 是当前多尺寸基础模型，采用 U-Net 风格策略网络，支持静态转换适配器和在线搜索包装。
- Web champion：当前强策略组合为 `adaptive-unet-ppo-v4 + static conversion adapter v1 + online search top4/r16/rpa2`，默认只在满足门控条件时启用昂贵搜索。
- 纯 checkpoint 压缩、搜索蒸馏和更大地图优势仍是后续研究重点。

模型文件和训练产物通常较大，默认作为本地 artifact 管理，不建议直接提交到 git。

## 常用训练入口

README 不再维护长实验命令。下面只列脚本入口，完整 recipe 见中文手册和训练策略文档。

- 固定尺寸 BC/PPO：`examples/_experimental/ppo/behavior_clone.py`、`examples/_experimental/ppo/train.py`、`examples/_experimental/ppo/evaluate_policy.py`
- 自适应多尺寸：`examples/_experimental/ppo/behavior_clone_adaptive.py`、`examples/_experimental/ppo/train_adaptive.py`、`examples/_experimental/ppo/evaluate_adaptive_policy.py`
- 教师数据与搜索：`examples/_experimental/ppo/adaptive_teacher_imitation.py`、`examples/_experimental/ppo/adaptive_strategy_supervised.py`、`examples/_experimental/ppo/adaptive_online_search_trace_dataset.py`
- 搜索/蒸馏：`examples/_experimental/ppo/search_policy.py`、`examples/_experimental/ppo/conservative_search_distill.py`

## 开发校验

常用回归检查：

```bash
uv run pytest -q
uv run python -m compileall generals examples tests
git diff --check
```

文档-only 改动通常只需要 `git diff --check` 和人工检查链接/命令是否仍然准确。

## Artifact 约定

- `.eqx`、`runs/`、大规模 replay/dataset 和临时评测输出默认忽略。
- 小型配置、文档、脚本和可复现实验说明应提交。
- 重要训练结论写入 `docs/expander-training-strategy.md` 或 `statusquo.md`，不要把完整原始日志塞回 README。
