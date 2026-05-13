# Large Map Heuristic Optimization

Date: 2026-05-13

## Summary

I extended the heuristic pool so it is not only usable on 8x8 maps, but also better aligned with larger generated maps such as 12x12 and 16x16. The main goal was to reduce the local-myopia that small-map heuristics tend to develop when the board becomes larger and longer-horizon.

## What Changed

### Map-scale-aware heuristics

File: `generals/agents/_heuristic_logic.py`

I refactored the heuristic scoring around a few shared, map-aware features:

- map scale factor
- large-map activation factor
- frontier masks
- full-board distance maps for cities, enemies, generals, and frontier targets

The heuristics now scale their attraction terms with board size, instead of using fixed 8x8-oriented constants. In practice, that means:

- expansion gets stronger on larger maps,
- city and general pursuit gain more weight when the board is larger,
- defensive play becomes less cramped and more frontier-aware,
- mixed selection shifts toward the more global strategies on large boards.

### Larger-map test coverage

File: `tests/test_heuristic_agents.py`

Added a 12x12 smoke test for every heuristic so the JAX path is exercised on a larger board, not just the 8x8 baseline.

### Documentation

File: `examples/_experimental/README.md`

Added a note that the heuristic pool is intended for larger boards such as 12x12 and 16x16, with longer evaluation horizons.

## Verification

Passed:

- `uv run python -m compileall generals/agents examples/_experimental/ppo tests`
- `uv run pytest -q`
- smoke runs for heuristic teacher and PPO opponent wiring on CUDA

## Notes

The heuristics are now better suited for larger maps, but the practical strength still depends on the generated terrain distribution and the evaluation horizon. Longer boards need longer time limits before the new distance-aware logic can pay off.

## Large-Map Evaluation

All runs used generated maps with:

- mountain density `0.12-0.22`
- city count `4-8`
- minimum general distance `5`
- seed `123`
- 256 games per condition

### 12x12, 500 Steps, Versus Random

- `expander`: 132/1/123, win rate `0.5156`, draw rate `0.4805`
- `general-hunter`: 56/0/200, win rate `0.2188`, draw rate `0.7812`
- `balanced`: 34/0/222, win rate `0.1328`, draw rate `0.8672`
- `mixed`: 65/0/191 after tuning, win rate `0.2539`, draw rate `0.7461`

### 16x16, 500 Steps, Versus Random

- `expander`: 72/0/184, win rate `0.2812`, draw rate `0.7188`
- `general-hunter`: 29/0/227, win rate `0.1133`, draw rate `0.8867`
- `balanced`: 18/0/238 after tuning, win rate `0.0703`, draw rate `0.9297`
- `mixed`: 40/0/216 after tuning, win rate `0.1562`, draw rate `0.8438`

### 16x16, 1000 Steps, Versus Random

- `expander`: 157/0/99, win rate `0.6133`, draw rate `0.3867`
- `mixed`: 83/0/173, win rate `0.3242`, draw rate `0.6758`

## Interpretation

The optimization helped the mixed heuristic on larger maps, but it did not make the new heuristic family stronger than Expander. The strongest practical conclusion is:

- Expander remains the best single baseline for decisive wins.
- Mixed is now more suitable as a diverse large-map teacher/opponent than before, especially with longer horizons.
- 16x16 maps should not be judged at only 500 steps; 1000 steps gives a much clearer signal.
