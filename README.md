# Snake DQN AI

A simple Snake game where an AI learns to play on its own using Deep Q-Learning.

This started as a small experiment to better understand reinforcement learning, and gradually turned into a more complete project with visualization and training logic.

---

## What’s inside

* Playable Snake (WASD controls)
* AI that learns from trial and error
* Neural network built with PyTorch
* Experience replay
* Real-time view of what the network is doing internally

---

## How it works (in plain terms)

The AI doesn’t know how to play at the beginning.

It just:

* tries random moves
* gets rewarded for good decisions (like eating food)
* gets punished for bad ones (like crashing)

Over time, it starts to figure out patterns and improve.

---

## Run it

```bash
pip install pygame numpy torch
python snake.py
```

Switch between modes in the code:

```python
MODE = "human"  # or "ai"
```

---

## Note

This is still a learning project, not a perfect implementation.

The AI is far from optimal and there are many things that could be improved (training stability, reward tuning, architecture, etc.), but that’s part of the process.

---

## Why I made this

Mostly to understand how reinforcement learning actually works in practice, not just in theory.

Seeing the model slowly go from random moves to something that looks like strategy is surprisingly satisfying.
