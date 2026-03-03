---
title: "TIL: Tiny Weight Initialization Causes Dying ReLU in Deep Networks"
date: 2026-03-03
category: machine-learning
---

### What I learned
While ReLU solves the vanishing gradient problem for positive values, it is highly sensitive to weight initialization. If weights are initialized too small (e.g., from a normal distribution with `std=0.01`), the pre-activations are tiny, and roughly half of them will randomly fall below zero.

Because ReLU outputs exactly zero for any negative input, a large percentage of neurons will output zero right from the first forward pass. Consequently, their gradients become zero during backpropagation. If a neuron's gradient is zero, its weights never update—it becomes "dead on arrival." In shallow networks, the damage might be limited, but in deep networks, this effect cascades, causing the entire network's learning to flatline just like it would with a vanishing gradient.

### Simple example

```python
import numpy as np

# Tiny random weight initialization
weights = np.random.normal(0, 0.01, (4, 4))
x = np.array([0.5, -0.3, 1.2, -0.8])

# Calculate pre-activations
pre_activation = x @ weights
print(f"Pre-activations:\n{np.round(pre_activation, 4)}")
# Output: Values hover very close to 0, roughly half are randomly negative.

# Apply ReLU activation
relu_output = np.maximum(0, pre_activation)
print(f"ReLU outputs:\n{np.round(relu_output, 4)}")
# Output: Many exact zeros. These neurons are dead and will not pass gradients backward.
```

### Why it matters
This proves that simply swapping Sigmoid for ReLU is not a magic fix for deep networks; ReLU isn't a "free fix" and requires weights to be scaled properly to match its mathematical behavior. The standard architectural solution for this is **He Initialization** (also known as Kaiming Initialization). It specifically scales the initial random weights based on the number of inputs to the layer, preserving the variance of the signal and ensuring neurons stay alive to learn.