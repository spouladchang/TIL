---
title: "TIL: Sigmoid with Tiny Weights Causes Learning to Flatline"
date: 2026-03-03
category: machine-learning
---

### What I learned
When you use sigmoid activations combined with tiny random weights (e.g., standard deviation of 0.01), the network's pre-activations are close to zero. The sigmoid function maps zero to 0.5, placing the outputs exactly in the middle of the sigmoid curve. 



The maximum possible gradient of the sigmoid function (which occurs at x=0) is exactly 0.25. During backpropagation, the chain rule multiplies these gradients together layer by layer. In a shallow network, this severely slows down learning. In a deep network, it is catastrophic: multiplying fractions like 0.25 together across multiple layers exponentially shrinks the gradient, killing the learning signal entirely before it reaches the earlier layers (known as the Vanishing Gradient Problem).

### Simple example

```python
import numpy as np

def sigmoid_grad(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

# With tiny weights, pre-activations are near 0.
# The gradient of sigmoid at 0 is exactly 0.25.
max_grad = sigmoid_grad(0)
print(f"Max gradient at a single layer: {max_grad}")

# Across 4 hidden layers, the gradient shrinks exponentially:
vanishing_signal = max_grad ** 4
print(f"Signal reaching the first layer: {vanishing_signal}") 
# Output: 0.0039 — almost no learning signal left!
```

### Why it matters
This explains why simply tweaking hyperparameters (like increasing the learning rate or adding more epochs) won't fix a model that is stuck at baseline accuracy (e.g., 50% for binary classification). The architecture itself blocks the learning process. Switching the hidden layers to an activation function like **ReLU** fixes this instantly, because ReLU's gradient for positive inputs is exactly 1, passing the full learning signal backward without shrinking it.