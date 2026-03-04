---
title: "TIL: He Initialization is the Key to Unlocking Deep ReLU Networks"
date: 2026-03-04
category: machine-learning
---

### What I learned
He initialization (also known as Kaiming initialization) sets the standard deviation of a layer's weights to `√(2/N)`, where `N` is the number of inputs to that layer (fan-in). 

It is mathematically designed specifically for the **ReLU** activation function. Because ReLU outputs zero for any negative input, it effectively destroys roughly half of the data passing through it. If left uncompensated, the variance of the signal would shrink layer by layer until it vanishes. The factor of `2` in the numerator exactly compensates for this loss, keeping the signal's variance perfectly stable across deep layers. 

*Note:* **Xavier/Glorot Initialization** uses `√(1/N)` and is mathematically derived for symmetric activation functions like Sigmoid or Tanh. Using Xavier with ReLU in deep networks still leads to signal decay.

### Simple example

```python
import numpy as np

n_inputs = 4  # e.g., 4 neurons feeding into the next layer

# A naive, generic initialization (far too small for deep ReLU networks)
naive_std = 0.01

# He Initialization (specifically designed to compensate for ReLU)
he_std = np.sqrt(2 / n_inputs)

print(f"Naive init std: {naive_std}")
print(f"He init std:    {he_std:.3f}") 
# Output: 0.707 
# The He initialization is roughly 70x larger, ensuring neurons fire 
# immediately and gradients can flow backward from epoch 1.
```

### Why it matters
He initialization is often the single architectural change that turns a "dead" deep network into one that converges rapidly. Without correct initialization, a deep ReLU network will flatline at baseline accuracy (like a random guess) regardless of how many epochs you train it or how you tweak the learning rate. Proper weight scaling is not an optimization trick; it is a fundamental mathematical requirement for deep learning.