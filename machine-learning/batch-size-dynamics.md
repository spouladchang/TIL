---
title: "TIL: Small Batch Sizes Add Noise That Helps Generalization"
date: 2026-03-05
category: machine-learning
---

### What I learned
Batch size isn't just a hardware constraint for fitting data into GPU memory; it fundamentally changes the math of how your neural network navigates the loss landscape.

When you use a **large batch size** (or the entire dataset), the gradient calculation is highly accurate. The loss drops very smoothly. However, this lack of noise means the optimizer can easily get trapped in sharp, narrow local minima, which often generalize poorly to new unseen data.

When you use a **small batch size** (e.g., 16 or 32), the gradient is calculated on a small, random subset of data. This creates a "noisy" or "stochastic" estimation of the true true gradient.  This noise is actually a good thing! It acts as a regularization effect, constantly jolting the optimizer and helping it bounce out of sharp valleys into wider, flatter, and more robust minima that perform much better on the test set.

### Simple example

```python
from tensorflow import keras

# Assume 'model' is already compiled

# Batch Size = 1 (Pure SGD): Extremely noisy, erratic updates, very slow hardware utilization
# model.fit(X_train, y_train, epochs=10, batch_size=1)

# Batch Size = 32 (Mini-batch SGD): The industry "sweet spot"
# Balances hardware efficiency with enough gradient noise for good generalization
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# Batch Size = 1000+ (Large Batch): Very smooth gradient, fast hardware utilization
# But risks getting stuck in sharp local minima and poor test accuracy
# model.fit(X_train, y_train, epochs=10, batch_size=1024)
```

### Why it matters
If your model has high training accuracy but struggles on the test set (overfitting), simply reducing the batch size (e.g., from 128 down to 32 or 16) can sometimes act as a free regularization technique. It trades a bit of mathematical smoothness for the chaotic energy needed to find better, flatter solutions.