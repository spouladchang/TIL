---
title: "TIL: L1 vs L2 Regularization: The Tax System for Neural Networks"
date: 2026-03-11
category: machine-learning
---

### What I learned
When a neural network overfits (memorizes the training data), it usually does so by assigning massive, highly confident weights to a few specific features. **Regularization** is the mathematical cure for this. It adds a penalty—think of it as a "tax"—to the loss function based on the size of the model's weights. The larger the weights, the higher the tax the optimizer has to pay.

There are two main types of tax systems in machine learning:

1. **L1 Regularization (Lasso): The Ruthless Tax**
   It calculates the tax based on the *absolute value* of the weights. Mathematically, it pushes many weights exactly to zero. If a feature isn't incredibly important, L1 bankrupts it and shuts it off entirely. This creates **sparsity** and acts as an automatic feature selector.

2. **L2 Regularization (Ridge): The Progressive Tax**
   It calculates the tax based on the *squared value* of the weights. Squaring a huge number creates a massive penalty, so L2 acts like a progressive tax that heavily punishes "billionaire" neurons (huge weights) but leaves smaller weights mostly alone. It rarely pushes weights exactly to zero; instead, it shrinks and distributes them evenly across the network.

### Simple example

```python
from tensorflow import keras

# 1. L1 Regularization (Lasso)
# Forces many weights to become exactly 0.0. 
# Great if you have 10,000 features and want to ignore the useless ones.
layer_l1 = keras.layers.Dense(
    units=64, 
    activation='relu', 
    kernel_regularizer=keras.regularizers.l1(0.01) # 0.01 is the tax rate
)

# 2. L2 Regularization (Ridge)
# Shrinks weights to be small but non-zero.
# The industry standard for preventing a few neurons from dominating the network.
layer_l2 = keras.layers.Dense(
    units=64, 
    activation='relu', 
    kernel_regularizer=keras.regularizers.l2(0.01) # 0.01 is the tax rate
)
```

### Why it matters
This "tax system" matters because it is exactly how we prevent the model from memorizing the exam. By punishing excessively large weights, L2 forces the network to use *all* its neurons to find general, distributed patterns (teamwork) instead of relying on a few lucky pixels. Meanwhile, L1 acts as a brutal but effective filter, killing off useless inputs. Applying the right tax system transforms a fragile, cheating model into a robust, generalizable AI that performs beautifully on unseen real-world data.