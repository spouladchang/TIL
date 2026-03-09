---
title: "TIL: Adaptive Learning Rates: Why RMSprop Beats Vanilla SGD"
date: 2026-03-09
category: machine-learning
---

### What I learned
Vanilla Stochastic Gradient Descent (SGD) has a major blind spot: it uses a **single, fixed learning rate** for every single weight in the neural network. If one feature's landscape is a steep cliff, SGD takes a massive, erratic jump. If another feature's landscape is a flat plain, SGD takes a painfully slow micro-step. 

Imagine driving a car and keeping the gas pedal pressed exactly the same amount regardless of whether you are climbing a steep mountain or cruising on a flat highway. That is Vanilla SGD.



**RMSprop (Root Mean Square Propagation)** fixes this by introducing an **Adaptive Learning Rate**. It keeps a running average of the recent gradients' magnitudes for *each individual parameter*. 

Before taking a step, it divides the global learning rate by this running average:
* If a parameter has steep, explosive gradients, RMSprop automatically applies the brakes (shrinks the effective learning rate).
* If a parameter has flat, tiny gradients, RMSprop hits the gas pedal (maintains or boosts the effective learning rate).

### Simple example

```python
from tensorflow import keras

# 1. Vanilla SGD: One rigid learning rate for all parameters
# It is blind to the steepness of different features.
opt_sgd = keras.optimizers.SGD(learning_rate=0.01)

# 2. RMSprop: Adapts the learning rate for EACH parameter dynamically
# rho=0.9 is the decay factor (how much it remembers of past gradients)
opt_rmsprop = keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)

'''
The conceptual math difference:

Vanilla SGD step:  
step = learning_rate * current_gradient

RMSprop step:  
step = (learning_rate / sqrt(average_of_past_squared_gradients + epsilon)) * current_gradient

-> High past gradients (steep slope) = denominator is large = smaller step (brakes)
-> Low past gradients (flat slope) = denominator is small = larger step (gas pedal)
'''
```

### Why it matters
RMSprop explicitly **beats Vanilla SGD** because it eliminates the nightmare of perfectly tuning a global learning rate. With SGD, a slightly wrong learning rate means the model either diverges completely (explodes) or freezes in flat zones. RMSprop survives these complex, warped loss landscapes by acting as an automatic transmission for your neural network. It speeds up training drastically and prevents the optimizer from getting helplessly trapped—which is exactly why modern networks rarely use pure Vanilla SGD anymore.