---
title: "TIL: Momentum and Nesterov Prevent Optimizers from Getting Stuck"
date: 2026-03-05
category: machine-learning
---

### What I learned
Vanilla Stochastic Gradient Descent (SGD) is short-sighted: it only calculates the slope at its exact current position. If it hits a flat region (saddle point) or a shallow ditch (local minimum), the gradient becomes near zero and learning completely stops. Furthermore, in narrow, steep valleys, it tends to bounce wildly from wall to wall instead of moving forward toward the goal.

**Momentum** solves this by adding "inertia". Instead of relying solely on the current gradient, it adds a fraction of the *previous* update's velocity. Like a snowball rolling down a hill, it builds up speed in consistent directions (powering through flat zones) and cancels out conflicting directions (dampening the wild, inefficient bounces).

**Nesterov Accelerated Gradient (NAG)** makes Momentum smarter. A standard momentum update can build up so much speed that it completely overshoots the minimum at the bottom of the hill. Nesterov calculates the gradient *slightly ahead* in the direction of the momentum. This "lookahead" lets the optimizer effectively hit the brakes *before* it overshoots the target.

### Simple example

```python
from tensorflow import keras

# 1. Vanilla SGD: Short-sighted, easily gets stuck, bounces around
opt_vanilla = keras.optimizers.SGD(learning_rate=0.01)

# 2. SGD with Momentum: Builds inertia (momentum=0.9 is the industry standard)
# It adds 90% of the previous step's velocity to the current step
opt_momentum = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# 3. Nesterov Momentum: Looks ahead to prevent overshooting
opt_nesterov = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)


'''
The conceptual difference in math:
Vanilla:  step = lr * gradient
Momentum: velocity = (0.9 * old_velocity) + (lr * gradient) -> step = velocity
'''
```

### Why it matters
In deep learning, the loss landscape is incredibly complex, heavily warped by unscaled features, and filled with saddle points. Switching from Vanilla SGD to SGD with Momentum (and Nesterov) is often the key to accelerating convergence and helping a stuck model finally escape local minima, all with virtually zero extra computational cost.