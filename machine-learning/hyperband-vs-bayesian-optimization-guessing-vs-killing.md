---
title: "TIL: Hyperband vs. Bayesian Optimization: Guessing vs. Killing"
date: 2026-03-24
category: machine-learning
---

### What I learned
When navigating a massive hyperparameter search space (thousands of possible combinations of layers, neurons, and learning rates), Random Search is simply too dumb and slow. To fix this, the industry relies on two advanced algorithms, but they have completely different philosophies: **Bayesian Optimization** (The Detective) and **Hyperband** (The Ruthless Judge).

**1. Bayesian Optimization (Guessing)**
Bayesian Optimization is smart. It selects a combination, trains the model for the full 50 epochs, and logs the final accuracy. Then, it uses complex probability math to look at past results and *guess* a slightly better combination for the next trial. 
*The Flaw:* If it accidentally guesses a terrible combination, it still lets that bad model train for the full 50 epochs, wasting hours of GPU time.

**2. Hyperband (Killing)**
Hyperband doesn't try to guess the perfect combination. Instead, it relies on a brutal tournament system called **Successive Halving**. It starts by training 20 different random models, but it only gives them 5 epochs each. At epoch 5, it evaluates them and ruthlessly *kills* the worst-performing 50%. The surviving 10 models get another 10 epochs. Then it kills half again. Only the absolute best, most promising models ever reach the full 50 epochs.



### Simple example

```python
import keras_tuner as kt

# 1. The Detective (Bayesian)
# Will run exactly 20 trials. Each trial goes all the way to the end (e.g., 50 epochs).
# Great for small search spaces, but very slow if the models take long to train.
tuner_bayesian = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=20
)

# 2. The Ruthless Judge (Hyperband)
# Will test many more combinations but aggressively prune (kill) the bad ones early.
# max_epochs=50 means the absolute maximum a surviving model can train.
# factor=3 means it kills roughly 2/3 of the models at each evaluation round.
tuner_hyperband = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3 
)
```

### Why it matters
Choosing the right tuner is the difference between a project taking 2 hours versus taking an entire weekend. While Bayesian Optimization is mathematically elegant, Deep Learning models are so computationally heavy that letting a bad model train to completion is a luxury we cannot afford. Hyperband explicitly solves this by reallocating your hardware's compute power: it steals resources away from garbage models and gives them entirely to the winners. In the modern AI workflow, Hyperband is the undeniable king of efficient hyperparameter tuning.