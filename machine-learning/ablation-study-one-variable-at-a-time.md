---
title: "TIL: An Ablation Study Changes Exactly One Variable at a Time"
date: 2026-04-01
category: machine-learning
---

### What I learned
When tuning a neural network, the temptation is to change several things at once — add dropout, switch to AdamW, and increase layer width all in one go. If accuracy improves, you feel good. But you have no idea which change caused the improvement, or whether one of them actually hurt and the others compensated.

An **ablation study** enforces a strict rule: change exactly one variable per experiment phase. Every other setting is frozen at its previous best. The result is that each accuracy delta becomes a clean causal statement, not a correlation. "Dropout added +0.4%" is a fact. "Dropout plus AdamW plus wider layers added +1.2%" tells you almost nothing useful.

The word "ablation" comes from surgery — you remove or add one component at a time to understand what each part actually does to the system.

### Simple example

```python
# ❌ Exploratory tuning — no causal isolation
# Added dropout AND switched to AdamW AND widened layers.
# Accuracy went up 1.2%. But why?

# ✅ Ablation study structure
# Phase 0: baseline (fixed arch, no regularization)       → 87.8%
# Phase 1: + architecture search (layers, width, act)     → 88.2%  (+0.4%)
# Phase 2: + dropout search (everything else frozen)      → 88.6%  (+0.4%)
# Phase 3: + batch norm search (everything else frozen)   → 88.7%  (+0.1%)
# Phase 4: + AdamW + weight decay (everything else frozen)→ 89.1%  (+0.4%)
# Phase 5: + LR search + Hyperband (everything frozen)    → 89.2%  (+0.1%)

# Each delta is attributable to exactly one introduced variable.
```

### Why it matters
Ablation studies are how you distinguish signal from noise in your own experiments. Without them, you are just guessing which knob mattered. With them, you can make confident decisions — like "BatchNorm added almost nothing on this dataset, so I won't carry it into the next architecture" — and those decisions will hold up when you move to a new problem.

**Source**: fashion_mnist_mlp_glass_ceiling.ipynb — 6-phase controlled experiment on Fashion-MNIST
