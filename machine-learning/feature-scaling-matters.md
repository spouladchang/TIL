---
title: "TIL: Feature Scaling is the Difference Between Convergence and Total Failure"
date: 2026-03-04
category: machine-learning
---

### What I learned
If you feed raw, unscaled features into a neural network, even a perfectly architected model will likely fail to learn. When features span drastically different ranges (e.g., one feature ranges from -10 to 10, while another ranges from 1000 to 5000), the gradient steps become wildly uneven across dimensions.

Geometrically, unscaled data stretches the loss surface into a long, narrow, elliptical valley.  The optimizer will oscillate erratically across the steep dimension and crawl painfully slow along the flat dimension, often failing to find the minimum entirely. Scaling transforms this surface into a symmetrical bowl, allowing gradient descent to step smoothly and directly toward the optimal solution.

**Common Scaling Methods:**
* **Standardization (`StandardScaler`):** Centers data around a mean of 0 with a standard deviation of 1. This is generally the default and best choice for deep neural networks.
* **Normalization (`MinMaxScaler`):** Compresses data into a strict fixed range, usually 0 to 1. Great when you need hard boundaries (e.g., image pixel values).
* **Robust Scaling (`RobustScaler`):** Uses the median and interquartile range instead of the mean. Perfect when your dataset contains extreme, crazy outliers that would otherwise ruin the mean calculation.

### Simple example

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Raw features with wildly different scales
features_raw = np.array([[6.0, 4000.0], 
                         [4.0, 6500.0], 
                         [-5.8, 4640.0]])

# Standardization (Mean = 0, Std = 1)
std_scaler = StandardScaler()
features_std = std_scaler.fit_transform(features_raw)

# Normalization (Min = 0, Max = 1)
minmax_scaler = MinMaxScaler()
features_minmax = minmax_scaler.fit_transform(features_raw)

print("Raw Data:\n", features_raw)
print("\nStandardized:\n", features_std.round(2))
print("\nMin-Max Scaled:\n", features_minmax.round(2))
```

### Why it matters
A perfectly working model can drop from 95% accuracy to random guessing (~50%) simply by skipping the scaling step. Feature scaling isn't just a "nice-to-have" trick to boost accuracy by a few percent—it is a fundamental mathematical prerequisite for gradient descent to navigate the loss surface effectively.