---
title: "TIL: Prevent Data Leakage by Splitting Before Scaling"
date: 2026-03-02
category: machine-learning
---

### What I learned
A very common beginner mistake is to scale the entire dataset (e.g., using `StandardScaler` or `MinMaxScaler`) *before* splitting it into training and testing sets. This causes **Data Leakage**.

Scaling relies on global metrics like the "mean" and "standard deviation" of the data. If you scale the whole dataset first, the test set's mean and variance are secretly included in the calculation. This means your training data gets a "sneak peek" at the test data. It is the equivalent of a student seeing parts of the final exam while still studying. 

**The Golden Rule:** Always split your data first. Then, calculate the scaling metrics (`fit`) **only** on the training set, and use those exact same metrics to apply the scaling (`transform`) to the test set.

### Simple example

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ❌ BAD WAY (Data Leakage)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X) # The scaler sees the entire dataset!
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


# ✅ GOOD WAY (Strict Isolation)
# Step 1: Split the data first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Initialize the scaler
scaler = StandardScaler()

# Step 3: Learn the mean/std ONLY from the training data, and scale it
X_train_scaled = scaler.fit_transform(X_train)

# Step 4: Scale the test data using the existing rules learned from the train data
# NEVER use .fit() or .fit_transform() on the test set!
X_test_scaled = scaler.transform(X_test)
```

### Why it matters
If you leak test data into your training process, your model's accuracy during evaluation will be artificially high because it "cheated." When you deploy this model to production and feed it truly unseen real-world data, its performance will drop significantly. Keeping the test set strictly isolated ensures your evaluation metrics are honest and reliable.

**Source:** Foundational machine learning best practices for data preprocessing and model evaluation.