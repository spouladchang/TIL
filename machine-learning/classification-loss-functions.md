---
title: "TIL: The Cheat Sheet for Classification Loss Functions"
date: 2026-03-02
category: machine-learning
---

### What I learned
While `sparse_categorical_crossentropy` is perfect for multi-class problems with integer labels, it is not the only option. Choosing the right loss function depends entirely on the shape of your labels and the specific goal of your model. 

Here is a quick high-level overview of the most common classification scenarios:

* **Binary Classification (2 classes, e.g., Spam vs. Not Spam):**
    * **Loss:** `binary_crossentropy`
    * **Output Activation:** `sigmoid`
* **Multi-Class Classification (3+ classes, integer labels like 0, 1, 2):**
    * **Loss:** `sparse_categorical_crossentropy`
    * **Output Activation:** `softmax`
* **Multi-Class Classification (3+ classes, one-hot encoded labels like [0, 1, 0]):**
    * **Loss:** `categorical_crossentropy`
    * **Output Activation:** `softmax`
* **Multi-Label Classification (Multiple tags per sample, e.g., an image with a dog AND a car):**
    * **Loss:** `binary_crossentropy` (applied independently to each output node)
    * **Output Activation:** `sigmoid`
* **Imbalanced Classification (e.g., 99% healthy, 1% sick):**
    * **Loss:** `Focal Loss` (Forces the model to focus on hard, rare examples instead of easy, common ones).

### Simple example
Here is how you switch the "compass" of your neural network in Keras based on your specific problem:

```python
from tensorflow import keras

# Scenario 1: Predicting Spam vs. Not Spam (Binary)
model_binary = keras.Sequential([...])
# Notice the binary loss function
model_binary.compile(loss='binary_crossentropy', optimizer='adam')

# Scenario 2: Predicting Spiral Colors with labels 0, 1, 2 (Multi-Class Integer)
model_multi = keras.Sequential([...])
# Notice the sparse loss function
model_multi.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

### Why it matters
The loss function is the mathematical objective your model tries to minimize. If you use `categorical_crossentropy` on integer labels, your model will silently produce completely wrong results — which is often worse than a crash because you won't know something is broken. If you use standard loss functions on highly imbalanced data, your model will cheat (always predicting the majority class) and become useless in production. Matching the loss function to your data's reality is the foundational step of building any classifier.
