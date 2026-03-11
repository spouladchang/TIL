---
title: "TIL: Overfitting: When Your Model Memorizes the Exam Instead of Learning"
date: 2026-03-11
category: machine-learning
---

### What I learned
The biggest trap in machine learning is thinking that a model with 99% accuracy is a "good" model. Often, it is just a model that has **memorized the answers** instead of understanding the underlying rules.

Imagine a student studying for a math exam by looking at a practice test. Instead of learning the formulas (generalizing), the student just memorizes that "Question 1 is B" and "Question 2 is 42". If you give them the exact same practice test (the **Training Set**), they score 100%. But if you change the numbers in the questions for the final exam (the **Test/Validation Set**), they fail completely. 

This is exactly what **Overfitting** is in neural networks. The model has too much capacity (too many layers/neurons) and ends up memorizing the noise and exact pixel values of the training data rather than extracting the true, generalizable patterns.



The classic symptom of overfitting is a growing gap between training and validation metrics: the training loss keeps going down, but the validation loss hits a minimum and then starts going back up (forming a U-shape).

### Simple example

```python
from tensorflow import keras

# Imagine training a massive, overly complex model on a small dataset
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

'''
Look at the terminal output during the final epochs.
This is the absolute textbook definition of Overfitting:

Epoch 98/100
accuracy: 0.9950 - loss: 0.0120 - val_accuracy: 0.6500 - val_loss: 1.8500
Epoch 99/100
accuracy: 0.9980 - loss: 0.0050 - val_accuracy: 0.6420 - val_loss: 2.1000
Epoch 100/100
accuracy: 0.9995 - loss: 0.0010 - val_accuracy: 0.6380 - val_loss: 2.4500

Notice how the model is practically perfect on the training data (99.9% accuracy), 
but it is getting progressively worse on the validation data (loss is rising).
'''
```

### Why it matters
Overfitting matters because **the real world is the validation set.** A model that memorizes the training data is fundamentally useless in production. It is the core reason why AI engineers cannot just build infinitely large networks to solve problems. Understanding overfitting is the mandatory prerequisite before you can learn about "Regularization" techniques (like L1/L2, Dropout, and Early Stopping), which are the tools we use to force the model to stop memorizing and start actually learning.