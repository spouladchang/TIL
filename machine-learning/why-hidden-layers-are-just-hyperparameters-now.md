---
title: "TIL: The Number of Hidden Layers is Just Another Hyperparameter (NAS)"
date: 2026-03-24
category: machine-learning
---

### What I learned
When building a Neural Network, the first question is always: *"How many hidden layers do I need? And how many neurons in each?"* In the old days of Deep Learning, this was a manual, trial-and-error process. An engineer would build a 2-layer model, train it, check the accuracy, then rewrite the code to build a 3-layer model, train it again, and compare. They would manually lock in an architecture before they even started tuning things like the Learning Rate or Batch Size.



Today, this manual guessing game is obsolete thanks to **Neural Architecture Search (NAS)**. 
We no longer treat the structure of the network as a fixed, sacred design. Instead, we treat the *number of layers* and the *number of neurons per layer* exactly like any other hyperparameter (like Dropout rate or Learning rate). We give a search algorithm (like KerasTuner) a defined "Search Space" (e.g., "Try between 1 and 3 layers, with 32 to 512 neurons") and let the machine build, test, and discard hundreds of different architectures automatically to find the mathematical optimum.

### Simple example

```python
import keras_tuner as kt
from tensorflow import keras

# Instead of hardcoding the model, we write a "model builder" function
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
    # The magic of NAS: The number of hidden layers is dynamically chosen (1, 2, or 3)
    for i in range(hp.Int('num_layers', min_value=1, max_value=3)):
        
        # The number of neurons in each layer is also dynamically chosen (32 to 512, step=32)
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation='relu'
        ))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

'''
The Tuner will now dynamically generate different architectures:
Trial 1: A shallow 1-layer network with 64 neurons.
Trial 2: A deep 3-layer network with [256, 128, 64] neurons.
Trial 3: A 2-layer network with [512, 32] neurons.
'''
```

### Why it matters
Treating architecture as a hyperparameter fundamentally changes how we approach Deep Learning. It removes human bias from the design process. Sometimes a simple, shallow network outperforms a deep, complex one on a specific dataset, but a human engineer might be too biased towards complexity to even test the shallow version. By defining a broad search space and letting an algorithm explore it, we guarantee that we find the most efficient, data-driven architecture possible, saving both engineering time and computational resources.