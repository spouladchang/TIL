---
title: "TIL: Flattening Images for MLPs Destroys Spatial Relationships"
date: 2026-03-21
category: machine-learning
---

### What I learned
When we look at an image, we don't just see a random list of colors; we see shapes. A shape (like a wheel or an eye) is defined entirely by **spatial relationships**—which pixels are next to, above, or below each other. Meaning exists in the 2D local geometry.

Standard Multi-Layer Perceptrons (MLPs or Dense Networks) only accept 1D arrays as input. To feed a 28x28 pixel image (like Fashion MNIST) into an MLP, we must use a `Flatten` layer to stretch it into a single line of 784 pixels. 



When we do this, we permanently destroy the 2D geometry. 
In the original 2D grid, Pixel #14 and Pixel #42 might be directly above and below each other (vertically adjacent, sharing a strong visual relationship). But after flattening, they are separated by 27 other unrelated pixels in the 1D line. The network has no built-in understanding that these pixels are physically connected in the real world. It just sees 784 independent, disconnected variables.

### Simple example

```python
from tensorflow import keras
import numpy as np

# A simulated 3x3 image (9 pixels total)
image_2d = np.array([
    [1, 2, 3],  # Top row
    [4, 5, 6],  # Middle row
    [7, 8, 9]   # Bottom row
])

# Pixel '2' is directly above Pixel '5'. They are vertically adjacent.

model = keras.Sequential([
    # The Flatten layer converts the 2D grid into a 1D array
    keras.layers.Flatten(input_shape=(3, 3))
])

'''
What the Dense layer actually receives:
[1, 2, 3, 4, 5, 6, 7, 8, 9]

Now look at Pixel '2' and Pixel '5'. 
In this 1D array, they are separated by '3' and '4'. 
The spatial relationship ("above/below") is completely gone. 
The Dense layer treats them as entirely unrelated inputs.
'''
```

### Why it matters
Because flattening destroys spatial awareness, MLPs are incredibly rigid. If a model learns to recognize a shoe located perfectly in the center of the image, the specific weights connected to the "center pixels" get optimized. If you show the exact same shoe, but shifted slightly to the top-left corner, the MLP will fail completely. The activated pixels have changed, and the model doesn't know how to share knowledge across different locations. This structural flaw is the exact reason why the Deep Learning industry had to invent Convolutional Neural Networks (CNNs), which scan images in 2D without flattening them first.