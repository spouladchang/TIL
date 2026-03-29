# TIL — Today I Learned

A personal collection of small, self-contained notes on things I learned while building and debugging real projects.

Each note follows the same format: what I learned, a minimal working example, and why it matters. No long tutorials, no textbook chapters — just the single idea, clearly stated.

---

## How This Repo Is Organized

Each topic lives in its own folder. Every file inside is one standalone concept.

```
TIL/
├── machine-learning/
│   └── ... (20 notes on NNs, training, regularization, and tuning)
├── computer-vision/    (coming soon)
├── Natural Language Processing    (coming soon)
```

New folders get added as new topics come up. This README gets updated each time.

---

## 📂 machine-learning

Notes from hands-on experiments with neural networks using TensorFlow/Keras. The concepts here follow the natural sequence of building a model end-to-end — from raw data to a working, well-trained network.

Rather than reading these as a random list, use the roadmap below. Each section represents one stage of that pipeline.

---

### 🗺️ End-to-End Roadmap

---

#### 1 — Data Preparation
*Before any model sees your data, it needs to be clean, fairly split, and scaled. Mistakes here silently corrupt everything downstream.*

| Note | One-line summary |
|------|-----------------|
| [Prevent Data Leakage by Splitting Before Scaling](machine-learning/data-leakage-scaling.md) | Always split first, then fit the scaler on training data only — never on the full dataset |
| [Feature Scaling is the Difference Between Convergence and Total Failure](machine-learning/feature-scaling-matters.md) | Unscaled features warp the loss surface so badly that even a correct model can't learn |

---

#### 2 — Choosing the Right Loss Function
*The loss function is the mathematical objective your model optimizes. It must match your label format and problem type exactly.*

| Note | One-line summary |
|------|-----------------|
| [The Cheat Sheet for Classification Loss Functions](machine-learning/classification-loss-functions.md) | Binary, multi-class integer, multi-class one-hot, multi-label, and imbalanced — each needs a different loss |

---

#### 3 — Network Architecture & Activation Functions
*How you wire your network determines whether learning is even possible. The activation function, weight initialization, and structural choices are the decisions that matter most.*

| Note | One-line summary |
|------|-----------------|
| [Softmax is an Output Specialist, Not a Hidden Layer Team Player](machine-learning/softmax-regression-vs-activation.md) | Softmax in hidden layers forces neurons to compete and suppress each other — use ReLU instead |
| [Sigmoid with Tiny Weights Causes Learning to Flatline](machine-learning/vanishing-gradient-sigmoid.md) | Sigmoid gradients shrink at every layer — in deep networks, the signal reaches the first layer as nearly zero |
| [Tiny Weight Initialization Causes Dying ReLU in Deep Networks](machine-learning/dead-relu-bad-init.md) | ReLU with std=0.01 init kills most neurons on the first pass — they output zero and never recover |
| [He Initialization is the Key to Unlocking Deep ReLU Networks](machine-learning/he-initialization.md) | std=√(2/N) compensates for ReLU's dead half, keeping signal variance stable across every layer |
| [Flattening Images for MLPs Destroys Spatial Relationships](machine-learning/why-flattening-images-destroys-spatial-context.md) | Stretching a 2D image into a 1D array permanently erases which pixels are neighbors |
| [Data Augmentation Destroys Pure MLPs](machine-learning/why-data-augmentation-destroys-pure-mlps.md) | Shifting an image 2 pixels moves a feature to a different fixed weight — the MLP sees a completely different world |

*Read the first four in order — each one is the solution to the problem described by the previous one. The last two show where the MLP architecture hits its structural ceiling, setting up the case for CNNs.*

---

#### 4 — Training Dynamics
*Once the architecture is sound, two things shape how training goes: how much data the model sees per update, and how many epochs are enough.*

| Note | One-line summary |
|------|-----------------|
| [Small Batch Sizes Add Noise That Helps Generalization](machine-learning/batch-size-dynamics.md) | Large batches converge smoothly but land in sharp minima — small batches add regularizing noise |

---

#### 5 — Optimization Algorithms
*The optimizer decides how to move through the loss landscape. These notes follow the natural evolution from basic SGD to the modern standard.*

| Note | One-line summary |
|------|-----------------|
| [Momentum and Nesterov Prevent Optimizers from Getting Stuck](machine-learning/optimizer-momentum-nesterov.md) | Momentum adds inertia to power through flat zones; Nesterov looks ahead to avoid overshooting |
| [Adaptive Learning Rates: Why RMSprop Beats Vanilla SGD](machine-learning/why-rmsprop-beats-vanilla-sgd.md) | RMSprop adapts the step size per parameter — steep slopes get smaller steps, flat zones get larger ones |
| [Why Adam is the Deep Learning Industry Standard](machine-learning/why-adam-is-the-industry-standard.md) | Adam combines Momentum and RMSprop — speed in consistent directions, stability across all parameters |

*Read these three in order — each optimizer is built on top of the previous one's idea.*

---

#### 6 — Regularization
*A model that fits training data perfectly is often useless in production. Regularization is the set of tools that forces your model to generalize instead of memorize.*

| Note | One-line summary |
|------|-----------------|
| [Dropout: Forcing Neurons to Work as a Team](machine-learning/how-dropout-forces-neurons-into-teamwork.md) | Randomly disabling neurons during training prevents co-adaptation and builds redundant, robust features |
| [Why L1 Regularization "Blinds" Neural Networks on Image Data](machine-learning/why-l1-regularization-blinds-image-networks.md) | L1 forces weights to exactly zero — on images, it permanently deletes pixels the model can never see again |
| [The Regularization Trinity: Dropout, AdamW, and Early Stopping](machine-learning/the-regularization-trinity-dropout-adamw-early-stopping.md) | Three non-conflicting defenses: one spatial, one mathematical, one temporal — use all three together |

*Read these three in order — Dropout and L1 each address one angle, then the Trinity note shows how to combine the full defense.*

---

#### 7 — Evaluation & Generalization
*A model isn't done when training loss is low. The real question is whether it learned the rules or just memorized the answers.*

| Note | One-line summary |
|------|-----------------|
| [Overfitting: When Your Model Memorizes the Exam Instead of Learning](machine-learning/why-overfitting-is-memorizing-the-exam.md) | High training accuracy + rising validation loss = memorization, not learning |
| [Why Deep Learning Abandoned K-Fold Cross Validation](machine-learning/why-deep-learning-abandoned-k-fold-cv.md) | With large datasets the law of large numbers makes a single split statistically solid — K-Fold is 5× the cost for no gain |

---

#### 8 — Hyperparameter Tuning
*Once your model works, the question becomes: is this the best version of it? Tuning is how you stop guessing and start searching systematically.*

| Note | One-line summary |
|------|-----------------|
| [The Number of Hidden Layers is Just Another Hyperparameter](machine-learning/why-hidden-layers-are-just-hyperparameters-now.md) | Neural Architecture Search treats layer count and neuron count as searchable values, not fixed design decisions |
| [Hyperband vs. Bayesian Optimization: Guessing vs. Killing](machine-learning/hyperband-vs-bayesian-optimization-guessing-vs-killing.md) | Bayesian guesses smarter combinations; Hyperband kills bad ones early — each wins in different conditions |

---

## 📂 computer-vision *(coming soon)* 
## 📂 Natural Language Processing *(coming soon)*
