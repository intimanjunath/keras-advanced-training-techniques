# 🧠 Advanced Neural Network Training with Keras

This repository contains the complete solution to **Assignment 6** from our Deep Learning course, all consolidated into a **single Colab notebook**. The assignment explores:

- **Part 1**: Data Augmentation, Generalization & Regularization (a–l)  
- **Part 2**: Advanced Keras Constructs (i–xi)

Each section is clearly marked in the notebook with code examples, visualizations, and explanations.

---

## 📌 Contents

- [`[assignment6_full_pipeline.ipynb](https://github.com/intimanjunath/keras-advanced-training-techniques/blob/main/assignment6_full_pipeline.ipynb)`](assignment6_full_pipeline.ipynb) — one-stop Colab covering **all** tasks  
- [`README.md`](README.md) — this overview  
- `/videos/assignment_walkthrough.mp4` — narrated screen‐capture of the notebook  

---

## 🎥 Video Walkthrough

👉 [Watch the walkthrough](videos/assignment_walkthrough.mp4)

---

## 🚩 Part 1: Data Augmentation & Regularization

| Task | Description |
|------|-------------|
| **a)** | L1 & L2 regularization demos (A/B test) |
| **b)** | Dropout vs. no-dropout comparison |
| **c)** | EarlyStopping callback example |
| **d)** | Monte Carlo Dropout for uncertainty |
| **e)** | Kernel initializers: Glorot, He, RandomNormal |
| **f)** | BatchNormalization impact |
| **g)** | Custom Dropout & custom L1 regularizer |
| **h)** | Callbacks & TensorBoard logging |
| **i)** | Hyperparameter search with Keras Tuner |
| **j)** | `keras_cv`–based augmentation layers |
| **k)** | Cross‐modality augmentation & classification (image, video, text, time-series, tabular, speech, document images) |
| **l)** | FastAI `aug_transforms` demo |

---

## 🚀 Part 2: Advanced Keras Constructs

| Task | Description |
|------|-------------|
| **i)**  | Custom learning-rate scheduler (OneCycle) |
| **ii)** | MC-AlphaDropout for inference |
| **iii)**| `MaxNormDense` weight constraint layer |
| **iv)** | Advanced TensorBoard callback (scalars & histograms) |
| **v)**  | Custom Huber loss for regression |
| **vi)** | Custom activation, initializer, regularizer & constraint |
| **vii)**| Custom Huber metric implementation |
| **viii)**| Custom layers: AddGaussianNoise & Exponential |
| **ix)** | ResidualBlock & ResidualRegressor model |
| **x)**  | Custom momentum optimizer (`MyMomentumOptimizer`) |
| **xi)** | Manual training loop with momentum-SGD (`tf.GradientTape`) |

---

## 🛠️ Tools & Libraries

- **TensorFlow 2.x / Keras**  
- **keras_cv** for image augmentations  
- **FastAI** for advanced transforms  
- **TensorBoard** & **Keras Tuner**  
- **AugLy**, **nlpaug**, **tf.signal** for various modalities  
- **scikit-learn**, **numpy**, **matplotlib**

---

## 📖 How to Run

1. Open `assignment6_full_pipeline.ipynb` in Colab (use “Open in Colab” badge).  
2. Execute cells sequentially; all datasets are built-in or synthetic.  
3. For TensorBoard sections, after training run:
   ```bash
   %tensorboard --logdir=logs/
