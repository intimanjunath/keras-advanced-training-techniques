# 🧠 Advanced Neural Network Training with Keras

This repository contains the complete solution to **Assignment 6** from our Deep Learning course. The assignment explores **advanced Keras training techniques**, including data augmentation, regularization, hyperparameter tuning, custom training components, and training loops—demonstrated using multiple Colab notebooks, across different data types.

---

## 📌 Contents

This assignment is divided into **two parts**:

- **Part 1**: Data Augmentation, Generalization & Regularization Techniques
- **Part 2**: Advanced Keras Constructs (Custom Models, Losses, Schedulers, etc.)

Each section includes Colab notebooks with code, visualizations, A/B tests, and explanations.

📹 A complete walkthrough video is also provided below.

---

## 🎥 Video Walkthrough

👉 [Click here to watch the video walkthrough](https://www.youtube.com))  

---

## 🧩 Part 1: Generalization, Regularization & Augmentation

| Notebook | Description |
|---------|-------------|
| [`1a_data_augmentation.ipynb`](part1_augmentation_regularization/1a_data_augmentation.ipynb) | Classic image augmentation using Keras (`ImageDataGenerator`, `tf.image`, `keras_cv`) with A/B testing. |
| [`1b_regularization_l1_l2.ipynb`](part1_augmentation_regularization/1b_regularization_l1_l2.ipynb) | L1 and L2 regularization comparisons and effect on overfitting. |
| [`1c_dropout_earlystop.ipynb`](part1_augmentation_regularization/1c_dropout_earlystop.ipynb) | Dropout, EarlyStopping, and their impact on generalization. |
| [`1d_mc_dropout.ipynb`](part1_augmentation_regularization/1d_mc_dropout.ipynb) | Monte Carlo Dropout: performing inference with uncertainty estimates. |
| [`1e_initializations.ipynb`](part1_augmentation_regularization/1e_initializations.ipynb) | Comparisons of initializers like He, Glorot, and LeCun. |
| [`1f_batchnorm_custom_regularization.ipynb`](part1_augmentation_regularization/1f_batchnorm_custom_regularization.ipynb) | Batch normalization and custom regularizers implemented manually. |
| [`1g_callbacks_tensorboard_keras_tuner.ipynb`](part1_augmentation_regularization/1g_callbacks_tensorboard_keras_tuner.ipynb) | Using callbacks (`EarlyStopping`, `ModelCheckpoint`), TensorBoard, and Keras Tuner. |
| [`1h_data_augmentation_various_modalities.ipynb`](part1_augmentation_regularization/1h_data_augmentation_various_modalities.ipynb) | Augmentation examples across image, video, text (NLP), speech, tabular, time-series using `AugLy`, `nlpaug`, and others. |

---

## 🌿 Part 1 (FastAI)

| Notebook | Description |
|---------|-------------|
| [`1i_fastai_augmentation.ipynb`](part1_fastai/1i_fastai_augmentation.ipynb) | Demonstration of FastAI’s powerful data augmentation and TTA (Test-Time Augmentation). |

---

## 🚀 Part 2: Advanced Keras Constructs

| Notebook | Description |
|---------|-------------|
| [`2a_custom_scheduler_dropout_norm.ipynb`](part2_advanced_constructs/2a_custom_scheduler_dropout_norm.ipynb) | Custom OneCycle LR Scheduler, Alpha Dropout, MaxNormDense layers. |
| [`2b_custom_losses_activation_metrics.ipynb`](part2_advanced_constructs/2b_custom_losses_activation_metrics.ipynb) | Custom Huber loss, LeakyReLU activation, custom initializers, metrics. |
| [`2c_custom_layers_models_optimizers.ipynb`](part2_advanced_constructs/2c_custom_layers_models_optimizers.ipynb) | Creating custom layers, residual blocks, and optimizers. |
| [`2d_custom_training_loop.ipynb`](part2_advanced_constructs/2d_custom_training_loop.ipynb) | Custom training loop using `tf.GradientTape` (manual training with metrics). |

---

## 🛠️ Libraries & Tools Used

- TensorFlow 2.x / Keras
- keras_cv
- FastAI
- TensorBoard
- Keras Tuner
- AugLy, NLP-Aug, librosa
- matplotlib, seaborn
- pandas, scikit-learn
