# Quantization-vs-Complexity

This repository contains the official PyTorch implementation for our research project, **"On the Relationship Between Quantization Bit-width and Classification Task Complexity"**.

Recent advancements in model quantization have enabled the deployment of deep neural networks on resource-constrained devices. However, the trade-offs between the degree of quantization (i.e., bit-width) and a model's capacity to handle complex tasks are not yet fully understood. This project presents a systematic and quantitative analysis to answer the fundamental question:

> **"What is the optimal quantization bit-width for a given classification task complexity (number of classes)?"**

We investigate the performance, efficiency, and energy consumption of deep learning models across various bit-widths (1, 2, 4, 8, 16, 32-bit) as the number of target classes increases. Our analysis reveals critical "performance cliffs" for low-bit models and establishes a Pareto-optimal relationship between model precision and task complexity, providing practical guidelines for designing efficient AI systems.

## 🔑 Key Features

- **Systematic Analysis**: Experiments conducted on CIFAR-100 by varying the number of classes (10, 20, 50, 100).
- **Multiple Quantization Levels**: Implementation and comparison of models quantized to **1, 2, 4, 8, 16, and 32 bits**.
- **Comprehensive Benchmarking**: Measurement of not only achncuracy but also **inference time, model size, and energy consumption**.
- **Reproducibility**: All code, trained models, and experiment scripts are provided to ensure full reproducibility.
- **Custom Quantization Layers**: PyTorch implementation of custom layers for low-bit (1, 2, 4-bit) quantization.
