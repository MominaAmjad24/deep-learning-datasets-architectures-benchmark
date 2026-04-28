# Deep Learning Datasets × Architectures Benchmark

Author: Momina Amjad

Course: Machine Learning / Deep Learning

Goal: Evaluate how neural network architecture performance depends on data modality.

---

 # Project Overview

This project benchmarks MLP (fully connected networks) and CNNs (convolutional neural networks) across three different data modalities:

1. Tabular data (Adult Income)
2. Natural images (CIFAR-100)
3. Medical histopathology images (PCam)

The central hypothesis is that model inductive bias must align with data structure for optimal performance.

---

 # Datasets

1. Adult Income (Tabular)

 - Binary classification
 - Structured feature vectors
 - No spatial structure

2. CIFAR-100 (Images)

- 100 class image classification
- 32×32 RGB images
- Strong spatial locality

3. PCam (Histopathology)

- Binary tumor detection
- 96×96 RGB patches
- Texture heavy medical imagery

---

#  Architectures

1. MLP

- Fully connected layers
- Treats input as a flat vector
- No built-in spatial assumptions

2. CNN

- Convolution + pooling
- Strong locality + hierarchical feature learning
- Best suited for spatial data (images)


3. ViT (Vision Transformer) Bonus

- Patch embedding + self-attention
- Global context modeling across patches
- Used on image dataset (CIFAR-100)

---

#  Inductive Bias Summary (Why architecture matters)

| Architecture  | Built-in inductive bias | Best matched data type | What it assumes |
| MLP | Minimal | Tabular (sometimes) | No locality; every feature interacts equally |
| CNN | Locality + translation tolerance | Images (CIFAR/PCam) | Nearby pixels matter; patterns repeat |
| ViT | Global attention over patches | Images (with enough data/tuning) | Relationships between any patches can matter |

How this affects results:  
- For tabular data, there is no natural “neighbor” relationship like pixels in an image, so CNN structure often doesn’t help much and may increase compute.  
- For images, CNNs are strong because edges/textures are local and reusable. ViTs can also work well because attention models long range relationships, but on small images CNNs can still be slightly better unless ViTs are heavily tuned or trained with more data.


#  Experimental Setup

- Optimizer: Adam
- Loss: Cross entropy
- Early stopping on validation accuracy
- Device: Apple MPS (when available)
    -- Metrics:
         - Accuracy
         - F1 score (binary tasks)
         - Training time

PCam note: Experiments used a subset (50k train / 10k val / 10k test) to fit laptop compute limits.

---

#  Results

| Dataset   | Architecture | Accuracy | F1     | Time   |
| --------- | ------------ | -------  | ------ | ------ |
| Adult     | MLP          | 0.8555   | 0.6463 | 2.0s   |
| Adult     | CNN          | 0.8573   | 0.6792 | 16.4s  |
| CIFAR-100 | MLP          | 0.2049   | —      | 1100.0s|
| CIFAR-100 | CNN          | 0.4826   | —      | 784.0s |
| CIFAR-100 | ViT (Bonus)  | 0.4502   | —      | 601.7s |
| PCam      | MLP          | 0.6815   | 0.6883 | 152.3s |
| PCam      | CNN          | 0.8474   | 0.8399 | 987.3s |

---

# Key Findings

 1. Tabular Data: MLP is sufficient

On Adult Income, MLP and CNN achieved similar accuracy. Because tabular features lack spatial meaning, CNN inductive bias provides little benefit while increasing computation.

 2. Image Tasks: CNN dominates

On CIFAR-100, CNN accuracy more than doubled the MLP baseline. Convolutions effectively capture local visual structure that MLPs cannot model efficiently.

 3. Medical Imaging: Spatial bias is critical

On PCam, CNN dramatically outperformed MLP in both Accuracy and F1. Histopathology classification depends heavily on local tissue morphology, which CNN receptive fields model well.

 4. Architecture data alignment matters

Results strongly support that **matching model inductive bias to data modality is essential for performance and efficiency**.

---

Final takeaway:
There is no universally best neural network architecture. Choosing a model whose inductive bias aligns with the underlying data structure is critical for achieving both high accuracy and computational efficiency.

---

##  How to Run

 1. Clone repo

```bash
git clone https://github.com/MominaAmjad24/deep-learning-datasets-architectures-benchmark.git
cd deep-learning-datasets-architectures-benchmark
```

 2. Create environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

 3. Run experiments

Adult:
```bash
python train.py
```

CIFAR-100:
```bash
python train_cifar100.py
```

PCam:
```bash
python train_pcam.py
```

Config-driven runner (flexible):
Edit configs/config.yaml to change dataset/architecture/hyperparameters and run:
```bash
python run_experiment.py
```

Bonus scripts (plots / params / visualization):
```bash
python -m analysis.plot_learning_curves
python -m analysis.param_count_vs_perf
python -m analysis.visualize_weights
```
note: python analysis/visualize_weights.py was running into an error as it was setting Python’s import path to the analysis/ folder (not the project root), so it couldn't see models.

---

#  Reproducibility Notes

- Random seed fixed to 42
- Apple MPS used when available
- PCam uses subset for tractable runtime
- Data directories excluded via `.gitignore`

---

#  Repository Structure

```
analysis/
configs/
datasets/
models/
utils/
.gitignore
README.txt
requirements.txt
run_experiment.py
train.py
train_cifar100.py
train_vit_cifar100.py
train_pcam.py

```

---

#  Conclusion

This study demonstrates that no single neural architecture is universally optimal. Performance depends strongly on the relationship between model inductive bias and data structure. MLPs remain competitive for tabular problems, while CNNs are essential for spatial data such as natural and medical images.

---
