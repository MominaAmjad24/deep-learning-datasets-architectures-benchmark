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
The CNN was slower because convolution layers perform many more computations than the simple linear layers used in an MLP. Convolutions are designed to scan local spatial patterns (like pixels in images), but tabular features do not have meaningful spatial relationships. As a result, the CNN does extra work that does not improve accuracy, leading to higher training time without significant performance gains.
Interpretation:
For structured tabular features, the flexibility of fully connected layers is sufficient. Introducing convolution increases runtime (≈8× slower in this experiment) without improving performance, demonstrating that architectural complexity must match data structure to be efficient.

 2. Image Tasks: CNN dominates

On CIFAR-100, the CNN accuracy (0.4826) more than doubled the MLP baseline (0.2049). This large gap occurs because CNNs are specifically designed to work with image data.

CNNs use convolutional filters that focus on local pixel neighborhoods, which allows them to detect edges, textures, and shapes that are important for image recognition. They also use weight sharing, meaning the same filter is applied across the whole image. This makes CNNs both more efficient and better at learning visual patterns that repeat in different locations.

In contrast, the MLP treats the image as a flat vector and loses the spatial relationships between pixels. Because it has no built-in understanding of locality or structure, it must learn these relationships from scratch, which is much harder and less efficient. As a result, the MLP performs significantly worse on image classification.

Interpretation:
For natural images, models with spatial inductive bias like CNNs are far better suited than fully connected networks.

 3. Medical Imaging: Spatial bias is critical

On PCam, the CNN achieved 0.8474 accuracy and 0.8399 F1, much higher than the MLP baseline. Histopathology image classification relies heavily on detecting local tissue textures and morphological patterns, which are naturally spatial.

CNNs are well suited for this task because their convolutional filters focus on small regions of the image and build hierarchical features (edges → textures → tissue structures). This allows the model to capture subtle visual cues that indicate the presence of tumors.

In contrast, the MLP flattens the image and loses spatial organization. Without built-in locality, it struggles to learn fine-grained texture patterns efficiently, leading to much lower performance.

Interpretation:
For medical imaging tasks where local structure is crucial, CNN inductive bias provides a major advantage in both accuracy and F1 score.

4. Bonus: ViT is competitive on images

The Vision Transformer (ViT) achieved 0.4502 accuracy on CIFAR-100, far above the MLP and reasonably close to the CNN. Unlike CNNs, ViT models images by splitting them into patches and using self attention to learn relationships between patches across the entire image.

This global attention mechanism allows the model to capture long-range dependencies that CNNs may learn more slowly. However, on small images like CIFAR-100 (32×32), CNNs still have an advantage because their strong locality bias is very efficient for small-scale visual patterns.

Additionally, ViTs typically benefit from larger datasets and heavier tuning. With limited data and compute, CNNs often remain slightly superior.

Interpretation:
Attention-based models are powerful for image understanding, but CNNs remain highly effective and efficient for small image benchmarks.

 4. Architecture data alignment matters

Across all experiments, the results consistently show that model performance depends strongly on how well the architecture’s inductive bias matches the structure of the data.
   - For tabular data, simple MLPs are sufficient and more efficient.
   - For natural and medical images, CNNs provide large performance gains due to spatial bias.
   - Attention-based models like ViT are competitive on images but may require more data or tuning to surpass CNNs on small datasets.

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
utils.py
```

---

#  Conclusion

This study demonstrates that no single neural architecture is universally optimal. Performance depends strongly on the relationship between model inductive bias and data structure. MLPs remain competitive for tabular problems, while CNNs are essential for spatial data such as natural and medical images.

---


