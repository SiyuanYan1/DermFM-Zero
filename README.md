<div align="center">

# PanDerm-2

**Next-generation Dermatology Foundation Model**

*Enabling Zero-shot Clinician Collaboration and Automated Concept Discovery*

[![Paper](https://img.shields.io/badge/Paper-Nature%20Medicine-red)](xxx)
[![Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/redlessone/PanDerm2)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[📘 Documentation](xxx) | [🚀 Quick Start](#-quick-start) | [📊 Benchmarks](#-benchmark-results) | [💬 Discussion](xxx)

</div>

---

## 📑 Table of Contents

- [🌟 Highlights](#-highlights)
- [📊 Benchmark Results](#-benchmark-results)
- [🚀 Quick Start](#-quick-start)
- [📥 Installation](#-installation)
- [🔧 Evaluation Tasks](#-evaluation-tasks)
  - [Task 1: Zero-shot Classification](#task-1-zero-shot-classification)
  - [Task 2: Cross-modal Retrieval](#task-2-zero-shot-cross-modal-retrieval)
  - [Task 3: Linear Probing](#task-3-linear-probing)
  - [Task 4: Multimodal Finetuning](#task-4-multimodal-finetune)
  - [Task 5: Concept Discovery](#task-5-automated-concept-discovery)
- [📁 Repository Structure](#-repository-structure)
- [📝 Citation](#-citation)

---

## 🌟 Highlights

- 🏆 **State-of-the-art Performance**: Achieves 73.20% average accuracy across 7 zero-shot classification benchmarks
- 🔍 **Multimodal Fusion**: Supports clinical images, dermoscopic images, and metadata
- 🧠 **Interpretable AI**: Built-in concept discovery with Sparse Autoencoders (SAE)
- 🌍 **Multi-center Validation**: Evaluated on datasets from Austria, Brazil, Korea, Portugal, and more
- 🔧 **Easy to Use**: Simple API for zero-shot classification and retrieval

---

## 📊 Benchmark Results

### Zero-Shot Classification Performance

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="7">Datasets (Modality)</th>
    <th rowspan="2">Average</th>
  </tr>
  <tr>
    <th>HAM<br>(7-D)</th>
    <th>PAD<br>(6-C)</th>
    <th>ISIC2020<br>(2-D)</th>
    <th>PH2<br>(2-C)</th>
    <th>SNU<br>(134-C)</th>
    <th>SD-128<br>(128-C)</th>
    <th>Daffodil<br>(5-D)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="9" align="center"><b>Baselines</b></td>
  </tr>
  <tr>
    <td>CLIP-Large</td>
    <td>0.2754</td>
    <td>0.3839</td>
    <td>0.4772</td>
    <td>0.3855</td>
    <td>0.0857</td>
    <td>0.1210</td>
    <td>0.5304</td>
    <td>0.3227</td>
  </tr>
  <tr>
    <td>BiomedCLIP</td>
    <td>0.6347</td>
    <td>0.4512</td>
    <td>0.7305</td>
    <td>0.8441</td>
    <td>0.0966</td>
    <td>0.1153</td>
    <td>0.5785</td>
    <td>0.4930</td>
  </tr>
  <tr>
    <td>MONET</td>
    <td>0.3347</td>
    <td>0.4729</td>
    <td>0.6940</td>
    <td>0.8370</td>
    <td>0.1414</td>
    <td>0.2028</td>
    <td>0.7607</td>
    <td>0.4919</td>
  </tr>
  <tr>
    <td>DermLIP-PanDerm</td>
    <td>0.6281</td>
    <td>0.6247</td>
    <td>0.7876</td>
    <td>0.7975</td>
    <td>0.3332</td>
    <td>0.3822</td>
    <td>0.7812</td>
    <td>0.6192</td>
  </tr>
  <tr>
    <td colspan="9" align="center"><b>Ours</b></td>
  </tr>
  <tr style="background-color: #e8f4f8;">
    <td><b>PanDerm-2</b></td>
    <td><b>0.7957</b></td>
    <td><b>0.6941</b></td>
    <td><b>0.8663</b></td>
    <td><b>0.9304</b></td>
    <td><b>0.4450</b></td>
    <td><b>0.5075</b></td>
    <td><b>0.8848</b></td>
    <td><b>0.7320</b></td>
  </tr>
</tbody>
</table>

> **Note**: D = Dermoscopic, C = Clinical | Full benchmark results in [BENCHMARKS.md](BENCHMARKS.md)

### Few-Shot Learning (10% training data)

| Model | HAM (7) | ISIC'20 (Mel) | PAD (6) | SD-128 (128) | Average |
|-------|:-------:|:-------------:|:-------:|:------------:|:-------:|
| CLIP | 0.7798 | 0.7828 | 0.6161 | 0.3146 | 0.6233 |
| DermLIP-PanDerm | 0.8184 | 0.8707 | 0.6529 | 0.3637 | 0.6764 |
| DINOv3-ViT-7B | 0.7871 | 0.8226 | **0.6985** | 0.3345 | 0.6607 |
| **PanDerm-2** | **0.8416** | **0.8687** | 0.6855 | **0.4007** | **0.6991** |

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone git@github.com:SiyuanYan1/PanDerm-2.git
cd PanDerm-2

# Create environment
conda create -n panderm python=3.9.20
conda activate panderm
pip install -r requirements.txt
```

### 2. Run Zero-Shot Classification
```python
import torch
from src.model import PanDerm2

# Load model
model = PanDerm2.from_pretrained('hf-hub:redlessone/PanDerm2')

# Classify image
image_path = "examples/skin_lesion.jpg"
diseases = ["melanoma", "basal cell carcinoma", "actinic keratosis"]

predictions = model.zero_shot_classify(image_path, diseases)
print(predictions)  # {'melanoma': 0.85, 'basal cell carcinoma': 0.10, ...}
```

📖 **More Examples**: Check out our [interactive notebook](examples/zero-shot-classification.ipynb)

---

## 📥 Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM

### Setup
```bash
# Clone and install
git clone git@github.com:SiyuanYan1/PanDerm-2.git
cd PanDerm-2
pip install -r requirements.txt

# Download data
wget [GOOGLE_DRIVE_LINK] -O data.zip
unzip data.zip -d data/
```

### Expected Data Structure
```
data/
├── zero-shot-classification/     # 7 benchmark datasets
├── zero-shot-retrieval/          # 2 retrieval datasets  
├── linear_probe/                 # 4 linear probe datasets
├── multimodal_finetune/          # 3 multimodal datasets
└── automated-concept-discovery/  # Concept learning datasets
```

---

## 🔧 Evaluation Tasks

<div align="center">

| Task | Description | Datasets | Quick Run |
|:----:|:-----------|:--------:|:---------:|
| [#1](#task-1-zero-shot-classification) | Zero-shot Classification | 7 | ✅ |
| [#2](#task-2-zero-shot-cross-modal-retrieval) | Cross-modal Retrieval | 2 | ✅ |
| [#3](#task-3-linear-probing) | Linear Probing | 4 | ✅ |
| [#4](#task-4-multimodal-finetune) | Multimodal Finetuning | 3 | ✅ |
| [#5](#task-5-automated-concept-discovery) | Concept Discovery | 3 | ✅ |

</div>

---

### Task 1: Zero-shot Classification

<table>
<tr><td>

**📝 Description**

Evaluate PanDerm-2's zero-shot classification performance across 7 diverse dermatology datasets without any fine-tuning.

**📊 Datasets**
- HAM (7 skin cancers, Austria)
- PAD (6 classes, Brazil)  
- ISIC2020 (Melanoma detection)
- PH2 (Melanoma, Portugal)
- SNU (134 classes, Korea)
- SD-128 (128 diseases)
- Daffodil (5 rare diseases)

</td></tr>
</table>

#### Quick Run
```bash
bash script/zero-shot-eval/PanDerm-v2-zs-classification.sh
```

#### Custom Dataset

<table>
<tr><td>

**Step 1: Prepare CSV**
```csv
image_path,label,diag
data/image1.png,0,melanoma
data/image2.png,1,nevus
data/image3.png,0,melanoma
```

</td><td>

**Step 2: Configure Classes**

Edit `src/open_clip/zero_shot_metadata.py`:
```python
customized_CLASSNAMES = [
    'melanoma', 
    'nevus',
    # add more...
]
```

</td></tr>
</table>

**Step 3: Run**
```bash
python src/main.py \
   --dataset-type csv \
   --batch-size 1024 \
   --csv-label-key label \
   --csv-img-key image_path \
   --zeroshot_eval_custom your_data.csv \
   --model 'hf-hub:redlessone/PanDerm2'
```

<details>
<summary><b>📋 Advanced Options</b></summary>
```bash
python src/main.py \
   --val-data="" \
   --dataset-type "csv" \
   --batch-size=1024 \
   --zeroshot-eval1=data/zero-shot-classification/pad-zero-shot-test.csv \
   --zeroshot-eval2=data/zero-shot-classification/HAM-official-7-zero-shot-test.csv \
   --zeroshot-eval3=data/zero-shot-classification/snu-134-zero-shot-test.csv \
   --zeroshot-eval4=data/zero-shot-classification/sd-128-zero-shot-test.csv \
   --zeroshot-eval5=data/zero-shot-classification/daffodil-5-zero-shot-test.csv \
   --zeroshot-eval6=data/zero-shot-classification/ph2-2-zero-shot-test.csv \
   --zeroshot-eval7=data/zero-shot-classification/isic2020-2-zero-shot-test.csv \
   --csv-label-key label \
   --csv-img-key image_path \
   --model 'hf-hub:redlessone/PanDerm2'
```

</details>

---

### Task 2: Zero-shot Cross-modal Retrieval

<table>
<tr><td>

**📝 Description**

Evaluate image-text retrieval performance: given an image, retrieve relevant text descriptions (and vice versa).

**📊 Datasets**
- Derm1M Hold-out (10K samples)
- SkinCAP (Clinical captions)

</td></tr>
</table>

#### Quick Run
```bash
bash script/zero-shot-eval/PanDerm-v2-zs-retrieval.sh
```

---

### Task 3: Linear Probing

<table>
<tr><td>

**📝 Description**

Evaluate the quality of learned representations by training a linear classifier on frozen features.

**📊 Datasets**
- HAM (7 classes)
- ISIC2020 (Binary)
- PAD (6 classes)  
- SD-128 (128 classes)

</td></tr>
</table>

#### Quick Run
```bash
bash script/linear-probe/PanDerm-v2-lp-eval.sh
```

---

### Task 4: Multimodal Finetune

<table>
<tr><td>

**📝 Description**

Fine-tune PanDerm-2 with multiple input modalities: clinical images, dermoscopic images, and patient metadata.

**📊 Dataset Modalities**

| Dataset | Clinical | Dermoscopic | Metadata | Classes |
|---------|:--------:|:-----------:|:--------:|:-------:|
| Derm7pt | ✅ | ✅ | ✅ | 2 |
| MILK-11 | ✅ | ✅ | ❌ | 11 |
| PAD-UFES | ✅ | ❌ | ✅ | 6 |

</td></tr>
</table>

#### Quick Run
```bash
cd multimodal_finetune

# Choose dataset
bash ../script/multimodal_finetune/Derm7pt\(C+D+M\).sh
bash ../script/multimodal_finetune/MILK11\(C+D\).sh  
bash ../script/multimodal_finetune/PAD\(C+M\).sh
```

#### Configuration

<details>
<summary><b>⚙️ Hyperparameters</b></summary>

**Model Configuration**
```bash
--model_name PanDerm-v2         # Base model
--dataset_name Derm7pt          # Target dataset
--hidden_dim 1024               # Hidden dimension
--meta_dim 768                  # Metadata embedding dim
```

**Training Settings**
```bash
--epochs 50                     # Training epochs
--batch_size 32                 # Batch size per GPU
--accum_freq 2                  # Gradient accumulation (effective batch=64)
--learning_rate 1e-5            # Learning rate
```

**Architecture**
```bash
--num_head 8                    # Attention heads (image)
--att_depth 2                   # Attention layers (image)
--meta_num_head 8               # Attention heads (metadata)
--meta_att_depth 4              # Attention layers (metadata)
--fusion "cross attention"      # Image fusion method
--meta_fusion_mode "cross attention"  # Metadata fusion
```

**Modality Flags**
```bash
--use_cli                       # Enable clinical images
--use_derm                      # Enable dermoscopic images  
--use_meta                      # Enable metadata
--use_text_encoder              # Text encoder for metadata
```

</details>

💡 **Metadata as Prompts**: We convert patient metadata (age, sex, location) into text prompts. See [`multimodal_finetune/dataset/prompt.py`](multimodal_finetune/dataset/prompt.py)

📁 **Results**: Saved to `multimodal_finetune-result/`

---

### Task 5: Automated Concept Discovery

<table>
<tr><td>

**📝 Description**

Discover interpretable visual concepts using Sparse Autoencoders (SAE) and build Concept Bottleneck Models (CBM) for explainable AI.

**📊 Experiments**
- Clinical Malignant Classification
- Dermoscopic Melanoma Detection
- ISIC Bias Intervention (hair, ink, ruler)

</td></tr>
</table>

#### Quick Start
```bash
# One-line execution
bash script/automated-concept-discovery/dermoscopic-melanoma-classification/PanDerm-v2-SAE.sh
```

#### Step-by-Step Pipeline

<table>
<tr>
<td width="50%">

**Step 1: Setup Environment**
```bash
bash script/automated-concept-discovery/env_setup.sh
```

**Step 2: Extract Features**
```bash
cd src
python export_visual_features.py \
    --model_name hf-hub:redlessone/PanDerm2 \
    --csv_path ../data/automated-concept-discovery/clinical-malignant/meta.csv \
    --data_root ../data/automated-concept-discovery/clinical-malignant/final_images/ \
    --img_col ImageID \
    --batch_size 2048 \
    --output_dir ../automated-concept-discovery-result/clinical-malignant/
cd ..
```

</td>
<td width="50%">

**Step 3: Extract SAE Concepts**
```bash
python automated-concept-discovery/0_extract_sae_activations.py \
  --checkpoint automated-concept-discovery-result/SAE-embeddings/autoencoder.pth \
  --embeddings automated-concept-discovery-result/clinical-malignant/all_embeddings.npy \
  --output automated-concept-discovery-result/clinical-malignant/learned_activation.npy
```

**Step 4: Train CBM Classifier**
```bash
python automated-concept-discovery/1_train_clf_binary-class.py \
  --csv data/automated-concept-discovery/clinical-malignant/meta.csv \
  --embeddings automated-concept-discovery-result/clinical-malignant/learned_activation.npy \
  --image_col ImageID \
  --output automated-concept-discovery-result/clinical-malignant/
```

</td>
</tr>
</table>

#### Analysis & Visualization

| Tool | Description | Location |
|------|-------------|----------|
| 🔬 **Concept Intervention** | Test concept manipulation effects | [`script/automated-concept-discovery/ISIC-intervention/`](script/automated-concept-discovery/ISIC-intervention/) |
| 📊 **Global Explanation** | Visualize learned concepts | [`automated-concept-discovery/global-explanation/`](automated-concept-discovery/global-explanation/) |
| 🔍 **Concept Retrieval** | Analyze concept patterns | [`automated-concept-discovery/concept-retrieval/`](automated-concept-discovery/concept-retrieval/) |

<details>
<summary><b>⚙️ Configuration Parameters</b></summary>

**Feature Extraction**
- `--model_name`: Model path (default: `hf-hub:redlessone/PanDerm2`)
- `--batch_size`: Processing batch size (default: 2048, reduce if OOM)
- `--num_workers`: Data loading workers (default: 16)
- `--device`: Computing device (`cuda` or `cpu`)

**SAE Activation**
- `--checkpoint`: Pre-trained SAE weights path
- `--embeddings`: Input visual features (.npy)
- `--output`: Output SAE activations path

**Classifier Training**
- `--embeddings`: SAE activations or raw embeddings
- `--csv`: Metadata with labels and splits
- `--gpu`: GPU device ID
- `--output`: Model save directory

</details>

📁 **Results**: Saved to `automated-concept-discovery-result/`

---

## 📁 Repository Structure
```
PanDerm-2/
│
├── 📂 src/                              # Core models and modules
│   ├── main.py                          # Main evaluation script
│   ├── model/                           # Model architectures
│   └── open_clip/                       # CLIP-based components
│
├── 📂 script/                           # Experiment scripts
│   ├── zero-shot-eval/                  # Zero-shot evaluation
│   ├── linear-probe/                    # Linear probing
│   ├── multimodal_finetune/             # Multimodal fine-tuning
│   └── automated-concept-discovery/     # Concept discovery
│
├── 📂 data/                             # Dataset storage
│   ├── zero-shot-classification/
│   ├── zero-shot-retrieval/
│   ├── linear_probe/
│   ├── multimodal_finetune/
│   └── automated-concept-discovery/
│
├── 📂 automated-concept-discovery/      # SAE & CBM implementation
├── 📂 linear_probe/                     # Linear probe utilities
├── 📂 multimodal_finetune/              # Multimodal training code
│
├── 📄 requirements.txt                  # Python dependencies
└── 📄 README.md                         # This file
```

---

## 📝 Citation

If you find PanDerm-2 useful in your research, please cite:
```bibtex
@article{panderm2_2025,
  title={PanDerm-2: Next-generation dermatology foundation model enables zero-shot clinician collaboration and automated concept discovery},
  author={Yan, Siyuan and Ge, Zongyuan},
  journal={Nature Medicine},
  year={2025}
}
```

**Related Works:**
```bibtex
@inproceedings{yan2025derm1m,
  title={Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge},
  author={Yan, Siyuan and others},
  booktitle={ICCV},
  year={2025}
}
```

---

## 🙏 Acknowledgments

This work was supported by:
- Monash University AIM for Health Lab
- Department of Data Science & AI
- [Add funding sources]

Special thanks to all collaborators and data contributors.

---

## 📧 Contact

**Siyuan Yan** - Research Fellow, Monash University  
📧 Email: [siyuan.yan@monash.edu]  
🔗 [Personal Website](xxx) | [Google Scholar](xxx) | [Twitter](xxx)

**Supervisor:** A/Prof. Zongyuan Ge

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ Star us on GitHub — it motivates us a lot!

[⬆ Back to Top](#panderm-2)

</div>
