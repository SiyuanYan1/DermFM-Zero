<div align="center">

# DermFM-Zero

### A Vision-Language Foundation Model for Dermatology

**Enabling Zero-Shot Clinical Collaboration & Automated Concept Discovery**

---

DermFM-Zero is the first multimodal foundation model to provide effective clinical decision support across primary care and specialty settings without fine-tuning. Beyond diagnosis, it unlocks emerging capabilities in automated concept discovery, advancing AI-assisted dermatology.

[![Paper](https://img.shields.io/badge/arXiv-2602.10624-b31b1b.svg)](https://arxiv.org/abs/2602.10624)
[![Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/redlessone/DermFM-Zero)
[![License](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[📘 Paper](https://arxiv.org/abs/2602.10624) | [🚀 Quick Start](#-quick-start) | [📊 Benchmarks](#-benchmark-results) | [🧪 Tasks](#-evaluation-tasks) | [💬 Issues](https://github.com/SiyuanYan1/DermFM-Zero/issues)

</div>

> 🔒 **Availability**:  The DermFM-Zero model weights are currently private and are expected to be released in the coming months.

## 📑 Table of Contents

- [✨ Highlights](#-highlights)
- [📰 Updates](#-updates)
- [📊 Benchmark Results](#-benchmark-results)
- [📂 Repository Structure](#-repository-structure)
- [🚀 Quick Start](#-quick-start)
- [🧪 Evaluation Tasks](#-evaluation-tasks)
  - [Task1: Zero-shot Classification](#task1-zero-shot-classification)
  - [Task2: Zero-shot Cross-modal Retrieval](#task2-zero-shot-cross-modal-retrieval)
  - [Task3: Linear Probing](#task3-linear-probing)
  - [Task4: Multimodal Finetuning](#task4-multimodal-finetuning)
  - [Task5: Visual Question Answering (VQA)](#task5-visual-question-answering-vqa)
  - [Task6: Automated Concept Discovery](#task6-automated-concept-discovery)
- [🧠 Reader Studies](#-reader-studies)
- [🧹 Data Deduplication / Leakage Analysis](#-data-deduplication--leakage-analysis)
- [📈 Statistic for Benchmarking](#-statistic-for-benchmarking)
- [👥 Contributors](#-contributors)
- [⚖️ License](#%EF%B8%8F-license)
- [📧 Contact](#-contact)
- [📚 Citation](#-citation)
## ✨ Highlights

🏆 **State-of-the-art Performance**: Achieves 73.20% average accuracy across 7 zero-shot classification benchmarks

🔍 **Multimodal Fusion**: Supports clinical images, dermoscopic images, and patient metadata

🧠 **Interpretable AI**: Built-in concept discovery with Sparse Autoencoders (SAE)

🌍 **Multi-center Validation**: Evaluated on datasets from Austria, Brazil, Korea, Portugal, and more

## 📰 Updates

- **2026-08-28** · 🔓 Released the full de-identified RS1 reader-level dataset (`reader_studies/reader_study_rs1/real_data/`) under an approved MUHREC amendment — all published RS1 results are now reproducible from the repository.
- **2026-08-28** · 🔓 Released the full de-identified RS2B reader-level dataset (`reader_studies/reader_study_rs2b/real_data/`) — all published RS2B results are now reproducible from the repository.
- **2026-06-01** · 📊 Released `statistic_reproduce/` — unified bootstrap 95% CI pipeline for zero-shot classification and linear-probing benchmark tables, with example prediction CSVs and reference outputs.
- **2026-05-31** · 🧪 Released `VQA/` — Visual Question Answering preprocessing and evaluation pipeline.
- **2026-05-31** · 🧹 Released `data_deduplication/` — image-level deduplication scripts and reports.
- **2026-05-25** · 🧠 Released `reader_studies/` — three multinational reader studies (RS1, RS2A, RS2B) with paired-design statistical pipelines.
- **2025-12-10** · 🧬 Released `automated-concept-discovery/` — sparse-autoencoder + concept-bottleneck-model pipeline.
- **2025-09-13** · 🚀 Initial public release.

## 📊 Benchmark Results

DermFM-Zero demonstrates state-of-the-art performance across diverse benchmarks.

**Modality:** D = Dermoscopic, C = Clinical

### Zero-Shot Classification Performance

| Model | HAM<br>(7-D) | PAD<br>(6-C) | ISIC2020<br>(2-D) | PH2<br>(2-C) | SNU<br>(134-C) | SD-128<br>(128-C) | Daffodil<br>(5-D) | **Average** |
|-------|:----:|:----:|:-----:|:---:|:------:|:-------:|:--------:|:------:|
| **Task** | Skin Cancer | Skin Cancer | Mel Det. | Mel Det. | DDX | DDX | Rare DX | - |
| **Country/Inst** | Austria | Brazil | Multi-center | Portugal | Korea | Multi-center | Multi-center | - |
| **Metric** | ACC | ACC | AUROC | AUROC | ACC | ACC | ACC | - |
| CLIP-Large [[1]](https://proceedings.mlr.press/v139/radford21a) | 0.2754 | 0.3839 | 0.4772 | 0.3855 | 0.0857 | 0.1210 | 0.5304 | 0.3227 |
| BiomedCLIP [[2]](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | 0.6347 | 0.4512 | 0.7305 | 0.8441 | 0.0966 | 0.1153 | 0.5785 | 0.4930 |
| MONET [[3]](https://www.nature.com/articles/s41591-024-02887-x) | 0.3347 | 0.4729 | 0.6940 | 0.8370 | 0.1414 | 0.2028 | 0.7607 | 0.4919 |
| MAKE [[4]](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_35) | 0.4551 | 0.5857 | 0.8141 | 0.9095 | 0.3260 | 0.3886 | 0.7785 | 0.6082 |
| DermLIP-ViT-B-16 [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.6813 | 0.6074 | 0.8235 | 0.8285 | 0.2532 | 0.2783 | 0.7246 | 0.5995 |
| DermLIP-PanDerm [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.6281 | 0.6247 | 0.7876 | 0.7975 | 0.3332 | 0.3822 | 0.7812 | 0.6192 |
| **DermFM-Zero (Ours)** | **0.7957** | **0.6941** | **0.8663** | **0.9304** | **0.4450** | **0.5075** | **0.8848** | **0.7320** |

### Few-Shot Learning (10% training data)

Evaluation with limited labeled data to assess data efficiency and representation quality.

| Model | HAM<br>(7-class) | ISIC'20<br>(Melanoma) | PAD<br>(6-class) | SD-128<br>(128-class) | **Average** |
|-------|:----:|:--------:|:---:|:--------:|:------:|
| **Task** | Skin Cancer | Mel Det. | Skin Cancer | DDX | - |
| **Metric** | ACC | AUROC | ACC | ACC | - |
| CLIP [[1]](https://proceedings.mlr.press/v139/radford21a) | 0.7798 | 0.7828 | 0.6161 | 0.3146 | 0.6233 |
| BiomedCLIP [[2]](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | 0.6959 | 0.4318 | 0.6499 | 0.2541 | 0.5079 |
| MONET [[3]](https://www.nature.com/articles/s41591-024-02887-x) | 0.8064 | 0.8036 | 0.6464 | 0.2747 | 0.6328 |
| BiomedGPT [[6]](https://arxiv.org/abs/2305.17100) | 0.7565 | 0.7838 | 0.5249 | 0.1694 | 0.5586 |
| PanDerm [[7]](https://www.nature.com/articles/s41591-024-02887-x) | 0.7898 | 0.8417 | 0.6508 | 0.3483 | 0.6577 |
| DermLIP-ViT-B-16 [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.8157 | 0.8058 | 0.6594 | 0.3552 | 0.6590 |
| DermLIP-PanDerm [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.8184 | **0.8707** | 0.6529 | 0.3637 | 0.6764 |
| MAKE [[4]](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_35) | 0.8257 | 0.7813 | 0.6790 | 0.3986 | 0.6712 |
| DINOv3-ViT-L16 [[8]](https://ai.meta.com/dinov3/) | 0.7705 | 0.8310 | 0.6573 | 0.3018 | 0.6401 |
| DINOv3-ViT-7B [[8]](https://ai.meta.com/dinov3/) | 0.7871 | 0.8226 | **0.6985** | 0.3345 | 0.6607 |
| **DermFM-Zero (Ours)** | **0.8416** | 0.8687 | 0.6855 | **0.4007** | **0.6991** |

### Zero-Shot Cross-Modal Retrieval (Mean Recall)

Evaluated on Derm1M validation set (n = 9,806) and SkinCap (n = 3,989).

| Model | Derm1M<br>I→T | Derm1M<br>T→I | SkinCap<br>I→T | SkinCap<br>T→I | Average |
|-------|:----:|:----:|:----:|:----:|:----:|
| CLIP-Large [[1]](https://proceedings.mlr.press/v139/radford21a) | 0.122 | 0.104 | 0.174 | 0.127 | 0.132 |
| BiomedCLIP [[2]](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | 0.188 | 0.179 | 0.187 | 0.175 | 0.182 |
| MONET [[3]](https://www.nature.com/articles/s41591-024-02887-x) | 0.171 | 0.159 | 0.215 | 0.203 | 0.187 |
| DermFM-Zero (Ours) | **0.457** | **0.454** | **0.369** | **0.349** | **0.407** |

## 📂 Repository Structure
```
DermFM-Zero/
├── src/                              # Core models and modules (bundled open_clip fork)
├── script/                           # Experiment shell scripts (one per task)
├── examples/                         # Quick-start notebook + sample image
├── automated-concept-discovery/      # SAE & CBM implementation
├── linear_probe/                     # Linear probe utilities
├── multimodal_finetune/              # Multimodal classification fine-tuning code
├── VQA/                              # VQA fine-tuning + preprocessing (Derm7pt-VQA, SkinCap-VQA)
├── reader_studies/                   # Three multinational clinical reader studies (RS1, RS2A, RS2B)
├── data_deduplication/               # SSCD-based train/eval leakage analysis pipeline
├── statistic_reproduce/              # Bootstrap 95% CI pipeline for benchmark tables
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```
## 🚀 Quick Start

### Installation
```bash
git clone git@github.com:SiyuanYan1/DermFM-Zero.git
cd DermFM-Zero

conda create -n dermfm-zero python=3.9.20
conda activate dermfm-zero
pip install -r requirements.txt
```

### Model Access

DermFM-Zero weights are currently hosted as a private repository on the Hugging Face Hub at `redlessone/DermFM-Zero`. The read-only access token is at present shared only with internal collaborators and authorised reviewers; it will be released openly once the manuscript is published.

If you have been provided with the token, set it as the `HF_TOKEN` environment variable before running any code:

```bash
# Option 1: set the token as an environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Option 2: log in interactively (paste the token when prompted)
huggingface-cli login
```

**Troubleshooting**: if you see a `401 Unauthorized` error, verify `huggingface_hub >= 0.20` is installed (`pip install -U huggingface_hub`) and the token has been set in the same shell session you run the code in.

### Download Data

Download benchmark data from [Google Drive](https://drive.google.com/drive/folders/1lH53xfkdkBLSMDn6WShjCRV9gi7xQTvM?usp=drive_link) and unzip to the `data/` folder.

Expected directory structure:
```
data/
├── zero-shot-classification/
├── zero-shot-retrieval/
├── linear_probe/
├── multimodal_finetune/               # classification finetune source datasets
├── VQA/                               # self-contained VQA bundle (images + meta + preprocessing inputs)
└── automated-concept-discovery/
```

### Quick Example

Verify your setup with a minimal zero-shot inference (run from the repo root):

```python
import sys, torch
from PIL import Image
sys.path.insert(0, "src")            # use the bundled open_clip fork
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:redlessone/DermFM-Zero")
tokenizer = open_clip.get_tokenizer("hf-hub:redlessone/DermFM-Zero")
model.eval()

image = preprocess(Image.open("examples/PAT_8_15_820.png")).unsqueeze(0)
classnames = ["nevus", "basal cell carcinoma", "actinic keratosis",
              "seborrheic keratosis", "squamous cell carcinoma", "melanoma"]
text = tokenizer([f"This is a skin image of {c}" for c in classnames])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features  = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features  /= text_features.norm(dim=-1, keepdim=True)

probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(classnames[probs.argmax().item()])    # → basal cell carcinoma
```

For a more interactive walkthrough, see [`examples/zero-shot-classification.ipynb`](examples/zero-shot-classification.ipynb).

## 🧪 Evaluation Tasks

### Task1: Zero-shot Classification

Evaluate DermFM-Zero on 7 dermatology datasets without fine-tuning.

**Benchmark datasets**: HAM, PAD, ISIC2020, PH2, SNU, SD-128, Daffodil
```bash
# Quick run
bash script/zero-shot-eval/DermFM-Zero-zs-classification.sh

# Or detailed command
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
   --model 'hf-hub:redlessone/DermFM-Zero'
```

**Custom Dataset Evaluation**

Prepare a CSV file:
```csv
image_path,label,diag
examples/image1.png,0,melanoma
examples/image2.png,1,nevus
```

Configure class names in [`src/open_clip/zero_shot_metadata.py`](src/open_clip/zero_shot_metadata.py#L13):
```python
customized_CLASSNAMES = ['melanoma', 'nevus', 'basal cell carcinoma']
```

Run evaluation:
```bash
python src/main.py \
   --dataset-type csv \
   --batch-size 1024 \
   --csv-label-key label \
   --csv-img-key image_path \
   --zeroshot_eval_custom your_data.csv \
   --model 'hf-hub:redlessone/DermFM-Zero'
```

### Task2: Zero-shot Cross-modal Retrieval

Evaluate image-text retrieval performance on Derm1M Hold-out and SkinCAP datasets.
```bash
bash script/zero-shot-eval/DermFM-Zero-zs-retrieval.sh
```

### Task3: Linear Probing

Evaluate feature quality by training linear classifiers on frozen features.

**Datasets**: HAM, ISIC2020, PAD, SD-128
```bash
bash script/linear-probe/DermFM-Zero-lp-eval.sh
```

### Task4: Multimodal Finetuning

Fine-tune DermFM-Zero with clinical images, dermoscopic images, and patient metadata.

**Dataset modalities:**
- **Derm7pt**: Clinical + Dermoscopic + Metadata
- **MILK-11**: Clinical + Dermoscopic  
- **PAD-UFES-20**: Clinical + Metadata
```bash
cd multimodal_finetune

# Choose dataset
bash ../script/multimodal_finetune/Derm7pt\(C+D+M\).sh
bash ../script/multimodal_finetune/MILK11\(C+D\).sh  
bash ../script/multimodal_finetune/PAD\(C+M\).sh
```

**Key hyperparameters:**
- `--model_name`: Base model (e.g., `DermFM-Zero`)
- `--dataset_name`: Target dataset (`Derm7pt`, `MILK-11`, `PAD`)
- `--epochs`: Training epochs (default: 50)
- `--batch_size`: Batch size per GPU (default: 32)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--use_cli`, `--use_derm`, `--use_meta`: Enable modalities

Metadata is converted to text prompts - see [`multimodal_finetune/dataset/prompt.py`](multimodal_finetune/dataset/prompt.py).

Results are saved to `multimodal_finetune-result/`.

### Task5: Visual Question Answering (VQA)

Fine-tune DermFM-Zero on dermatology VQA benchmarks (Derm7pt-VQA and SkinCap-VQA). Each shell script handles preprocessing on first run and is skipped on subsequent runs.

```bash
cd VQA

# Derm7pt-VQA (49 answers; Clinical + Dermoscopic + Metadata-question)
bash ../script/VQA/Derm7pt-VQA.sh

# SkinCap-VQA (188 answers; clinical photos + question)
bash ../script/VQA/SkinCap-VQA.sh
```

The `VQA/preprocessing/` step rebuilds the train/val/test splits from the
official upstream artefacts under
`data/VQA/preprocessing_inputs/` (Derm7pt `meta.csv` + the published
case-split manifest, and the DermVQA4 MCQA JSONs). See
[`VQA/preprocessing/README.md`](VQA/preprocessing/README.md) for the full
pipeline.

Results are saved to `VQA-result/{derm7pt,SkinCap}-VQA/`.

### Task6: Automated Concept Discovery

Discover interpretable concepts using Sparse Autoencoders (SAE) and build Concept Bottleneck Models (CBM).

**Prerequisites:**
```bash
bash script/automated-concept-discovery/env_setup.sh
```

Download SAE checkpoint from [Google Drive](https://drive.google.com/file/d/1OM16Bultiy8ZEoEw32ppvLvreraS4VB1/view?usp=sharing) to `automated-concept-discovery-result/SAE-embeddings/`.

**Quick run:**
```bash
bash script/automated-concept-discovery/dermoscopic-melanoma-classification/DermFM-Zero-SAE.sh
```

**Step-by-step pipeline:**
```bash
# Step 1: Extract visual features
cd src
python export_visual_features.py \
    --model_name hf-hub:redlessone/DermFM-Zero \
    --csv_path ../data/automated-concept-discovery/clinical-malignant/meta.csv \
    --data_root ../data/automated-concept-discovery/clinical-malignant/final_images/ \
    --img_col ImageID \
    --batch_size 2048 \
    --output_dir ../automated-concept-discovery-result/clinical-malignant/
cd ..

# Step 2: Extract SAE concepts
python automated-concept-discovery/0_extract_sae_activations.py \
  --checkpoint automated-concept-discovery-result/SAE-embeddings/autoencoder.pth \
  --embeddings automated-concept-discovery-result/clinical-malignant/all_embeddings.npy \
  --output automated-concept-discovery-result/clinical-malignant/learned_activation.npy

# Step 3: Train CBM classifier
python automated-concept-discovery/1_train_clf_binary-class.py \
  --csv data/automated-concept-discovery/clinical-malignant/meta.csv \
  --embeddings automated-concept-discovery-result/clinical-malignant/learned_activation.npy \
  --image_col ImageID \
  --output automated-concept-discovery-result/clinical-malignant/
```

**Analysis tools:**
- Concept Intervention: [`script/automated-concept-discovery/ISIC-intervention/`](script/automated-concept-discovery/ISIC-intervention/)
- Global Explanation: [`automated-concept-discovery/global-explanation/`](automated-concept-discovery/global-explanation/)
- Concept Retrieval: [`automated-concept-discovery/concept-retrieval/`](automated-concept-discovery/concept-retrieval/)

Results are saved to `automated-concept-discovery-result/`.

## 🧠 Reader Studies

Three multinational clinical reader studies that evaluate DermFM-Zero in collaborative diagnostic workflows: primary care (RS1), specialist benchmarking (RS2A), and specialist collaborative diagnosis (RS2B). Each subfolder is self-contained with code and real data: RS1 and RS2B ship full de-identified reader-level datasets, and RS2A ships its complete session-level analysis dataset.

| Study | Setting | Design | Readers | Cases | Real data |
|-------|---------|--------|---------|-------|-----------|
| RS1   | Primary care (CN + AU/EN)            | Within-subject, paired   | 38 PCPs   | 146     | **De-identified reader-level data included** |
| RS2A  | Specialist benchmark (TODIV)         | Independent cohort       | 652     | 1,117   | Aggregated scores included |
| RS2B  | Specialist collab. (DermaChallenge)  | Within-subject, paired   | 71        | 1,048   | **De-identified reader-level data included** |

```bash
# Quick run (RS1, de-identified reader-level data included)
cd reader_studies/reader_study_rs1
python rs1_statistical_analysis.py --real
```

See [`reader_studies/README.md`](reader_studies/README.md) for full documentation, study designs, statistical methods, and data sharing policy.

## 🧹 Data Deduplication / Leakage Analysis

Quantifies near-duplicate overlap between the DermFM-Zero pretraining corpus and every downstream evaluation set, using SSCD copy-detection embeddings + top-1 cosine search (cosine ≥ 0.75 flagged as potential leakage). The pipeline ships with aggregate overlap statistics; the pretraining image bank itself is private.

| Pipeline stage | What it does | Output |
|---|---|---|
| `embed.py`   | SSCD embeddings (ResNet-50 + GeM, 512-d, L2-normalised) | `*.npy` per dataset |
| `overlap.py` | Top-1 cosine search vs pretrain bank | `overlaps.csv`, `overlap_summary.csv` |
| `run.sh`     | End-to-end driver across all evaluation sets | full `results/` tree |

```bash
# Quick run
cd data_deduplication
pip install -r requirements.txt
bash run.sh
```

See [`data_deduplication/README.md`](data_deduplication/README.md) for the full pipeline, CLI flags, and the per-dataset overlap-rate report.

## 📈 Statistic for Benchmarking

Unified bootstrap 95% CI pipeline that reproduces the zero-shot classification and linear-probing benchmark tables from per-image prediction CSVs. A single script supports two `--task` modes; example prediction CSVs and reference outputs are bundled for one-command validation.

| Task | Input format | Output |
|---|---|---|
| `zero_shot`     | per-image softmax CSV per `<dataset>/<model>.csv`       | `model_comparison_results_comprehensive.csv` |
| `linear_probe`  | per-image softmax CSV per `<dataset>_<pct>pct/<model>/` | `lp_results_<pct>percent_bootstrap.csv`      |

```bash
# Quick run (example data bundled)
cd statistic_reproduce
python bootstrap_ci.py --task zero_shot --data-root ./examples/zero_shot --output-dir ./out_zs
python bootstrap_ci.py --task lp        --data-root ./examples/linear_probe --output-dir ./out_lp --fractions 100
```

See [`statistic_reproduce/README.md`](statistic_reproduce/README.md) for input schema, CLI flags, and how to run on the full benchmark prediction set.



## 👥 Contributors

- [Siyuan Yan](https://scholar.google.com/citations?user=LGcOLREAAAAJ&hl=en)
- [Xieji Li](https://scholar.google.com/citations?user=X50rN1oAAAAJ&hl=en)

## ⚖️ License

The model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial academic research purposes with proper attribution.

## 📧 Contact

**Siyuan Yan** - Research Fellow, Monash University  
📧 Email: siyuan.yan@monash.edu  

## 📚 Citation

If you find DermFM-Zero useful, please cite:
```bibtex
@misc{yan2026visionlanguagefoundationmodelzeroshot,
      title={A Vision-Language Foundation Model for Zero-shot Clinical Collaboration and Automated Concept Discovery in Dermatology}, 
      author={Siyuan Yan and Xieji Li and Dan Mo and Philipp Tschandl and Yiwen Jiang and Zhonghua Wang and Ming Hu and Lie Ju and Cristina Vico-Alonso and Yizhen Zheng and Jiahe Liu and Juexiao Zhou and Camilla Chello and Jen G. Cheung and Julien Anriot and Luc Thomas and Clare Primiero and Gin Tan and Aik Beng Ng and Simon See and Xiaoying Tang and Albert Ip and Xiaoyang Liao and Adrian Bowling and Martin Haskett and Shuang Zhao and Monika Janda and H. Peter Soyer and Victoria Mar and Harald Kittler and Zongyuan Ge},
      year={2026},
      eprint={2602.10624},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.10624}, 
}
```

**Related work:**
```bibtex
@article{yan2025multimodal,
  title={A multimodal vision foundation model for clinical dermatology},
  author={Yan, Siyuan and Yu, Zhen and Primiero, Clare and Vico-Alonso, Cristina and Wang, Zhonghua and Yang, Litao and Tschandl, Philipp and Hu, Ming and Ju, Lie and Tan, Gin and others},
  journal={Nature Medicine},
  pages={1--12},
  year={2025},
  publisher={Nature Publishing Group}
}
```
```bibtex
@inproceedings{yan2025derm1m,
  title={Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge},
  author={Yan, Siyuan and others},
  booktitle={ICCV},
  year={2025}
}
```
