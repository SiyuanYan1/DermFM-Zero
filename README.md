<div align="center">

# DermFM-Zero

### A Vision-Language Foundation Model for Dermatology

**Zero-shot diagnosis · Clinical collaboration · Automated concept discovery**

[![Paper](https://img.shields.io/badge/arXiv-2602.10624-b31b1b.svg)](https://arxiv.org/abs/2602.10624)
[![Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/redlessone/DermFM-Zero)
[![Report](https://img.shields.io/badge/Technical%20Report-PDF-blue.svg)](docs/DermFM-Zero-Open_Technical_Report.pdf)
[![License](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[🚀 Quick Start](#-quick-start) | [📊 Benchmarks](#-benchmark-results) | [🧪 Tasks](#-evaluation-tasks) | [🧠 Reader Studies](#-reader-studies) | [💬 Issues](https://github.com/SiyuanYan1/DermFM-Zero/issues)

</div>

DermFM-Zero is a dermatology vision–language foundation model pretrained with masked latent modelling on over 3 million dermatological images and contrastive alignment on 1 million image–text pairs. This repository contains the released weights, the evaluation code for every task in the paper, the de-identified reader-study data, and the deduplication and statistics pipelines.

### Key Features

- 🩺 **Zero-Shot Diagnosis**: Classifies 400+ skin conditions without task-specific training
- 🔗 **Multimodal Learning**: Supports combining clinical photographs, dermoscopy and patient metadata for skin disease diagnosis and prognosis
- 🔍 **Cross-Modal Retrieval**: Image-to-text and text-to-image search
- 🧠 **Built-in Interpretability**: Sparse autoencoders expose named clinical concepts and suppress artefact biases (ruler, hair, pen marks) at inference
- 👩‍⚕️ **Clinically Validated**: Improved diagnosis and management for 761 clinicians across three multinational reader studies
- 📐 **Native Resolution**: The released checkpoint accepts any image size and aspect ratio
- 🔓 **Open Release**: Public-data weights, evaluation code for every task, reader-study data and a technical report

## 🔓 Released checkpoint: DermFM-Zero-Open

The image–text corpus used for the checkpoint evaluated in the paper includes in-house pairs that cannot be redistributed, so the released weights (`hf-hub:redlessone/DermFM-Zero`) are a retrained checkpoint, **DermFM-Zero-Open**:

- **Data** — 517,455 publicly available image–text pairs (ISIC, BCN20000, MSKCC, DermNet, Fitzpatrick17k, Derm12345, HIBA and the web/literature sources of Derm1M).
- **Vision encoder** — initialised from the public [PanDerm](https://github.com/SiyuanYan1/PanDerm) ViT-L/16.
- **Native-resolution input (new)** — NaViT-style patch-and-pack replaces the fixed 224 × 224 input, keeping fine dermoscopic structure in the >60% of corpus images larger than 224 px; training uses ScaleJitter (short side 224 to native, long side ≤ 448). All results below are at 224 × 224 for a fair comparison; the native-resolution gain is in the technical report (Table 14).
- **Objective** — [MAKE](https://github.com/XiejiLi/MAGEN-O-MAKE) multi-aspect contrastive alignment (raw caption, disease aspect, concept aspect, sub-captions) plus a [KEP](https://github.com/MAGIC-AI4Med/KEP) knowledge-distillation term from a Derm1M-pretrained text encoder.
- **Performance** — on par with the paper checkpoint: mean zero-shot score 0.691 vs 0.675 over seven benchmarks, higher on linear probing and multimodal fine-tuning, lower on Derm1M retrieval (tables below).

Full details: [technical report](docs/DermFM-Zero-Open_Technical_Report.pdf). Results in the paper refer to the paper checkpoint.

## 📰 Updates

- **2026-09-18** · DermFM-Zero-Open weights released on the Hugging Face Hub with a technical report; benchmark tables list both checkpoints.
- **2026-08-28** · Full de-identified reader-study data released under an approved MUHREC amendment; all reader-study results are reproducible from `reader_studies/`.

## 📊 Benchmark Results

**Modality:** D = dermoscopic, C = clinical. Bold = best per column.

### Zero-shot classification

| Model | HAM<br>(7-D) | PAD<br>(6-C) | ISIC2020<br>(2-D) | PH2<br>(2-D) | SNU<br>(134-C) | SD-128<br>(128-C) | Daffodil<br>(5-D) | **Average** |
|-------|:----:|:----:|:-----:|:---:|:------:|:-------:|:--------:|:------:|
| **Task** | Skin Cancer | Skin Cancer | Mel Det. | Mel Det. | DDX | DDX | Rare DX | - |
| **Metric** | ACC | ACC | Macro F1 | Macro F1 | ACC | ACC | ACC | - |
| CLIP-Large [[1]](https://proceedings.mlr.press/v139/radford21a) | 0.2754 | 0.3839 | 0.4896 | 0.4494 | 0.0857 | 0.1210 | 0.5304 | 0.3336 |
| BiomedCLIP [[2]](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | 0.6347 | 0.4512 | 0.5317 | 0.6292 | 0.0966 | 0.1153 | 0.5785 | 0.4339 |
| MONET [[3]](https://www.nature.com/articles/s41591-024-02887-x) | 0.3347 | 0.4729 | 0.5208 | 0.6774 | 0.1414 | 0.2028 | 0.7607 | 0.4444 |
| MAKE [[4]](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_35) | 0.4551 | 0.5857 | 0.4259 | 0.8222 | 0.3260 | 0.3886 | 0.7785 | 0.5403 |
| DermLIP-PanDerm [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.6281 | 0.6247 | 0.5190 | 0.6799 | 0.3332 | 0.3822 | 0.7812 | 0.5640 |
| **DermFM-Zero (paper checkpoint)** | **0.7957** | 0.6941 | 0.5979 | 0.7998 | **0.4450** | 0.5075 | **0.8848** | 0.6750 |
| **DermFM-Zero-Open (released)** | 0.7858 | **0.7592** | **0.6112** | **0.8708** | 0.4212 | **0.5196** | 0.8686 | **0.6909** |

### Linear probing (10% training data)

| Model | HAM<br>(7-class) | ISIC'20<br>(Melanoma) | PAD<br>(6-class) | SD-128<br>(128-class) | **Average** |
|-------|:----:|:--------:|:---:|:--------:|:------:|
| **Task** | Skin Cancer | Mel Det. | Skin Cancer | DDX | - |
| **Metric** | ACC | AUROC | ACC | ACC | - |
| CLIP [[1]](https://proceedings.mlr.press/v139/radford21a) | 0.7798 | 0.7828 | 0.6161 | 0.3146 | 0.6233 |
| BiomedCLIP [[2]](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | 0.6959 | 0.4318 | 0.6499 | 0.2541 | 0.5079 |
| MONET [[3]](https://www.nature.com/articles/s41591-024-02887-x) | 0.8064 | 0.8036 | 0.6464 | 0.2747 | 0.6328 |
| BiomedGPT [[6]](https://arxiv.org/abs/2305.17100) | 0.7565 | 0.7838 | 0.5249 | 0.1694 | 0.5586 |
| PanDerm [[7]](https://www.nature.com/articles/s41591-025-03747-y) | 0.7898 | 0.8417 | 0.6508 | 0.3483 | 0.6577 |
| DermLIP-ViT-B-16 [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.8157 | 0.8058 | 0.6594 | 0.3552 | 0.6590 |
| DermLIP-PanDerm [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.8184 | 0.8707 | 0.6529 | 0.3637 | 0.6764 |
| MAKE [[4]](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_35) | 0.8257 | 0.7813 | 0.6790 | 0.3986 | 0.6712 |
| DINOv3-ViT-L16 [[8]](https://ai.meta.com/dinov3/) | 0.7705 | 0.8310 | 0.6573 | 0.3018 | 0.6401 |
| DINOv3-ViT-7B [[8]](https://ai.meta.com/dinov3/) | 0.7871 | 0.8226 | 0.6985 | 0.3345 | 0.6607 |
| **DermFM-Zero (paper checkpoint)** | 0.8416 | 0.8687 | 0.6855 | 0.4007 | 0.6991 |
| **DermFM-Zero-Open (released)** | **0.8629** | **0.9008** | **0.7527** | **0.4797** | **0.7490** |

### Zero-shot cross-modal retrieval (mean of R@5, R@10 and R@50; Derm1M held-out n = 9,806, SkinCap n = 3,989)

| Model | Derm1M<br>I→T | Derm1M<br>T→I | SkinCap<br>I→T | SkinCap<br>T→I | Average |
|-------|:----:|:----:|:----:|:----:|:----:|
| CLIP-Large [[1]](https://proceedings.mlr.press/v139/radford21a) | 0.124 | 0.105 | 0.176 | 0.128 | 0.133 |
| BiomedCLIP [[2]](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | 0.191 | 0.181 | 0.188 | 0.176 | 0.184 |
| MONET [[3]](https://www.nature.com/articles/s41591-024-02887-x) | 0.173 | 0.161 | 0.219 | 0.205 | 0.189 |
| **DermFM-Zero (paper checkpoint)** | **0.465** | **0.460** | 0.375 | 0.353 | **0.413** |
| **DermFM-Zero-Open (released)** | 0.365 | 0.367 | **0.400** | **0.382** | 0.378 |

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
├── docs/                             # DermFM-Zero-Open technical report (PDF)
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

## 🚀 Quick Start

```bash
git clone https://github.com/SiyuanYan1/DermFM-Zero.git
cd DermFM-Zero
conda create -n dermfm-zero python=3.9.20
conda activate dermfm-zero
pip install -r requirements.txt
```

The weights load directly from the Hub through the bundled `open_clip` fork (run from the repo root):

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

A notebook version is in [`examples/zero-shot-classification.ipynb`](examples/zero-shot-classification.ipynb). The upstream `open_clip` package builds a different architecture and will not load this checkpoint.

### Benchmark data

Download from [Google Drive](https://drive.google.com/drive/folders/1lH53xfkdkBLSMDn6WShjCRV9gi7xQTvM?usp=drive_link) and unzip to the `data/` folder:

```
data/
├── zero-shot-classification/
├── zero-shot-retrieval/
├── linear_probe/
├── multimodal_finetune/               # classification finetune source datasets
├── VQA/                               # self-contained VQA bundle (images + meta + preprocessing inputs)
└── automated-concept-discovery/
```

## 🧪 Evaluation Tasks

One script per task; results are written to `<task>-result/`. The table is an index; full commands and options for each task follow.

| Task | Datasets | Run |
|---|---|---|
| Zero-shot classification | HAM, PAD, ISIC2020, PH2, SNU, SD-128, Daffodil | `bash script/zero-shot-eval/DermFM-Zero-zs-classification.sh` |
| Zero-shot retrieval | Derm1M held-out, SkinCap | `bash script/zero-shot-eval/DermFM-Zero-zs-retrieval.sh` |
| Linear probing | HAM, ISIC2020, PAD, SD-128 | `bash script/linear-probe/DermFM-Zero-lp-eval.sh` |
| Multimodal fine-tuning | Derm7pt (C+D+meta), MILK-11 (C+D), PAD (C+meta) | `cd multimodal_finetune && bash ../script/multimodal_finetune/<dataset>.sh` |
| Visual question answering | Derm7pt-VQA, SkinCap-VQA | `cd VQA && bash ../script/VQA/<dataset>.sh` |
| Automated concept discovery | F17K+DDI, Derm7pt, ISIC-Intervention | `bash script/automated-concept-discovery/<task>/DermFM-Zero-SAE.sh` (Python 3.10 env via `env_setup.sh`) |

### Task 1: Zero-shot Classification

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
   --zeroshot-eval1=data/zero-shot-classification/pad-zero-shot/meta.csv \
   --zeroshot-eval2=data/zero-shot-classification/HAM-official-7-zero-shot/meta.csv \
   --zeroshot-eval3=data/zero-shot-classification/snu-134-zero-shot/meta.csv \
   --zeroshot-eval4=data/zero-shot-classification/sd-128-zero-shot/meta.csv \
   --zeroshot-eval5=data/zero-shot-classification/daffodil-5-zero-shot/meta.csv \
   --zeroshot-eval6=data/zero-shot-classification/ph2-2-zero-shot/meta.csv \
   --zeroshot-eval7=data/zero-shot-classification/isic2020-2-zero-shot/meta.csv \
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

### Task 2: Zero-shot Cross-modal Retrieval

Evaluate image-text retrieval performance on Derm1M Hold-out and SkinCAP datasets.
```bash
bash script/zero-shot-eval/DermFM-Zero-zs-retrieval.sh
```

### Task 3: Linear Probing

Evaluate feature quality by training linear classifiers on frozen features.

**Datasets**: HAM, ISIC2020, PAD, SD-128
```bash
bash script/linear-probe/DermFM-Zero-lp-eval.sh
```

### Task 4: Multimodal Finetuning

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

### Task 5: Visual Question Answering (VQA)

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

### Task 6: Automated Concept Discovery

Discover interpretable concepts using Sparse Autoencoders (SAE) and build Concept Bottleneck Models (CBM).

**Prerequisites:**
```bash
bash script/automated-concept-discovery/env_setup.sh
```

Download the SAE checkpoint (`autoencoder.pth`, trained on features of DermFM-Zero-Open) from [Google Drive](https://drive.google.com/drive/folders/10BWs5vZu8eaif9Y_kIQnHl36hDSA8-aR) to `automated-concept-discovery-result/SAE-embeddings/`.

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

Three multinational reader studies with code and real de-identified data in [`reader_studies/`](reader_studies/README.md):

| Study | Setting | Design | Readers | Cases | Data |
|-------|---------|--------|---------|-------|------|
| RS1  | Primary care (CN + AU/EN) | Within-subject, paired | 38 PCPs | 146 | Reader-level |
| RS2A | Specialist benchmark (TODIV) | Independent cohort | 652 (1,090 sessions) | 1,117 | Session-level |
| RS2B | Specialist collaboration (DermaChallenge) | Within-subject, paired | 71 | 1,048 | Reader-level |

```bash
# Quick run (RS1, de-identified reader-level data included)
cd reader_studies/reader_study_rs1
python rs1_statistical_analysis.py --real
```

## 🔁 Reproducibility

### Data deduplication / leakage analysis

[`data_deduplication/`](data_deduplication/README.md) quantifies near-duplicate overlap between the pretraining corpus and every evaluation set with SSCD copy-detection embeddings (cosine ≥ 0.75 flagged). `results/` holds the statistics for the paper checkpoint's corpus, as reported in the paper; `results_released/` the same for DermFM-Zero-Open. The pretraining image bank itself is private; users supply their own corpus via the paths in `run.sh`.

```bash
# Quick run
cd data_deduplication
pip install -r requirements.txt
bash run.sh
```

### Bootstrap confidence intervals

[`statistic_reproduce/`](statistic_reproduce/README.md) reproduces the zero-shot and linear-probing benchmark tables with 95% bootstrap CIs from per-image prediction CSVs. Example predictions (paper checkpoint) and reference outputs are bundled for one-command validation.

```bash
# Quick run (example data bundled)
cd statistic_reproduce
python bootstrap_ci.py --task zero_shot --data-root ./examples/zero_shot --output-dir ./out_zs
python bootstrap_ci.py --task lp        --data-root ./examples/linear_probe --output-dir ./out_lp --fractions 100
```

## ⚖️ License and Terms of Use

Model weights and code are released under [CC BY-NC-ND 4.0](LICENSE). They may be used for non-commercial academic research only, may not be redistributed or used to build derivative checkpoints for distribution, and are not a medical device: they must not be used for clinical diagnosis or patient management. Please cite the paper when using the model or code.

## 📚 Citation

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

The released vision encoder is initialised from PanDerm; please also cite:

```bibtex
@article{yan2025multimodal,
  title={A multimodal vision foundation model for clinical dermatology},
  author={Yan, Siyuan and Yu, Zhen and Primiero, Clare and Vico-Alonso, Cristina and Wang, Zhonghua and Yang, Litao and Tschandl, Philipp and Hu, Ming and Ju, Lie and Tan, Gin and others},
  journal={Nature Medicine},
  volume={31},
  pages={2691--2702},
  year={2025}
}
```

Related: [Derm1M](https://github.com/SiyuanYan1/Derm1M) (ICCV 2025) · [MAKE](https://github.com/XiejiLi/MAGEN-O-MAKE) (MICCAI 2025)

## 📧 Contact

[Siyuan Yan](https://scholar.google.com/citations?hl=en&user=HANR6RYAAAAJ&sortby=pubdate) (siyuan.yan@monash.edu) — Monash University
