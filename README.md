# PanDerm 2
Next-generation dermatology FM enables zero-shot clinician collaboration and automated concept discovery

The repo is Under Construction.

##  Repository Layout
```text
PanDerm-2/
├── data/                             # data folder (datasets with images and csvs)
├── src/                              # core models and modules(For zero-shot tasks)
├── script/                           # scripts for experiments
├── README.md                         # Project documentation
├── automated-concept-discovery/      # Automated concept discovery implementation
├── linear_probe/                     # Linear probing experiments and evaluation
├── multimodal_finetune/              # Multimodal fine-tuning scripts and models
└── requirements.txt                  # Python dependencies
```

## Getting Started

```bash
git clone git@github.com:SiyuanYan1/PanDerm-2.git
cd PanDerm-2

conda create -n PanDerm-v2 python=3.9.20
conda activate PanDerm-v2
pip install -r requirements.txt
```
Here is a [simple example](examples/zero-shot-classification.ipynb) to call PanDerm-2 to conduct **zero-shot disease classification**

## Evaluation

### Setup

1) Download benchmark data from [Google Drive](xxx)
2) Unzip to data folder

The directory structure should look like:
```bash
data/
├── automated-concept-discovery
│   ├── clinical-malignant
│   ├── dermoscopic-melanoma
│   ├── ISIC_hair_bias
│   ├── ISIC_ink_bias
│   └── ISIC_ruler_bias
├── linear_probe
│   ├── HAM-official-7-lp
│   ├── isic2020-2-lp
│   ├── pad-lp
│   └── sd-128-lp
├── multimodal_finetune
│   ├── derm7pt
│   ├── MILK-11
│   └── PAD
├── zero-shot-classification
│   ├── daffodil-5-zero-shot
│   ├── HAM-official-7-zero-shot
│   ├── isic2020-2-zero-shot
│   ├── pad-zero-shot
│   ├── ph2-2-zero-shot
│   ├── sd-128-zero-shot
│   └── snu-134-zero-shot
└── zero-shot-retrieval
    ├── Derm1M-hold_out
    └── skincap
```

## Running Evaluations

<summary><b>Zero-shot Classification</b></summary>

Evaluate PanDerm-2 on multiple dermatology datasets using zero-shot classification:

```bash
# Run the zero-shot classification benchmark script
bash script/zero-shot-eval/PanDerm-v2-zs-classification.sh

# Or run:
python src/main.py \
   --val-data=""  \
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


<details>
<summary><b>Zero-shot Cross-modal Retrieval</b></summary>

Evaluate cross-modal retrieval performance between images and text descriptions.

**Datasets evaluated:** Derm1M-Hold Out Dataset, SkinCAP

This script evaluates PanDerm-2 across two datasets for image-text retrieval tasks:
```bash
# Run the cross-modal retrieval benchmark script
bash script/zero-shot-eval/PanDerm-v2-zs-retrieval.sh
```

</details>

<details>
<summary><b>Linear Probing</b></summary>

Evaluate feature quality through linear probing on downstream classification tasks:
```bash
# Run the script
bash script/linear-probe/PanDerm-v2-lp-eval.sh
```

</details>

<details>

<summary><b>Multimodal Finetune</b></summary>

Finetune PanDerm-2 on three multi-modal dermatology datasets:

**Dataset Modalities Overview:**
- **Derm7pt**: Clinical + Dermoscopic + Metadata
- **MILK**: Clinical + Dermoscopic  
- **PAD-UFES-20**: Clinical + Metadata

**Usage:**
```bash
# Navigate to multimodal finetune directory
cd multimodal_finetune

# Finetune on different datasets (training + inference)
bash ../script/multimodal_finetune/Derm7pt\(C+D+M\).sh
bash ../script/multimodal_finetune/MILK11(C+D\).sh
bash ../script/multimodal_finetune/PAD\(C+M\).sh
```

**Note:** Each script performs both training and inference. Results and checkpoints will be saved in `multimodal_finetune-result/`.

</details>

<details>
<summary><b>Automated Concept Discovery</b></summary>

Discover interpretable concepts from dermatology images using Sparse Autoencoders (SAE) and build Concept Bottleneck Models (CBM).

**Prerequisites:**
- Setup environment: `bash script/automated-concept-discovery/env_setup.sh`
- Download pre-trained SAE checkpoint from [Google Drive](xxx) and place it in `automated-concept-discovery-result/SAE-embeddings/`

**Usage (Clinical Malignant Classification as Example):**

*You can also use: `bash script/automated-concept-discovery/dermoscopic-melanoma-classification/PanDerm-v2-SAE.sh`*
```bash
# Quick Start: Use pre-configured script
bash script/automated-concept-discovery/dermoscopic-melanoma-classification/PanDerm-v2-SAE.sh

# Or run step-by-step:

# Setup environment (run once)
bash script/automated-concept-discovery/env_setup.sh

# Step 1: Train Sparse Autoencoder (optional, skip if using pre-trained)
bash script/automated-concept-discovery/SAE-training/PanDerm-v2.sh

# Step 2: Extract visual features
cd src
python export_visual_features.py \
    --model_name hf-hub:redlessone/PanDerm2 \
    --csv_path ../data/automated-concept-discovery/clinical-malignant/meta.csv \
    --data_root ../data/automated-concept-discovery/clinical-malignant/final_images/ \
    --img_col 'ImageID' \
    --batch_size 2048 \
    --output_dir ../automated-concept-discovery-result/clinical-malignant/
cd ..

# Step 3: Extract SAE concepts
python automated-concept-discovery/0_extract_sae_activations.py \
  --checkpoint automated-concept-discovery-result/SAE-embeddings/autoencoder.pth \
  --embeddings automated-concept-discovery-result/clinical-malignant/all_embeddings.npy \
  --output automated-concept-discovery-result/clinical-malignant/learned_activation.npy

# Step 4: Build concept-based classifier (CBM)
python automated-concept-discovery/1_train_clf_binary-class.py \
  --csv data/automated-concept-discovery/clinical-malignant/meta.csv \
  --embeddings automated-concept-discovery-result/clinical-malignant/learned_activation.npy \
  --image_col ImageID \
  --output automated-concept-discovery-result/clinical-malignant/
```

**Concept Intervention & Visualization:**

Validate CBM effectiveness and interpretability through various experiments:

- **Concept Intervention**: Scripts in `script/automated-concept-discovery/ISIC-intervention/` for testing concept manipulation effects
- **Global Explanation**: Tools in `automated-concept-discovery/global-explanation/` for visualizing learned concepts
- **Concept Retrieval**: Scripts in `automated-concept-discovery/concept-retrieval/` for analyzing and retrieving concept patterns

**Available Datasets:**
- Clinical Malignant Classification
- Dermoscopic Melanoma Classification
- ISIC Intervention Experiments

**Results:** Saved in `automated-concept-discovery-result/`.

</details>
