# PanDerm 2
Next-generation dermatology FM enables zero-shot clinician collaboration and automated concept discovery

The repo is Under Construction.

## Benchmark Result


PanDerm-2 demonstrates state-of-the-art performance across multiple dermatology tasks and datasets.

**Task Overview:**
- **Skin Cancer**: General skin cancer classification
- **Mel Det.**: Melanoma detection (binary classification)
- **DDX**: Differential diagnosis (fine-grained classification)
- **Rare DX**: Rare disease diagnosis

**Modality:** D = Dermoscopic, C = Clinical

### Zero-Shot Classification Performance

| Model | HAM<br>(7-D) | PAD<br>(6-C) | ISIC2020<br>(2-D) | PH2<br>(2-C) | SNU<br>(134-C) | SD-128<br>(128-C) | Daffodil<br>(5-D) | **Average** |
|-------|:----:|:----:|:-----:|:---:|:------:|:-------:|:--------:|:------:|
| **Task** | Skin Cancer | Skin Cancer | Mel Det. | Mel Det. | DDX | DDX | Rare DX | - |
| **Country/Inst** | Austria | Brazil | Multi-center | Portugal | Korea | Multi-center | Multi-center | - |
| **Metric** | top-1 | top-1 | AUROC | AUROC | top-1 | top-1 | top-1 | - |
| CLIP-Large [[1]](https://proceedings.mlr.press/v139/radford21a) | 0.2754 | 0.3839 | 0.4772 | 0.3855 | 0.0857 | 0.1210 | 0.5304 | 0.3227 |
| BiomedCLIP [[2]](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | 0.6347 | 0.4512 | 0.7305 | 0.8441 | 0.0966 | 0.1153 | 0.5785 | 0.4930 |
| MONET [[3]](https://www.nature.com/articles/s41591-024-02887-x) | 0.3347 | 0.4729 | 0.6940 | 0.8370 | 0.1414 | 0.2028 | 0.7607 | 0.4919 |
| MAKE [[4]](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_35) | 0.4551 | 0.5857 | 0.8141 | 0.9095 | 0.3260 | 0.3886 | 0.7785 | 0.6082 |
| DermLIP-ViT-B-16 [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.6813 | 0.6074 | 0.8235 | 0.8285 | 0.2532 | 0.2783 | 0.7246 | 0.5995 |
| DermLIP-PanDerm [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.6281 | 0.6247 | 0.7876 | 0.7975 | 0.3332 | 0.3822 | 0.7812 | 0.6192 |
| **PanDerm-2 (Ours)** | **0.7957** | **0.6941** | **0.8663** | **0.9304** | **0.4450** | **0.5075** | **0.8848** | **0.7320** |

### Few-Shot Learning (10% training data)

Evaluation with limited labeled data to assess data efficiency and representation quality.

| Model | HAM<br>(7-class) | ISIC'20<br>(Melanoma) | PAD<br>(6-class) | SD-128<br>(128-class) | **Average** |
|-------|:----:|:--------:|:---:|:--------:|:------:|
| **Task** | Skin Cancer | Mel Det. | Skin Cancer | DDX | - |
| **Metric** | top-1 | AUROC | top-1 | top-1 | - |
| CLIP [[1]](https://proceedings.mlr.press/v139/radford21a) | 0.7798 | 0.7828 | 0.6161 | 0.3146 | 0.6233 |
| BiomedCLIP [[2]](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | 0.6959 | 0.4318 | 0.6499 | 0.2541 | 0.5079 |
| MONET [[3]](https://www.nature.com/articles/s41591-024-02887-x) | 0.8064 | 0.8036 | 0.6464 | 0.2747 | 0.6328 |
| BiomedGPT [[6]](https://arxiv.org/abs/2305.17100) | 0.7565 | 0.7838 | 0.5249 | 0.1694 | 0.5586 |
| PanDerm (NMED) [[7]](https://www.nature.com/articles/s41591-024-02887-x) | 0.7898 | 0.8417 | 0.6508 | 0.3483 | 0.6577 |
| DermLIP-ViT-B-16 [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.8157 | 0.8058 | 0.6594 | 0.3552 | 0.6590 |
| DermLIP-PanDerm [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.8184 | 0.8707 | 0.6529 | 0.3637 | 0.6764 |
| MAKE [[4]](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_35) | 0.8257 | 0.7813 | 0.6790 | 0.3986 | 0.6712 |
| DINOv3-ViT-L16 [[8]](https://ai.meta.com/dinov3/) | 0.7705 | 0.8310 | 0.6573 | 0.3018 | 0.6401 |
| DINOv3-ViT-7B [[8]](https://ai.meta.com/dinov3/) | 0.7871 | 0.8226 | **0.6985** | 0.3345 | 0.6607 |
| **PanDerm-2 (Ours)** | **0.8416** | **0.8687** | 0.6855 | **0.4007** | **0.6991** |

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

<details>
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

<summary><b>Run Zero-Shot Classification Evaluation on Customized dataset</b></summary>

### 1. Prepare Your CSV
```csv
image_path,label,diag
examples/PAT_8_15_820.png,0,disease1
examples/PAT_8_15_820.png,1,disease2
examples/PAT_8_15_820.png,2,disease3
examples/PAT_8_15_820.png,1,disease1
```

**Note:** `label` is the class index (0, 1, 2...) corresponding to the disease name in `diag`.

### 2. Configure Class Names

Edit [`src/open_clip/zero_shot_metadata.py`](src/open_clip/zero_shot_metadata.py#L120) (line 13):
```python
# Class names must match the order of labels: 0 -> disease1, 1 -> disease2, 2 -> disease3
customized_CLASSNAMES = ['disease1', 'disease2', 'disease3']
```

### 3. Run Evaluation
```bash
python src/main.py \
   --val-data="" \
   --dataset-type "csv" \
   --batch-size=1024 \
   --csv-label-key label \
   --csv-img-key image_path \
   --zeroshot_eval_custom examples/toy_dataset.csv \
   --model 'hf-hub:redlessone/PanDerm2'
```
</details>

</details>


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

<details>
<summary>Key Hyperparameters in the script</summary>

**Basic Configuration:**
- `--model_name`: Base model to finetune (e.g., `PanDerm-v2`)
- `--dataset_name`: Target dataset (`Derm7pt`, `MILK-11`, `PAD`)
- `--output_dir`: Directory to save checkpoints and results

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size per GPU (default: 32)
- `--accum_freq`: Gradient accumulation steps (default: 2, effective batch = 64)
- `--learning_rate`: Learning rate (default: 1e-5)

**Model Architecture:**
- `--hidden_dim`: Hidden dimension size (default: 1024)
- `--meta_dim`: Metadata embedding dimension (default: 768)
- `--num_head`: Number of attention heads for image fusion (default: 8)
- `--att_depth`: Depth of attention layers for image fusion (default: 2)
- `--meta_num_head`: Number of attention heads for metadata fusion (default: 8)
- `--meta_att_depth`: Depth of attention layers for metadata (default: 4)
- `--fusion`: Fusion method for images (`'cross attention'`, `'concatenate'`)
- `--meta_fusion_mode`: Fusion method for metadata (`'cross attention'`, `'concatenate'`)
- `--encoder_pool`: Pooling method for encoder features (`'mean'`)
- `--out`: Output layer type (`'mlp'`, `'linear'`)

**Modality Flags** (use flags to enable modalities):
- `--use_cli`: Use clinical images
- `--use_derm`: Use dermoscopic images
- `--use_meta`: Use metadata (age, sex, location, etc.)
- `--use_text_encoder`: Use text encoder for metadata

**Testing Only:**
- `--model_path`: Path to trained model checkpoint for inference

</details>

```bash
# Navigate to multimodal finetune directory
cd multimodal_finetune

# Finetune on different datasets (training + inference)
bash ../script/multimodal_finetune/Derm7pt\(C+D+M\).sh
bash ../script/multimodal_finetune/MILK11\(C+D\).sh
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

<details>
<summary>Key Hyperparameters</summary>

**Step2: Feature Extraction ([`export_visual_features.py`](src/export_visual_features.py)):**
- `--model_name`: Model path (e.g., `hf-hub:redlessone/PanDerm2`)
- `--batch_size`: Processing batch size (default: 2048, reduce if OOM)
- `--num_workers`: Data loading workers (default: 16)
- `--device`: `cuda` or `cpu`

**Step3: SAE Activation Extraction ([`0_extract_sae_activations.py`](automated-concept-discovery/0_extract_sae_activations.py)):**
- `--checkpoint`: Path to pre-trained SAE weights
- `--embeddings`: Input visual features (.npy file)
- `--output`: Output path for SAE activations

**Step4: Classifier Training ([`1_train_clf_binary-class.py`](automated-concept-discovery/3_train_biased-cbm_binary-class.py)):**
- `--embeddings`: Input features (SAE activations or raw embeddings)
- `--csv`: Metadata with labels and splits
- `--gpu`: GPU device ID
- `--output`: Directory for saving model and results

**Note:** Use SAE activations for interpretable CBM, or raw embeddings for baseline comparison.

</details>

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

- **Concept Intervention**: Scripts in [this folder](script/automated-concept-discovery/ISIC-intervention/) for testing concept manipulation effects
- **Global Explanation**: Visiualization jupyter notebooks are in [this folder](automated-concept-discovery/global-explanation/) for visualizing learned concepts
- **Concept Retrieval**: Visiualization jupyter notebooks are in [this folder](automated-concept-discovery/concept-retrieval/) for analyzing and retrieving concept patterns

**Available Datasets:**
- Clinical Malignant Classification
- Dermoscopic Melanoma Classification
- ISIC Intervention Experiments

**Results:** Saved in `automated-concept-discovery-result/`.

</details>
