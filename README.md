<div align="center">

# PanDerm-2

**Next-generation Dermatology Foundation Model**

*Enabling Zero-shot Clinician Collaboration and Automated Concept Discovery*

[![Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/redlessone/PanDerm2)
[![License](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[📘 Documentation](xxx) | [🚀 Quick Start](#quick-start) | [📊 Benchmarks](#benchmark-results) | [💬 Discussion](xxx)

</div>

## 📑 Table of Contents

- [Highlights](#highlights)
- [Benchmark Results](#benchmark-results)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Evaluation Tasks](#evaluation-tasks)
  - [Zero-shot Classification](#zero-shot-classification)
  - [Zero-shot Cross-modal Retrieval](#zero-shot-cross-modal-retrieval)
  - [Linear Probing](#linear-probing)
  - [Multimodal Finetuning](#multimodal-finetuning)
  - [Automated Concept Discovery](#automated-concept-discovery)
- [Contributors](#contributors)
- [License](#license)
- [Contact](#contact)
- [Citation](#citation)

## Highlights

🏆 **State-of-the-art Performance**: Achieves 73.20% average accuracy across 7 zero-shot classification benchmarks

🔍 **Multimodal Fusion**: Supports clinical images, dermoscopic images, and patient metadata

🧠 **Interpretable AI**: Built-in concept discovery with Sparse Autoencoders (SAE)

🌍 **Multi-center Validation**: Evaluated on datasets from Austria, Brazil, Korea, Portugal, and more

## Benchmark Results

PanDerm-2 demonstrates state-of-the-art performance across diverse benchmarks.

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
| **PanDerm-2 (Ours)** | **0.7957** | **0.6941** | **0.8663** | **0.9304** | **0.4450** | **0.5075** | **0.8848** | **0.7320** |

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
| DermLIP-PanDerm [[5]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Derm1M_A_Million-scale_Vision-Language_Dataset_Aligned_with_Clinical_Ontology_Knowledge_ICCV_2025_paper.pdf) | 0.8184 | 0.8707 | 0.6529 | 0.3637 | 0.6764 |
| MAKE [[4]](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_35) | 0.8257 | 0.7813 | 0.6790 | 0.3986 | 0.6712 |
| DINOv3-ViT-L16 [[8]](https://ai.meta.com/dinov3/) | 0.7705 | 0.8310 | 0.6573 | 0.3018 | 0.6401 |
| DINOv3-ViT-7B [[8]](https://ai.meta.com/dinov3/) | 0.7871 | 0.8226 | **0.6985** | 0.3345 | 0.6607 |
| **PanDerm-2 (Ours)** | **0.8416** | **0.8687** | 0.6855 | **0.4007** | **0.6991** |

## Repository Structure
```
PanDerm-2/
├── src/                              # Core models and modules
├── script/                           # Experiment scripts
├── data/                             # Dataset storage
├── automated-concept-discovery/      # SAE & CBM implementation
├── linear_probe/                     # Linear probe utilities
├── multimodal_finetune/              # Multimodal training code
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```
## Quick Start

### Installation
```bash
git clone git@github.com:SiyuanYan1/PanDerm-2.git
cd PanDerm-2

conda create -n panderm python=3.9.20
conda activate panderm
pip install -r requirements.txt
```

### Download Data

Download benchmark data from [Google Drive](xxx) and unzip to the `data/` folder.

Expected directory structure:
```
data/
├── zero-shot-classification/
├── zero-shot-retrieval/
├── linear_probe/
├── multimodal_finetune/
└── automated-concept-discovery/
```

### Quick Example

See our [interactive notebook](examples/zero-shot-classification.ipynb) for zero-shot disease classification.

## Evaluation Tasks

### Task1: Zero-shot Classification

Evaluate PanDerm-2 on 7 dermatology datasets without fine-tuning.

**Benchmark datasets**: HAM, PAD, ISIC2020, PH2, SNU, SD-128, Daffodil
```bash
# Quick run
bash script/zero-shot-eval/PanDerm-v2-zs-classification.sh

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
   --model 'hf-hub:redlessone/PanDerm2'
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
   --model 'hf-hub:redlessone/PanDerm2'
```

### Task2: Zero-shot Cross-modal Retrieval

Evaluate image-text retrieval performance on Derm1M Hold-out and SkinCAP datasets.
```bash
bash script/zero-shot-eval/PanDerm-v2-zs-retrieval.sh
```

### Task3: Linear Probing

Evaluate feature quality by training linear classifiers on frozen features.

**Datasets**: HAM, ISIC2020, PAD, SD-128
```bash
bash script/linear-probe/PanDerm-v2-lp-eval.sh
```

### Task4: Multimodal Finetuning

Fine-tune PanDerm-2 with clinical images, dermoscopic images, and patient metadata.

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
- `--model_name`: Base model (e.g., `PanDerm-v2`)
- `--dataset_name`: Target dataset (`Derm7pt`, `MILK-11`, `PAD`)
- `--epochs`: Training epochs (default: 50)
- `--batch_size`: Batch size per GPU (default: 32)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--use_cli`, `--use_derm`, `--use_meta`: Enable modalities

Metadata is converted to text prompts - see [`multimodal_finetune/dataset/prompt.py`](multimodal_finetune/dataset/prompt.py).

Results are saved to `multimodal_finetune-result/`.

### Task5: Automated Concept Discovery

Discover interpretable concepts using Sparse Autoencoders (SAE) and build Concept Bottleneck Models (CBM).

**Prerequisites:**
```bash
bash script/automated-concept-discovery/env_setup.sh
```

Download SAE checkpoint from [Google Drive](xxx) to `automated-concept-discovery-result/SAE-embeddings/`.

**Quick run:**
```bash
bash script/automated-concept-discovery/dermoscopic-melanoma-classification/PanDerm-v2-SAE.sh
```

**Step-by-step pipeline:**
```bash
# Step 1: Extract visual features
cd src
python export_visual_features.py \
    --model_name hf-hub:redlessone/PanDerm2 \
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



## Contributors

- [Siyuan Yan](https://scholar.google.com/citations?user=LGcOLREAAAAJ&hl=en)
- [Xieji Li](https://scholar.google.com/citations?user=X50rN1oAAAAJ&hl=en)

## License

The model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial academic research purposes with proper attribution.

## Contact

**Siyuan Yan** - Research Fellow, Monash University  
📧 Email: siyuan.yan@monash.edu  
**Supervisor:** A/Prof. Zongyuan Ge

## Citation

If you find PanDerm-2 useful, please cite:
```bibtex
TBD
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
