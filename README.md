# ASEEL: A Multimodal AI System for Counterfeit Product Detection in E-Commerce

A deep learning framework for detecting counterfeit luxury handbags using a multimodal fusion architecture that integrates **product images**, **structured metadata**, **customer reviews**, and **OCR-extracted text**.

This repository contains two parallel experimental pipelines built on the same core architecture:

- **🛍️ Aseel Benchmark Experiment** — trained on a self-curated English dataset (DHgate + Farfetch)
- **🇸🇦 Mahally + Salla Experiment** — trained on a localized Arabic e-commerce dataset (Saudi marketplaces)

---

## 📊 Results

| Dataset | Accuracy | F1-Score | AUC-ROC |
|---------|----------|----------|---------|
| Aseel Benchmark (held-out test, 215 products) | **93.95%** | **94.17%** | **98.29%** |
| Mahally + Salla (5-fold CV, 2,477 products) | **98.87%** ± 0.58% | **98.78%** ± 0.63% | **99.97%** ± 0.02% |

---

## 🏗️ Architecture

ASEEL uses an **early-to-intermediate fusion strategy** that concatenates feature representations from four specialized branches into a unified 2,848-dimensional vector before classification.

### Per-Branch Specifications

| Branch | Encoder | Output Dim |
|--------|---------|------------|
| **Image** | EfficientNet-B0 + Attention Pooling (up to 3 imgs) | 1,280 |
| **Tabular** | MLP (Linear → BN → ReLU → Dropout) | 32 |
| **Text** | BERT-base-uncased (Aseel) / AraBERT (Mahally) | 768 |
| **OCR** | Shared BERT + Linear projection (768→768) | 768 |
| **Fusion** | Concatenation → MLP classifier head | 2,848 → 1 |

---

## 📁 Repository Structure

```
ASEEL/
├── README.md
├── requirements.txt
│
├── aseel_benchmark/
│   ├── ASEEL_Own_Data_FINAL_v2.ipynb
│   ├── data/
│   └── results/
│
└── mahally_localized/
    ├── Mahally_Multimodal_FIXED.ipynb
    ├── data/
    └── results/
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (NVIDIA Tesla T4 or better, ≥15 GB VRAM)
- Google Colab compatible

### Installation

```bash
git clone https://github.com/Manarali101/ASEEL.git
cd ASEEL
pip install -r requirements.txt
```

### Required Packages

```
torch>=2.0
torchvision
transformers>=4.30
easyocr
scikit-learn
pandas
numpy
Pillow
matplotlib
seaborn
tqdm
openpyxl
```

---

## 🛠️ Reproducing the Experiments

### Experiment 1 — Aseel Benchmark (English)

**Pipeline:**
1. Stratified 80/20 hold-out split (CV: 860 products | Test: 215 products)
2. 5-Fold Stratified Cross-Validation on the 80% CV partition
3. Retrain final model from scratch on full CV partition (774 train + 86 early-stop val)
4. Single-pass evaluation on the held-out test set

**Configuration:**
- Encoder: `bert-base-uncased`
- Max images per product: 3
- Tabular features: 6 (price + 5 categorical)

### Experiment 2 — Mahally + Salla (Arabic)

**Pipeline:**
1. 5-Fold Stratified Cross-Validation on 2,477 products
2. Per-fold encoder/scaler fitting (data leakage prevention)
3. Final retraining on full dataset

**Configuration:**
- Encoder: `aubmindlab/bert-base-arabertv02`
- Max images per product: 1 (thumbnails)
- Tabular features: 2 (price_scaled + brand_encoded)
- OCR: bilingual (Arabic + English)

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Batch Size | 16 |
| Max Epochs | 20 |
| Learning Rate | 2e-5 |
| Weight Decay | 1e-4 |
| Optimizer | AdamW |
| LR Scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Early Stopping | Patience 7, min_delta 0.001 |
| Loss | BCEWithLogitsLoss (with pos_weight) |
| Gradient Clipping | max_norm = 1.0 |
| Random Seed | 42 |

---

## 🔒 Data Leakage Prevention

Both pipelines enforce strict data integrity:

1. **Hold-out test set** sealed before any preprocessing (Aseel only)
2. **LabelEncoder / StandardScaler** fitted only on training fold (per CV iteration)
3. **`pos_weight`** computed from training fold class distribution only

---

## 🔑 Key Implementation Details

### Image Branch — Multi-Image Attention Pooling
For each product, up to 3 images pass through a shared EfficientNet-B0 backbone. An attention sub-network computes per-image weights with padded slots masked via `-inf` before softmax.

### OCR Branch — Shared BERT Encoder
The text and OCR branches share the same BERT encoder to reduce trainable parameters. An additional Linear projection layer learns a modality-specific transformation for the OCR stream.

### Manual Annotation (Mahally Only)
Authenticity labels were assigned through **manual annotation by domain knowledge** of the Saudi luxury market. Each product was labeled by inspecting the seller's store profile:

- **Authentic (1):** Products listed by verified official stores, authorized distributors, or recognized retailers carrying authentic brand inventory.
- **Counterfeit (0):** Products listed by stores known to sell replicas, copies, or unauthorized imitations.

Manual annotation produces gold-standard ground-truth labels and is the most reliable annotation method when domain expertise is available (Sambasivan et al., 2021).

---

## 📚 Citation

```bibtex
@misc{aseel2026,
  title  = {ASEEL: A Multimodal AI System for Counterfeit Product Detection in E-Commerce},
  year   = {2026},
  school = {Umm Al-Qura University, College of Computing},
  note   = {Graduation Project, Supervised by Dr. Ghader Kurdi}
}
```

---

## 🙏 Acknowledgments

- **Supervisor:** Dr. Ghader Kurdi, College of Computing, Umm Al-Qura University
- **Data Sources:** DHgate, Farfetch, Mahally.com, Salla
- **Pre-trained Models:** Hugging Face (BERT, AraBERT), PyTorch (EfficientNet-B0)

---

## 📄 License

This project is released for academic research purposes.

