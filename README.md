<h1 align="center">🛰️ Hybrid Spectral–Deep Learning Framework for Satellite Image Forgery Detection</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Published-Wiley-blue?style=flat" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?style=flat&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Sentinel--2-Satellite%20Imagery-green?style=flat" />
  <img src="https://img.shields.io/badge/CNN%20%2B%20Random%20Forest-Hybrid%20Model-orange?style=flat" />
</p>

> 📄 **Published in Wiley** | Remote Sensing & Geospatial Forensics Research

---

## 📌 About

Satellite images are used in disaster management, urban planning, defense, and environmental monitoring. But what if those images are **manipulated**?

This research proposes a **Hybrid Spectral–Deep Learning Framework** that detects forged/manipulated Sentinel-2 satellite images by combining:
- 🧠 **CNN-based visual feature learning** (RGB spatial artifacts)
- 🌿 **Spectral anomaly analysis** using NDVI, NDWI, NDBI indices
- 🌲 **Random Forest classifier** for hybrid feature fusion

---

## 🗺️ Study Areas

| Region | Purpose |
|--------|---------|
| **Agra, India** | Model training & internal validation |
| **Jaipur, India** | External cross-region validation (domain shift) |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?style=flat&logo=google&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## 🔬 Methodology

The framework follows a 6-step pipeline:

1. **Data Collection** — Multi-temporal Sentinel-2 imagery acquired via Google Earth Engine for Agra (2021–2025) and Jaipur (2025)
2. **Preprocessing** — RGB bands extracted and combined with NDVI, NDWI, NDBI to form a 6-channel input
3. **Patch Extraction** — Images divided into 64×64 pixel patches for localized analysis
4. **Synthetic Manipulation** — Forged samples generated using copy-paste, noise injection, radiometric distortion, and spectral inconsistency techniques
5. **Dual-Branch Learning** — CNN branch extracts visual features; Spectral branch detects hidden spectral anomalies
6. **Hybrid Fusion** — CNN + Spectral features fused and classified using Random Forest

---

## 📊 Results

### Internal Validation (Agra Dataset)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| CNN Only | 62.67% | 90.91% | 27.03% | 41.67% | 62.16% |
| Spectral Only | 62.67% | 57.89% | 89.19% | 70.21% | 72.62% |
| **Hybrid (Ours)** | **74.67%** | **78.12%** | **67.57%** | **72.46%** | **73.19%** |

### Cross-Region Validation (Jaipur Dataset)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Hybrid (Cross-Region) | 65.00% | 74.19% | 46.00% | 56.79% | 70.52% |

---

## 📁 Project Structure
├── data/          # Agra training dataset
├── data_roi2/          # Jaipur validation dataset
├── outputs/            # Results and output files
├── src/                # Source code
├── comparative_confusion_heatmap.png
├── internal_confusion_heatmaps.png
└── README.md
---

## 🗞️ Publication

> **A Hybrid Spectral–Deep Learning Framework for Manipulated Satellite Image Detection with Cross-Region Validation Using Sentinel-2 Imagery**
>
> Pari Gupta, Aman Raj, Narayan Vyas, Varsha Devi
>
> Vivekananda Global University, Jaipur, India
>
> 📰 **Published in Wiley**

---

## 👩‍💻 Author

**Pari Gupta** — AI/ML Researcher | BTech Student  
[GitHub](https://github.com/Parigupta12)
