# ⚽ Eredivisie Player Valuation & Performance Alignment

An end-to-end machine learning classification project evaluating the alignment between market valuation and on-pitch performance metrics for Dutch Eredivisie players (2023/2024 season) to detect market inefficiencies (*Weak-form Efficient Market Hypothesis*).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Business Problem & Motivation

In modern football transfer markets, player market values are frequently distorted by media hype, club prestige, and short-term performance spikes. This project shifts the paradigm from **price prediction (regression)** to **value alignment diagnosis (investment classification)**:

> *"Does a player's market valuation accurately reflect their objective tactical and statistical contribution on the pitch?"*

The model categorizes players into three actionable investment classes:
* **Undervalued:** Top-tier performance (Q4) combined with low-to-mid market value (Q1–Q3) — *High-priority recruitment targets / hidden gems*.
* **Fairly Valued:** Market valuation aligns proportionally with on-pitch performance.
* **Overvalued:** High market value (Q4) coupled with low-to-mid performance (Q1–Q3) — *High-risk assets / potential overpayment*.

---

## ⚙️ Methodology & Architecture

* **Dataset:** 347 Dutch Eredivisie players (season 2023/2024) with a minimum playing threshold of 450 minutes, gathered from WhoScored and Transfermarkt.
* **Composite Metric:** Position-specific *Weighted Linear Combination* (WLC) tailored to distinct positional Key Performance Indicators (Forward, Midfielder, Defender, Goalkeeper).
* **Handling Class Imbalance:** The target variable exhibits natural market imbalance dominated by *Fairly Valued* players (~70%). Addressed via **Cost-Sensitive Learning** (*Inverse Class Weighting*: Fairly Valued = 0.457, Overvalued = 2.638, Undervalued = 2.308) to preserve raw statistical variance without synthetic noise.
* **Model Benchmark:** Random Forest vs. Gradient Boosting evaluated via 5-Fold Stratified Cross-Validation.

### 🔬 Metodologi v2.0 (Enhanced Football Analytics & Interview-Ready)
Untuk mengatasi keterbatasan teknis baseline (v1), proyek ini mengimplementasikan metodologi **v2.0**:
1. **Konversi Metrik ke Per-90 Menit (P90):** Mengonversi metrik volume dan rata-rata pertandingan menjadi standard $\text{Stat}_{P90} = \frac{\text{Total Statistik}}{\text{Menit Bermain}} \times 90$, menetralkan bias pemain pengganti (supersub).
2. **Min-Max Scaling per Posisi Sebelum WLC:** Menormalisasi metrik ke skala $[0, 1]$ di dalam kelompok posisi masing-masing sebelum dikalikan bobot $w_j$, mencegah metrik kiper (skala puluhan) mendominasi penyerang (skala satuan).
3. **Grouped Quantiles per Posisi:** Menghitung kuartil performa dan nilai pasar per posisi (`df.groupby('Position')['Performance_Score'].transform(...)`) sehingga 25% pemain terbaik diukur terhadap rekan seposisinya (Bek diadu sesama bek, Striker sesama striker), menghapuskan bias posisi global.

---

## 📊 Model Performance (Test Set, N = 70)

Random Forest dipilih sebagai model unggulan (*champion model*) karena stabilitas generalisasi yang lebih konsisten dan profil risiko investasi yang lebih aman:

| Metric | Random Forest (Selected) | Gradient Boosting |
| :--- | :---: | :---: |
| **Accuracy** | **81.43%** | 80.00% |
| **Macro F1-Score** | **0.7306** | 0.7262 |
| **Macro ROC-AUC** | **0.9261** | 0.8991 |
| **Recall (Undervalued)** | **80.00%** | 70.00% |
| **Precision (Undervalued)** | **61.54%** | 46.67% |

> **Business Impact:** Random Forest menghasilkan presisi 61.5% untuk target *Undervalued* (vs. 46.7% pada Gradient Boosting), secara signifikan memangkas risiko alokasi dana transfer yang salah pada target "palsu" (*false bargains*).

---

## 🔍 Key Empirical Insights

1. **Development League Hypothesis:** Metrik distribusi bola mendominasi klasifikasi valuasi dibanding statistik hasil akhir (*end-product*). **Average Pass Length** (importance: 0.0844) dan **Pass Success Accuracy** (0.0843) menempati peringkat teratas fitur prediktif, mengonfirmasi bahwa scout Eropa memprioritaskan kemampuan *build-up* modern di Eredivisie.
2. **Statistical Validation:** Uji non-parametrik **Mann-Whitney U Test** membuktikan perbedaan *Average Pass Length* antara pemain *Undervalued* (rata-rata: 30.98 m) dan *Fairly Valued* (rata-rata: 26.87 m) signifikan secara statistik ($p = 0.0069$).
3. **Midfield Playmaking Sensitivity:** Uji sensitivitas membuktikan bahwa penyesuaian bobot *Assist* gelandang meningkatkan F1-Macro model hingga **0.8139 (+11.4%)**, menandakan *playmaking output* sebagai pendorong valuasi pasar paling sensitif.

---

## 🚀 Quickstart & Installation

Ikuti panduan berikut untuk mereplikasi analisis secara lokal:

### 1. Clone Repository
```bash
git clone https://github.com/jhodifachriansyah15/eredivisie-player-valuation-ml.git
cd eredivisie-player-valuation-ml
```

### 2. Buat & Aktifkan Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan Analisis

**Jalankan Metodologi v2.0 (Default, P90 + Position Scaling + Grouped Quantiles):**
```bash
python src/main_analysis.py
```

**Audit Perbandingan v1 vs v2 (Distribusi & Bias Posisi):**
```bash
python src/main_analysis.py --compare
```

**Jalankan Metodologi Baseline v1 (Skripsi / Tugas Akhir):**
```bash
python src/main_analysis.py --version v1
```

### 5. Eksplorasi Interaktif melalui Jupyter Notebook
```bash
jupyter notebook notebooks/eredivisie_analysis.ipynb
```

---

## 📁 Project Structure

```text
eredivisie-player-valuation-ml/
├── assets/                                  # Visualisasi grafik & artefak analisis (.png, .csv)
│   ├── comparison_class_weight_sample_weight.png
│   ├── confusion_matrix_gb.png
│   ├── confusion_matrix_rf.png
│   ├── correlation_heatmap_numerical.png
│   ├── distribusi_kelas_posisi.png
│   ├── distribusi_v1_vs_v2_posisi.png
│   ├── feature_importance_gb.png
│   ├── feature_importance_rf.png
│   ├── learning_curve_gb.png
│   ├── learning_curve_rf.png
│   ├── roc_curve_gb.png
│   └── roc_curve_rf.png
├── data/                                    # Dataset mentah & terproses
│   └── Data_Convert.csv
├── models/                                  # Model terlatih (.pkl) & metrik (.json)
│   ├── best_gb_model.pkl
│   ├── best_rf_model.pkl
│   └── summary_metrics_v2.json
├── notebooks/                               # Eksplorasi interaktif & visualisasi
│   └── eredivisie_analysis.ipynb
├── src/                                     # Skrip kode produksi modular
│   └── main_analysis.py
├── .gitignore                               # Aturan ignore file sampah Python/Jupyter
├── LICENSE                                  # Lisensi open-source MIT
├── README.md                                # Dokumentasi utama portofolio
└── requirements.txt                         # Daftar dependensi library Python
```

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** Scikit-Learn (Random Forest, Gradient Boosting, GridSearchCV, StratifiedKFold)
* **Data Processing & Analytics:** Pandas, NumPy, SciPy (Mann-Whitney U Test)
* **Data Visualization:** Matplotlib, Seaborn
* **Model Persistence:** Joblib

---

## 👤 Author & Contact

**Jhodi Fachriansyah**
* **GitHub:** [@jhodifachriansyah15](https://github.com/jhodifachriansyah15)
* **LinkedIn:** [linkedin.com/in/jhodifachriansyah](https://www.linkedin.com/in/jhodifachriansyah)
* **Email:** [jhodifachriansyah15@gmail.com](mailto:jhodifachriansyah15@gmail.com)

---

## 📜 License

Proyek ini dilisensikan di bawah lisensi open-source [MIT License](LICENSE).
Silakan digunakan untuk referensi riset, akademis, dan portofolio data science.