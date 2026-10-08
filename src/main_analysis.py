"""
==============================================================================
SKRIP ANALISIS KLASIFIKASI MARKET VALUE & PERFORMA PEMAIN EREDIVISIE 2023/2024
Random Forest vs Gradient Boosting (End-to-End Pipeline)
Dukungan Metodologi:
  - v1: Baseline Skripsi / Tugas Akhir (Global Scaling & League-Wide Quantiles)
  - v2: Metodologi Lanjutan / Rekomendasi DS Lead (P90 Metrics, 
        Min-Max Scaling per Posisi, Grouped Quantiles per Posisi)
==============================================================================
"""

import os
import sys

# Konfigurasi encoding stdout untuk kompatibilitas Windows terminal
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import copy
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, GridSearchCV, learning_curve,
    StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import OneHotEncoder, label_binarize, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# Konfigurasi path direktori proyek
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

POSSIBLE_DATA_PATHS = [
    os.path.join(ROOT_DIR, 'data', 'Data_Convert.csv'),
    os.path.join(BASE_DIR, '..', 'data', 'Data_Convert.csv'),
    os.path.join(ROOT_DIR, 'Data_Convert.csv'),
    'data/Data_Convert.csv',
    'Data_Convert.csv'
]
DATA_PATH = next((p for p in POSSIBLE_DATA_PATHS if os.path.exists(p)), POSSIBLE_DATA_PATHS[0])

MODELS_DIR = os.path.join(ROOT_DIR, 'models')
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Konfigurasi visualisasi
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid', context='notebook', palette='colorblind')
plt.rcParams['figure.dpi'] = 100


# ==============================================================================
# BOBOT POSISI (WLC)
# ==============================================================================
WEIGHTS_BASE = {
    'Forward': {
        'Goals': 0.35, 'Assist': 0.20, 'Shots_per_game': 0.15,
        'Key_passes': 0.15, 'Dribbles_per_game': 0.10, 'Man_of_the_Match': 0.05
    },
    'Midfielder': {
        'Assist': 0.25, 'Key_passes': 0.20, 'Goals': 0.15,
        'Pass_success_accuracy': 0.15, 'Tackles': 0.10,
        'Dribbles_per_game': 0.10, 'Man_of_the_Match': 0.05
    },
    'Defender': {
        'Tackles': 0.25, 'Clearances': 0.20, 'Interceptions': 0.20,
        'Aerial_duels_won': 0.15, 'Pass_success_accuracy': 0.10,
        'Blocks': 0.05, 'Man_of_the_Match': 0.05
    },
    'Goalkeeper': {
        'Pass_success_accuracy': 0.30, 'Long_balls': 0.25,
        'Average_pass_length': 0.20, 'Aerial_duels_won': 0.15,
        'Man_of_the_Match': 0.10
    }
}


# ==============================================================================
# FUNGSI FEATURE ENGINEERING: V1 (BASELINE) VS V2 (IMPROVED)
# ==============================================================================
def prepare_data_v1(df_raw):
    """
    Metodologi v1 (Baseline Skripsi / Original):
    - Metrik dihitung langsung dari data mentah.
    - Performance_Score dihitung dengan rumus WLC langsung pada nilai raw.
    - Normalisasi min-max dilakukan secara GLOBAL lintas seluruh liga.
    - Kuartil (pd.qcut) dihitung secara GLOBAL.
    """
    df = df_raw.copy()

    def calculate_performance_score(row):
        pos = row['Position']
        score = 0.0
        if pos in WEIGHTS_BASE:
            for feature, weight in WEIGHTS_BASE[pos].items():
                score += row.get(feature, 0.0) * weight
        return score

    df['Performance_Score'] = df.apply(calculate_performance_score, axis=1)
    df['Perf_Score_Normalized'] = (df['Performance_Score'] - df['Performance_Score'].min()) / \
                                   (df['Performance_Score'].max() - df['Performance_Score'].min())
    df['MV_Normalized'] = (df['Market_Value'] - df['Market_Value'].min()) / \
                          (df['Market_Value'].max() - df['Market_Value'].min())

    df['Perf_Quantile'] = pd.qcut(df['Perf_Score_Normalized'], 4, 
                                  labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    df['MV_Quantile'] = pd.qcut(df['MV_Normalized'], 4, 
                                 labels=['Q1','Q2','Q3','Q4'], duplicates='drop')

    def classify_alignment(row):
        perf_q, mv_q = row['Perf_Quantile'], row['MV_Quantile']
        if perf_q == 'Q4' and mv_q in ['Q1','Q2','Q3']:
            return 'Undervalued'
        elif mv_q == 'Q4' and perf_q in ['Q1','Q2','Q3']:
            return 'Overvalued'
        else:
            return 'Fairly Valued'

    df['Alignment_Class'] = df.apply(classify_alignment, axis=1)
    return df


def prepare_data_v2(df_raw):
    """
    Metodologi v2 (Rekomendasi Data Science Lead & Football Analytics):
    1. Konversi Statistik ke Per-90 Menit (P90):
       Stat_P90 = (Total_Stat / Menit_Bermain) * 90
    2. Min-Max Scaling per Posisi Sebelum Rumus WLC:
       Mencegah metrik kiper (skala puluhan) mendominasi penyerang (skala satuan).
    3. Grouped Quantiles per Posisi:
       Menilai pemain terhadap rekan seposisinya (Bek vs Bek, Striker vs Striker).
    """
    df = df_raw.copy()

    # 1. Konversi metrik ke P90
    total_matches = df['Starter'] + df['Subs']
    p90_factor = 90.0 / df['Minutes']

    count_cols = ['Goals', 'Assist', 'Yellow_cards', 'Red_cards', 'Man_of_the_Match', 'Own_goals']
    per_game_cols = [
        'Shots_per_game', 'Aerial_duels_won', 'Tackles', 'Interceptions',
        'Fouls_committed', 'Offsides_won', 'Clearances', 'Times_dribbled_past',
        'Blocks', 'Key_passes', 'Dribbles_per_game', 'Times_fouled', 'Offsides',
        'Dispossessed', 'Unsuccessful_touches', 'Crosses_attempted', 'Long_balls', 'Through_balls'
    ]

    for col in count_cols:
        df[col + '_p90'] = df[col] * p90_factor

    for col in per_game_cols:
        # Total kejadian = rata-rata per game * total penampilan
        # P90 = Total / Minutes * 90
        df[col + '_p90'] = df[col] * total_matches * p90_factor

    # Pemetaan bobot v2 menggunakan metrik P90 untuk variabel volume
    weights_v2 = {
        'Forward': {
            'Goals_p90': 0.35, 'Assist_p90': 0.20, 'Shots_per_game_p90': 0.15,
            'Key_passes_p90': 0.15, 'Dribbles_per_game_p90': 0.10, 'Man_of_the_Match_p90': 0.05
        },
        'Midfielder': {
            'Assist_p90': 0.25, 'Key_passes_p90': 0.20, 'Goals_p90': 0.15,
            'Pass_success_accuracy': 0.15, 'Tackles_p90': 0.10,
            'Dribbles_per_game_p90': 0.10, 'Man_of_the_Match_p90': 0.05
        },
        'Defender': {
            'Tackles_p90': 0.25, 'Clearances_p90': 0.20, 'Interceptions_p90': 0.20,
            'Aerial_duels_won_p90': 0.15, 'Pass_success_accuracy': 0.10,
            'Blocks_p90': 0.05, 'Man_of_the_Match_p90': 0.05
        },
        'Goalkeeper': {
            'Pass_success_accuracy': 0.30, 'Long_balls_p90': 0.25,
            'Average_pass_length': 0.20, 'Aerial_duels_won_p90': 0.15,
            'Man_of_the_Match_p90': 0.10
        }
    }

    # 2. Min-Max Scaling dalam Kelompok Posisi Sebelum WLC
    df['Performance_Score'] = 0.0
    for pos, w_dict in weights_v2.items():
        mask = df['Position'] == pos
        pos_score = np.zeros(mask.sum())
        for feat, w in w_dict.items():
            vals = df.loc[mask, feat]
            f_min, f_max = vals.min(), vals.max()
            scaled = (vals - f_min) / (f_max - f_min) if f_max > f_min else 0.0
            pos_score += scaled * w
        df.loc[mask, 'Performance_Score'] = pos_score

    df['Perf_Score_Normalized'] = df['Performance_Score']
    df['MV_Normalized'] = (df['Market_Value'] - df['Market_Value'].min()) / \
                          (df['Market_Value'].max() - df['Market_Value'].min())

    # 3. Kuartil Dihitung per Posisi (Grouped Quantiles)
    df['Perf_Quantile'] = df.groupby('Position')['Performance_Score'].transform(
        lambda s: pd.qcut(s.rank(method='first'), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    )
    df['MV_Quantile'] = df.groupby('Position')['Market_Value'].transform(
        lambda s: pd.qcut(s.rank(method='first'), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    )

    def classify_alignment(row):
        perf_q, mv_q = row['Perf_Quantile'], row['MV_Quantile']
        if perf_q == 'Q4' and mv_q in ['Q1', 'Q2', 'Q3']:
            return 'Undervalued'
        elif mv_q == 'Q4' and perf_q in ['Q1', 'Q2', 'Q3']:
            return 'Overvalued'
        else:
            return 'Fairly Valued'

    df['Alignment_Class'] = df.apply(classify_alignment, axis=1)
    return df


# ==============================================================================
# AUDIT PERBANDINGAN METODOLOGI V1 VS V2
# ==============================================================================
def compare_v1_vs_v2(df_raw):
    print("\n" + "="*80)
    print("AUDIT EVALUASI METODOLOGI: V1 (BASELINE) VS V2 (IMPROVED)")
    print("="*80)

    df_v1 = prepare_data_v1(df_raw)
    df_v2 = prepare_data_v2(df_raw)

    print("\n1. Distribusi Kelas Global:")
    cmp_global = pd.DataFrame({
        'v1_Counts': df_v1['Alignment_Class'].value_counts(),
        'v1_Pct (%)': (df_v1['Alignment_Class'].value_counts(normalize=True) * 100).round(2),
        'v2_Counts': df_v2['Alignment_Class'].value_counts(),
        'v2_Pct (%)': (df_v2['Alignment_Class'].value_counts(normalize=True) * 100).round(2)
    })
    print(cmp_global)

    print("\n2. Distribusi Kelas Berdasarkan Posisi (v1 vs v2):")
    v1_ct = pd.crosstab(df_v1['Position'], df_v1['Alignment_Class'], normalize='index') * 100
    v2_ct = pd.crosstab(df_v2['Position'], df_v2['Alignment_Class'], normalize='index') * 100

    print("\n[v1] Persentase per Posisi (Perhatikan: Kiper 76.9% Undervalued, Bek 0%):")
    print(v1_ct.round(2))

    print("\n[v2] Persentase per Posisi (Seimbang antar posisi berkat Grouped Quantiles & Min-Max):")
    print(v2_ct.round(2))

    # Visualisasi perbandingan distribusi
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    v1_ct.plot(kind='bar', ax=axes[0], colormap='viridis', edgecolor='black')
    axes[0].set_title('Distribusi Kelas v1 (Baseline Bias Posisi)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Persentase (%)')
    axes[0].legend(title='Kelas')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0)

    v2_ct.plot(kind='bar', ax=axes[1], colormap='viridis', edgecolor='black')
    axes[1].set_title('Distribusi Kelas v2 (Normalisasi & Kuartil per Posisi)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Persentase (%)')
    axes[1].legend(title='Kelas')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    comparison_fig_path = os.path.join(ASSETS_DIR, 'distribusi_v1_vs_v2_posisi.png')
    plt.savefig(comparison_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualisasi audit tersimpan: {comparison_fig_path}")


# ==============================================================================
# PIPELINE UTAMA (TAHAP 1 - 10)
# ==============================================================================
def run_pipeline(version='v2', quick=False):
    print("="*80)
    print(f"MENJALANKAN PIPELINE ANALISIS ML (METODOLOGI: {version.upper()})")
    print("="*80)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)
    print(f"\nDataset berhasil dimuat dari '{DATA_PATH}': {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")

    # ==============================================================================
    # TAHAP 1: FEATURE ENGINEERING
    # ==============================================================================
    print("\n" + "="*80)
    print(f"TAHAP 1: FEATURE ENGINEERING ({version.upper()})")
    print("="*80)

    if version == 'v1':
        df = prepare_data_v1(df_raw)
        print("Menggunakan metodologi v1: Raw Metrics, Global WLC Scaling, Global Quantiles.")
    else:
        df = prepare_data_v2(df_raw)
        print("Menggunakan metodologi v2: P90 Metrics, Min-Max Scaling per Posisi, Grouped Quantiles.")

    print("\nDistribusi Alignment_Class:")
    print(df['Alignment_Class'].value_counts())
    print("\nPersentase distribusi:")
    print((df['Alignment_Class'].value_counts(normalize=True) * 100).round(2))

    # ==============================================================================
    # TAHAP 2: EKSPLORASI DATA & STATISTIK DESKRIPTIF
    # ==============================================================================
    print("\n" + "="*80)
    print("TAHAP 2: EKSPLORASI DATA - STATISTIK DESKRIPTIF")
    print("="*80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nStatistik deskriptif fitur numerik (sample 10 fitur):")
    print(df[numeric_cols[:10]].describe().T[['mean', 'std', 'min', 'max']])

    print("\nDistribusi Posisi Pemain:")
    print(df['Position'].value_counts())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    class_counts = df['Alignment_Class'].value_counts()
    colors_bar = ['#4ECDC4', '#FF6B6B', '#FFE66D']
    class_counts.plot(kind='bar', ax=ax1, color=colors_bar, edgecolor='black')
    ax1.set_title(f'Distribusi Kelas Alignment_Class ({version.upper()})', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Kelas')
    ax1.set_ylabel('Jumlah Pemain')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(class_counts):
        ax1.text(i, v + 3, str(v), ha='center', fontweight='bold')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

    ax2 = axes[1]
    position_counts = df['Position'].value_counts()
    colors_pie = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']
    ax2.pie(position_counts, labels=position_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
    ax2.set_title('Distribusi Posisi Pemain', fontsize=13, fontweight='bold')

    plt.tight_layout()
    dist_fig_path = os.path.join(ASSETS_DIR, 'distribusi_kelas_posisi.png')
    plt.savefig(dist_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gambar disimpan: {dist_fig_path}")

    # ==============================================================================
    # TAHAP 3: ANALISIS KORELASI FITUR NUMERIK
    # ==============================================================================
    print("\n" + "="*80)
    print("TAHAP 3: ANALISIS KORELASI FITUR NUMERIK")
    print("="*80)

    target = 'Alignment_Class'
    features_to_drop = [
        'Player', 'Market_Value',
        'Performance_Score', 'Perf_Score_Normalized', 'MV_Normalized',
        'Perf_Quantile', 'MV_Quantile', 'Alignment_Class'
    ]
    features = [c for c in df.columns if c not in features_to_drop]
    categorical_features = ['Position']
    numerical_features = [c for c in features if c not in categorical_features]

    X_for_corr = df[numerical_features]
    corr_matrix = X_for_corr.corr(method='pearson')

    threshold = 0.8
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr.append({
                    'Fitur_1': corr_matrix.columns[i],
                    'Fitur_2': corr_matrix.columns[j],
                    'Korelasi': corr_matrix.iloc[i, j]
                })

    if high_corr:
        print(f"\nPasangan fitur dengan |r| > {threshold}:")
        print(pd.DataFrame(high_corr).sort_values('Korelasi', ascending=False).head(10))
    else:
        print(f"\nTidak ada pasangan fitur dengan |r| > {threshold}.")

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Matriks Korelasi Pearson (Fitur Numerik)',
                 fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    corr_fig_path = os.path.join(ASSETS_DIR, 'correlation_heatmap_numerical.png')
    plt.savefig(corr_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Heatmap disimpan: {corr_fig_path}")

    # ==============================================================================
    # TAHAP 4: PREPROCESSING & SPLIT DATA
    # ==============================================================================
    print("\n" + "="*80)
    print("TAHAP 4: PREPROCESSING & SPLIT DATA")
    print("="*80)

    X = df[features]
    y = df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Ukuran data train: {X_train.shape[0]} sampel")
    print(f"Ukuran data test:  {X_test.shape[0]} sampel")

    # ==============================================================================
    # TAHAP 5: CLASS_WEIGHT & SAMPLE_WEIGHT
    # ==============================================================================
    print("\n" + "="*80)
    print("TAHAP 5: ANALISIS CLASS_WEIGHT & SAMPLE_WEIGHT")
    print("="*80)

    class_dist_before = y_train.value_counts(normalize=False)
    class_pct_before = y_train.value_counts(normalize=True) * 100

    classes = np.unique(y_train)
    cw_arr = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    cw_dict = dict(zip(classes, cw_arr))
    sw_train = compute_sample_weight(class_weight=cw_dict, y=y_train)

    print("\nClass weight dictionary (Balanced):")
    for kelas, bobot in cw_dict.items():
        print(f"  {kelas}: {bobot:.4f}")

    weighted_dist = pd.DataFrame({'Kelas': y_train.values, 'Weight': sw_train})
    effective_dist = weighted_dist.groupby('Kelas')['Weight'].sum()
    effective_pct = (effective_dist / effective_dist.sum()) * 100

    comparison_df = pd.DataFrame({
        'Kelas': classes,
        'Jumlah_Asli': [class_dist_before[c] for c in classes],
        'Persen_Asli': [class_pct_before[c] for c in classes],
        'Class_Weight': [cw_dict[c] for c in classes],
        'Jumlah_Efektif': [effective_dist[c] for c in classes],
        'Persen_Efektif': [effective_pct[c] for c in classes]
    })
    print("\n--- RINGKASAN PERBANDINGAN SEBELUM vs SESUDAH WEIGHT ---")
    print(comparison_df.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1 = axes[0, 0]
    comparison_df[['Kelas', 'Jumlah_Asli', 'Jumlah_Efektif']].set_index('Kelas').plot(
        kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4']
    )
    ax1.set_title('Jumlah Sampel: Asli vs Efektif (dengan Weight)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Jumlah Sampel')
    ax1.legend(['Asli (Sebelum)', 'Efektif (Sesudah)'])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

    ax2 = axes[0, 1]
    comparison_df[['Kelas', 'Persen_Asli', 'Persen_Efektif']].set_index('Kelas').plot(
        kind='bar', ax=ax2, color=['#FFB6B9', '#8FE1C4']
    )
    ax2.set_title('Persentase Distribusi: Asli vs Efektif', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Persentase (%)')
    ax2.legend(['Asli (Sebelum)', 'Efektif (Sesudah)'])
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

    ax3 = axes[1, 0]
    sns.barplot(x='Kelas', y='Class_Weight', data=comparison_df, ax=ax3, palette='Set2')
    ax3.set_title('Class Weight per Kelas', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Bobot Kelas')
    for i, v in enumerate(comparison_df['Class_Weight']):
        ax3.text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = comparison_df[['Kelas', 'Persen_Asli', 'Persen_Efektif', 'Class_Weight']].values.tolist()
    table_data = [[c[0], f'{c[1]:.1f}%', f'{c[2]:.1f}%', f'{c[3]:.3f}'] for c in table_data]
    table_data.insert(0, ['Kelas', 'Asli (%)', 'Efektif (%)', 'Weight'])
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    ax4.set_title('Ringkasan: Sebelum vs Sesudah Weight', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    cw_fig_path = os.path.join(ASSETS_DIR, 'comparison_class_weight_sample_weight.png')
    plt.savefig(cw_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gambar disimpan: {cw_fig_path}")

    # ==============================================================================
    # TAHAP 6: OPTIMISASI RANDOM FOREST
    # ==============================================================================
    print("\n" + "="*80)
    print("TAHAP 6: OPTIMISASI RANDOM FOREST (GridSearchCV)")
    print("="*80)

    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    if quick:
        param_grid_rf = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 15],
            'classifier__min_samples_split': [2, 5],
            'classifier__class_weight': ['balanced', cw_dict]
        }
    else:
        param_grid_rf = {
            'classifier__n_estimators': [50, 100, 150, 200],
            'classifier__max_depth': [5, 10, 15, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 3, 5],
            'classifier__class_weight': [cw_dict, 'balanced', None]
        }

    gs_rf = GridSearchCV(rf_pipe, param_grid=param_grid_rf,
                         cv=5, scoring='f1_macro', verbose=0, n_jobs=-1)

    t0 = time.time()
    gs_rf.fit(X_train, y_train)
    t1 = time.time()

    print(f"\n✓ Selesai RF GridSearch dalam {t1-t0:.2f} detik")
    print(f"RF params terbaik: {gs_rf.best_params_}")
    print(f"RF CV F1-macro terbaik: {gs_rf.best_score_:.4f}")

    y_pred_rf = gs_rf.best_estimator_.predict(X_test)
    print("\n--- Classification Report RF (Test Set) ---")
    print(classification_report(y_test, y_pred_rf))

    rf_clf = gs_rf.best_estimator_.named_steps['classifier']
    labels_rf = list(rf_clf.classes_)
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=labels_rf)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='viridis',
                xticklabels=labels_rf, yticklabels=labels_rf)
    plt.title(f'Confusion Matrix - Random Forest ({version.upper()})', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    cm_rf_path = os.path.join(ASSETS_DIR, 'confusion_matrix_rf.png')
    plt.savefig(cm_rf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gambar disimpan: {cm_rf_path}")

    cat_names = list(gs_rf.best_estimator_.named_steps['preprocessor'].\
                    named_transformers_['cat'].get_feature_names_out(categorical_features))
    feat_names = numerical_features + cat_names
    imp_rf = rf_clf.feature_importances_
    fi_rf = pd.DataFrame({'feature': feat_names, 'importance': imp_rf}).\
            sort_values('importance', ascending=False)

    print("\n--- Top 15 Feature Importance RF ---")
    print(fi_rf.head(15))

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=fi_rf.head(15), palette='plasma')
    plt.title(f'Top 15 Feature Importances - Random Forest ({version.upper()})', fontsize=13, fontweight='bold')
    plt.xlabel('Importance'); plt.ylabel('Feature')
    plt.tight_layout()
    fi_rf_path = os.path.join(ASSETS_DIR, 'feature_importance_rf.png')
    plt.savefig(fi_rf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gambar disimpan: {fi_rf_path}")

    # Learning Curve RF
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sizes = np.linspace(0.1, 1.0, 10)
    sizes_abs, tr_rf, va_rf = learning_curve(
        gs_rf.best_estimator_, X_train, y_train,
        train_sizes=sizes, cv=cv5, scoring='f1_macro', shuffle=True, n_jobs=-1
    )

    plt.figure(figsize=(10, 6))
    plt.plot(sizes_abs, tr_rf.mean(axis=1), 'o-', label='Train', color='navy')
    plt.fill_between(sizes_abs, tr_rf.mean(axis=1)-tr_rf.std(axis=1),
                     tr_rf.mean(axis=1)+tr_rf.std(axis=1), alpha=0.2, color='navy')
    plt.plot(sizes_abs, va_rf.mean(axis=1), 'o-', label='Validation (CV)', color='crimson')
    plt.fill_between(sizes_abs, va_rf.mean(axis=1)-va_rf.std(axis=1),
                     va_rf.mean(axis=1)+va_rf.std(axis=1), alpha=0.2, color='crimson')
    plt.xlabel('Jumlah Sampel Training', fontsize=11)
    plt.ylabel('F1-Macro', fontsize=11)
    plt.title(f'Learning Curve - Random Forest ({version.upper()})', fontsize=13, fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    lc_rf_path = os.path.join(ASSETS_DIR, 'learning_curve_rf.png')
    plt.savefig(lc_rf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gambar disimpan: {lc_rf_path}")

    # ROC-AUC RF
    proba_rf = gs_rf.best_estimator_.predict_proba(X_test)
    rf_classes = rf_clf.classes_
    y_test_bin = label_binarize(y_test, classes=rf_classes)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(rf_classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], proba_rf[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    macro_auc_rf = roc_auc_score(y_test_bin, proba_rf, average='macro', multi_class='ovr')

    plt.figure(figsize=(10, 7))
    colors = ['navy', 'crimson', 'darkorange']
    for i, c in enumerate(rf_classes):
        plt.plot(fpr[i], tpr[i], lw=2, color=colors[i % len(colors)],
                 label=f'{c} (AUC={roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.title(f'ROC One-vs-Rest - Random Forest ({version.upper()}, Macro AUC={macro_auc_rf:.3f})',
              fontsize=13, fontweight='bold')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_rf_path = os.path.join(ASSETS_DIR, 'roc_curve_rf.png')
    plt.savefig(roc_rf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gambar disimpan: {roc_rf_path}")

    # ==============================================================================
    # TAHAP 7: OPTIMISASI GRADIENT BOOSTING
    # ==============================================================================
    print("\n" + "="*80)
    print("TAHAP 7: OPTIMISASI GRADIENT BOOSTING (GridSearchCV)")
    print("="*80)

    gb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])

    if quick:
        param_grid_gb = {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 4]
        }
    else:
        param_grid_gb = {
            'classifier__n_estimators': [50, 100, 150, 200],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 4, 5],
            'classifier__subsample': [0.6, 0.7, 0.8],
            'classifier__min_samples_leaf': [2, 5, 10]
        }

    gs_gb = GridSearchCV(gb_pipe, param_grid=param_grid_gb,
                         cv=5, scoring='f1_macro', verbose=0, n_jobs=-1)

    t0 = time.time()
    gs_gb.fit(X_train, y_train, **{'classifier__sample_weight': sw_train})
    t1 = time.time()

    print(f"\n✓ Selesai GB GridSearch dalam {t1-t0:.2f} detik")
    print(f"GB params terbaik: {gs_gb.best_params_}")
    print(f"GB CV F1-macro terbaik: {gs_gb.best_score_:.4f}")

    y_pred_gb = gs_gb.best_estimator_.predict(X_test)
    print("\n--- Classification Report GB (Test Set) ---")
    print(classification_report(y_test, y_pred_gb))

    gb_clf = gs_gb.best_estimator_.named_steps['classifier']
    labels_gb = list(gb_clf.classes_)
    cm_gb = confusion_matrix(y_test, y_pred_gb, labels=labels_gb)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_gb, annot=True, fmt='d', cmap='viridis',
                xticklabels=labels_gb, yticklabels=labels_gb)
    plt.title(f'Confusion Matrix - Gradient Boosting ({version.upper()})', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    cm_gb_path = os.path.join(ASSETS_DIR, 'confusion_matrix_gb.png')
    plt.savefig(cm_gb_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gambar disimpan: {cm_gb_path}")

    imp_gb = gb_clf.feature_importances_
    fi_gb = pd.DataFrame({'feature': feat_names, 'importance': imp_gb}).\
            sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=fi_gb.head(15), palette='plasma')
    plt.title(f'Top 15 Feature Importances - Gradient Boosting ({version.upper()})', fontsize=13, fontweight='bold')
    plt.xlabel('Importance'); plt.ylabel('Feature')
    plt.tight_layout()
    fi_gb_path = os.path.join(ASSETS_DIR, 'feature_importance_gb.png')
    plt.savefig(fi_gb_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gambar disimpan: {fi_gb_path}")

    # Learning Curve GB
    try:
        sizes_abs, tr_gb, va_gb = learning_curve(
            gs_gb.best_estimator_, X_train, y_train,
            train_sizes=sizes, cv=cv5, scoring='f1_macro', shuffle=True, n_jobs=-1,
            params={'classifier__sample_weight': sw_train}
        )
    except Exception:
        sizes_abs, tr_gb, va_gb = learning_curve(
            gs_gb.best_estimator_, X_train, y_train,
            train_sizes=sizes, cv=cv5, scoring='f1_macro', shuffle=True, n_jobs=-1
        )

    plt.figure(figsize=(10, 6))
    plt.plot(sizes_abs, tr_gb.mean(axis=1), 'o-', label='Train', color='navy')
    plt.fill_between(sizes_abs, tr_gb.mean(axis=1)-tr_gb.std(axis=1),
                     tr_gb.mean(axis=1)+tr_gb.std(axis=1), alpha=0.2, color='navy')
    plt.plot(sizes_abs, va_gb.mean(axis=1), 'o-', label='Validation (CV)', color='crimson')
    plt.fill_between(sizes_abs, va_gb.mean(axis=1)-va_gb.std(axis=1),
                     va_gb.mean(axis=1)+va_gb.std(axis=1), alpha=0.2, color='crimson')
    plt.xlabel('Jumlah Sampel Training', fontsize=11)
    plt.ylabel('F1-Macro', fontsize=11)
    plt.title(f'Learning Curve - Gradient Boosting ({version.upper()})', fontsize=13, fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    lc_gb_path = os.path.join(ASSETS_DIR, 'learning_curve_gb.png')
    plt.savefig(lc_gb_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gambar disimpan: {lc_gb_path}")

    # ROC-AUC GB
    proba_gb = gs_gb.best_estimator_.predict_proba(X_test)
    gb_classes = gb_clf.classes_
    y_test_bin_gb = label_binarize(y_test, classes=gb_classes)

    fpr_gb, tpr_gb, roc_auc_gb = {}, {}, {}
    for i in range(len(gb_classes)):
        fpr_gb[i], tpr_gb[i], _ = roc_curve(y_test_bin_gb[:, i], proba_gb[:, i])
        roc_auc_gb[i] = auc(fpr_gb[i], tpr_gb[i])
    macro_auc_gb = roc_auc_score(y_test_bin_gb, proba_gb, average='macro', multi_class='ovr')

    plt.figure(figsize=(10, 7))
    for i, c in enumerate(gb_classes):
        plt.plot(fpr_gb[i], tpr_gb[i], lw=2, color=colors[i % len(colors)],
                 label=f'{c} (AUC={roc_auc_gb[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.title(f'ROC One-vs-Rest - Gradient Boosting ({version.upper()}, Macro AUC={macro_auc_gb:.3f})',
              fontsize=13, fontweight='bold')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_gb_path = os.path.join(ASSETS_DIR, 'roc_curve_gb.png')
    plt.savefig(roc_gb_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gambar disimpan: {roc_gb_path}")

    # ==============================================================================
    # TAHAP 8: PEMILIHAN MODEL TERBAIK & RINGKASAN KOMPARATIF
    # ==============================================================================
    print("\n" + "="*80)
    print("TAHAP 8: PEMILIHAN MODEL TERBAIK & RINGKASAN KOMPARATIF")
    print("="*80)

    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    f1_gb = f1_score(y_test, y_pred_gb, average='macro')

    if f1_gb >= f1_rf:
        best_name = 'Gradient Boosting'
        best_est = gs_gb.best_estimator_
        best_f1 = f1_gb
    else:
        best_name = 'Random Forest'
        best_est = gs_rf.best_estimator_
        best_f1 = f1_rf

    print(f"\n✓ MODEL TERBAIK: {best_name}")
    print(f"F1-macro (Test): {best_f1:.4f}")

    summary_cmp = pd.DataFrame([
        {'model': 'RF', 'f1_macro_test': f1_rf, 'auc_macro_test': macro_auc_rf},
        {'model': 'GB', 'f1_macro_test': f1_gb, 'auc_macro_test': macro_auc_gb},
    ]).sort_values('f1_macro_test', ascending=False).reset_index(drop=True)

    rf_f1_val = summary_cmp.loc[summary_cmp['model']=='RF', 'f1_macro_test'].values[0]
    summary_cmp['delta_f1_vs_RF'] = summary_cmp['f1_macro_test'] - rf_f1_val
    print("\n--- RINGKASAN KOMPARATIF RF vs GB ---")
    print(summary_cmp)

    # ==============================================================================
    # TAHAP 9: UJI SENSITIVITAS BOBOT SKOR PERFORMA
    # ==============================================================================
    print("\n" + "="*80)
    print("TAHAP 9: UJI SENSITIVITAS BOBOT SKOR PERFORMA")
    print("="*80)

    def renorm(d):
        s = sum(d.values())
        return {k: (v / s if s > 0 else 0.0) for k, v in d.items()}

    def apply_weight_modifier(base_w, modifier):
        W = copy.deepcopy(base_w)
        for pos, scale in modifier.get('pos_scale', {}).items():
            if pos in W:
                for feat in W[pos]:
                    W[pos][feat] *= scale
                W[pos] = renorm(W[pos])
        for pos, tweaks in modifier.get('feat_add', {}).items():
            if pos in W:
                for feat, delta in tweaks.items():
                    if feat in W[pos]:
                        W[pos][feat] = max(0.0, W[pos][feat] + delta)
                W[pos] = renorm(W[pos])
        return W

    scenarios = {
        'BASE': {},
        '+10% Forward': {'pos_scale': {'Forward': 1.10}},
        '-10% Forward': {'pos_scale': {'Forward': 0.90}},
        '+10% Midfielder': {'pos_scale': {'Midfielder': 1.10}},
        '+10% Defender': {'pos_scale': {'Defender': 1.10}},
    }

    results = []
    base_w_used = WEIGHTS_BASE
    for name, modifier in scenarios.items():
        W = apply_weight_modifier(base_w_used, modifier)
        df_tmp = df.copy()

        def calc_tmp(row):
            pos = row['Position']
            if pos not in W:
                return 0.0
            return sum(row.get(feat, 0.0) * w for feat, w in W[pos].items())

        df_tmp['Perf_Score_tmp'] = df_tmp.apply(calc_tmp, axis=1)
        if version == 'v2':
            df_tmp['Perf_Q_tmp'] = df_tmp.groupby('Position')['Perf_Score_tmp'].transform(
                lambda s: pd.qcut(s.rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'])
            )
            df_tmp['MV_Q_tmp'] = df_tmp.groupby('Position')['Market_Value'].transform(
                lambda s: pd.qcut(s.rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'])
            )
        else:
            df_tmp['Perf_Norm'] = (df_tmp['Perf_Score_tmp'] - df_tmp['Perf_Score_tmp'].min()) / \
                                  (df_tmp['Perf_Score_tmp'].max() - df_tmp['Perf_Score_tmp'].min())
            df_tmp['MV_Norm'] = (df_tmp['Market_Value'] - df_tmp['Market_Value'].min()) / \
                                (df_tmp['Market_Value'].max() - df_tmp['Market_Value'].min())
            df_tmp['Perf_Q_tmp'] = pd.qcut(df_tmp['Perf_Norm'], 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
            df_tmp['MV_Q_tmp'] = pd.qcut(df_tmp['MV_Norm'], 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')

        def class_tmp(row):
            if row['Perf_Q_tmp'] == 'Q4' and row['MV_Q_tmp'] != 'Q4':
                return 'Undervalued'
            elif row['MV_Q_tmp'] == 'Q4' and row['Perf_Q_tmp'] != 'Q4':
                return 'Overvalued'
            else:
                return 'Fairly Valued'

        y_tmp = df_tmp.apply(class_tmp, axis=1)
        X_tmp = df_tmp[features]

        pipe_sens = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe_sens, X_tmp, y_tmp, cv=skf, scoring='f1_macro', n_jobs=-1)
        results.append({
            'scenario': name,
            'f1_macro_mean': scores.mean(),
            'f1_macro_std': scores.std()
        })

    sensitivity_df = pd.DataFrame(results)
    print("\n--- HASIL UJI SENSITIVITAS BOBOT SKOR PERFORMA ---")
    print(sensitivity_df)

    # ==============================================================================
    # TAHAP 10: SIMPAN ARTEFAK HASIL
    # ==============================================================================
    print("\n" + "="*80)
    print("TAHAP 10: SIMPAN ARTEFAK HASIL")
    print("="*80)

    rf_model_path = os.path.join(MODELS_DIR, f'best_rf_model_{version}.pkl')
    gb_model_path = os.path.join(MODELS_DIR, f'best_gb_model_{version}.pkl')
    joblib.dump(gs_rf.best_estimator_, rf_model_path)
    joblib.dump(gs_gb.best_estimator_, gb_model_path)

    # Simpan juga versi default tanpa suffix
    joblib.dump(gs_rf.best_estimator_, os.path.join(MODELS_DIR, 'best_rf_model.pkl'))
    joblib.dump(gs_gb.best_estimator_, os.path.join(MODELS_DIR, 'best_gb_model.pkl'))
    print(f"\n✓ Model disimpan ke direktori models/:")
    print(f"  - {rf_model_path}")
    print(f"  - {gb_model_path}")

    metrics_summary = {
        'version': version,
        'rf': {
            'f1_macro_test': float(f1_rf),
            'auc_macro_test': float(macro_auc_rf),
            'best_params': gs_rf.best_params_,
            'cv_f1_macro': float(gs_rf.best_score_)
        },
        'gb': {
            'f1_macro_test': float(f1_gb),
            'auc_macro_test': float(macro_auc_gb),
            'best_params': gs_gb.best_params_,
            'cv_f1_macro': float(gs_gb.best_score_)
        }
    }

    metrics_json_path = os.path.join(MODELS_DIR, f'summary_metrics_{version}.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    fi_rf.to_csv(os.path.join(ASSETS_DIR, f'feature_importance_rf_{version}.csv'), index=False)
    fi_gb.to_csv(os.path.join(ASSETS_DIR, f'feature_importance_gb_{version}.csv'), index=False)
    sensitivity_df.to_csv(os.path.join(ASSETS_DIR, f'sensitivity_analysis_{version}.csv'), index=False)
    comparison_df.to_csv(os.path.join(ASSETS_DIR, f'comparison_class_weight_{version}.csv'), index=False)

    print("\n✓ Semua artefak analisis berhasil disimpan.")
    print(f"✓ MODEL TERBAIK: {best_name} (F1-macro: {best_f1:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Analisis Klasifikasi Eredivisie Player Valuation & Performance Alignment"
    )
    parser.add_argument(
        '--version',
        choices=['v1', 'v2'],
        default='v2',
        help="Pilih metodologi: 'v1' (baseline tugas akhir) atau 'v2' (improved P90 + Min-Max per posisi + Grouped Quantiles)"
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help="Lakukan audit perbandingan distribusi dan bias posisi antara v1 dan v2"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help="Gunakan hyperparameter grid ringkas untuk pengujian cepat"
    )

    args = parser.parse_args()

    df_raw = pd.read_csv(DATA_PATH)
    if args.compare:
        compare_v1_vs_v2(df_raw)

    run_pipeline(version=args.version, quick=args.quick)


if __name__ == '__main__':
    main()
