# ===========================
# EDA: Exploratory Data Analysis for Audio Features
# ===========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the generated dataset
CSV_PATH = "features.csv"

try:
    # =======================
    # LOAD DATA
    # =======================
    features_df = pd.read_csv(CSV_PATH)
    print("DataFrame loaded successfully!")

    # =======================
    # HEAD (FIRST 5 ROWS)
    # =======================
    print("\n=== First 5 Rows of the Dataset ===")
    print(features_df.head())

    # =======================
    # INFO SUMMARY
    # =======================
    print("\n=== DataFrame Info ===")
    features_df.info()

    # =======================
    # STATISTICAL SUMMARY
    # =======================
    print("\n=== Statistical Summary ===")
    print(features_df.describe())

    # =======================
    # MISSING VALUE CHECK
    # =======================
    print("\n=== Missing Values Check ===")
    missing_values = features_df.isnull().sum()
    print(missing_values)

    if missing_values.sum() == 0:
        print("\nConclusion: GOOD NEWS! No missing values found ✔️")
    else:
        print("\nAlert: Missing values detected ❗")

    # =======================
    # GENRE COUNT BAR PLOT
    # =======================
    print("\n=== Generating Genre Distribution Bar Plot ===")
    genre_names = ['blues', 'classical', 'country', 'disco', 'hiphop',
                   'jazz', 'metal', 'pop', 'reggae', 'rock']

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='genre_label', data=features_df, palette='viridis')
    ax.set_title('Distribution of Music Genres in the Dataset', fontsize=16)
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_ylabel('Number of Segments', fontsize=12)
    ax.set_xticklabels(genre_names, rotation=30)
    plt.tight_layout()
    plt.show()

    # =======================
    # BOX PLOT: Spectral Centroid (Feature 25)
    # =======================
    print("\n=== Generating Box Plot for Spectral Centroid ===")
    plt.figure(figsize=(14, 7))
    box_ax = sns.boxplot(x='genre_label', y='25', data=features_df, palette='cubehelix')
    box_ax.set_title('Spectral Centroid Distribution Across Genres', fontsize=18)
    box_ax.set_xlabel('Genre', fontsize=14)
    box_ax.set_ylabel('Spectral Centroid', fontsize=14)
    box_ax.set_xticklabels(genre_names, rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    # =======================
    # VIOLIN PLOT: First MFCC (Feature 0)
    # =======================
    print("\n=== Generating Violin Plot for First MFCC (Column 0) ===")
    plt.figure(figsize=(14, 7))
    violin_ax = sns.violinplot(x='genre_label', y='0', data=features_df, palette='Spectral')
    violin_ax.set_title('First MFCC (Timbre/Energy) Distribution Across Genres', fontsize=18)
    violin_ax.set_xlabel('Genre', fontsize=14)
    violin_ax.set_ylabel('MFCC 1 Value', fontsize=14)
    violin_ax.set_xticklabels(genre_names, rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    # =======================
    # CORRELATION MATRIX
    # =======================
    print("\n=== Computing Correlation Matrix ===")
    correlation_matrix = features_df.corr()
    print("Correlation matrix computed successfully.")
    print("\nTop 5 rows of correlation matrix:")
    print(correlation_matrix.head())

    # =======================
    # HEATMAP
    # =======================
    print("\n=== Generating Heatmap of Feature Correlations ===")
    plt.figure(figsize=(18, 15))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
    plt.title('Correlation Matrix of Music Features', fontsize=20)
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{CSV_PATH}' was not found. Run feature_extractor.py first.")

except Exception as e:
    print("An unexpected error occurred:", e)
