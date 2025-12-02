# model_evaluator.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Visualization libs
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Model Evaluation Script ---")

# CNN-related variables; may stay None if TensorFlow import fails
cnn_model = None
X_test_cnn = None
y_pred_cnn = None

try:
    # --- 1. Load and Prepare the Test Data ---
    print("\n[1/8] Loading and preparing test data...")

    features_df = pd.read_csv("features.csv")

    if 'genre_label' not in features_df.columns:
        raise ValueError("Column 'genre_label' not found in features.csv")

    X = features_df.drop('genre_label', axis=1)
    y = features_df['genre_label']

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("Test data loaded and split successfully.")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # --- 2. Load the scikit-learn Models and the Scaler ---
    print("\n[2/8] Loading scikit-learn models and scaler...")

    scaler = joblib.load("scaler.joblib")
    log_reg_model = joblib.load("logistic_regression_model.joblib")
    svm_model = joblib.load("svm_model.joblib")
    rf_model = joblib.load("random_forest_model.joblib")

    print("Scikit-learn assets loaded successfully:")
    print(f"  Scaler:                {type(scaler)}")
    print(f"  Logistic Regression:   {type(log_reg_model)}")
    print(f"  SVM Model:             {type(svm_model)}")
    print(f"  Random Forest Model:   {type(rf_model)}")

    # --- 3. Load the Keras CNN Model (safely) ---
    print("\n[3/8] Loading Keras CNN model...")

    try:
        import tensorflow as tf

        cnn_model = tf.keras.models.load_model("music_genre_cnn.h5")
        print("Keras CNN model loaded successfully.")
        print(f"  CNN Model:             {type(cnn_model)}")

    except ImportError as e:
        print("\n⚠ TensorFlow import failed. CNN model will be skipped for now.")
        print(f"  Details: {e}")
        print("  You can still evaluate Logistic Regression, SVM, and Random Forest.")
    except OSError as e:
        print("\n⚠ Error while loading 'music_genre_cnn.h5'. CNN will be skipped.")
        print(f"  Details: {e}")

    # --- 4. Prepare Test Data for Different Model Types ---
    print("\n[4/8] Preparing test data for model predictions...")

    X_test_scaled = scaler.transform(X_test)
    print(f"Shape of X_test_scaled (for scikit-learn models): {X_test_scaled.shape}")

    if cnn_model is not None:
        X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)
        print(f"Shape of X_test_cnn (for Keras CNN):             {X_test_cnn.shape}")
    else:
        print("CNN input not created because CNN model is not available.")

    # --- 5. Generate Predictions for Each Model ---
    print("\n[5/8] Generating predictions on the test set...")

    y_pred_log_reg = log_reg_model.predict(X_test_scaled)
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_scaled)

    print("Predictions generated for scikit-learn models.")

    if (cnn_model is not None) and (X_test_cnn is not None):
        y_pred_cnn_probs = cnn_model.predict(X_test_cnn)
        y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)
        print("Predictions generated for Keras CNN model.")
    else:
        print("CNN predictions not generated because CNN model is unavailable.")

    print("\n--- Verifying Prediction Shapes ---")
    print(f"Logistic Regression Predictions Shape: {y_pred_log_reg.shape}")
    print(f"SVM Predictions Shape:                 {y_pred_svm.shape}")
    print(f"Random Forest Predictions Shape:       {y_pred_rf.shape}")
    if y_pred_cnn is not None:
        print(f"CNN Predictions Shape:                 {y_pred_cnn.shape}")
    else:
        print("CNN Predictions Shape:                 None (CNN not used)")

    # --- 6. Classification Reports ---
    print("\n[6/8] Generating classification reports...")

    genre_names = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ]

    print("\n" + "="*70)
    print("   Classification Report: Logistic Regression")
    print("="*70)
    print(classification_report(y_test, y_pred_log_reg, target_names=genre_names))

    print("\n" + "="*70)
    print("   Classification Report: Support Vector Machine (SVM)")
    print("="*70)
    print(classification_report(y_test, y_pred_svm, target_names=genre_names))

    print("\n" + "="*70)
    print("   Classification Report: Random Forest")
    print("="*70)
    print(classification_report(y_test, y_pred_rf, target_names=genre_names))

    if y_pred_cnn is not None:
        print("\n" + "="*70)
        print("   Classification Report: Convolutional Neural Network (CNN)")
        print("="*70)
        print(classification_report(y_test, y_pred_cnn, target_names=genre_names))
    else:
        print("\n" + "="*70)
        print("   Classification Report: CNN")
        print("="*70)
        print("CNN model was not available (TensorFlow error), so no report generated.")

    # --- 7. Confusion Matrices (raw) ---
    print("\n[7/8] Computing confusion matrices...")

    cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    print("\n--- Logistic Regression Confusion Matrix (raw) ---")
    print(cm_log_reg)
    print(f"Shape: {cm_log_reg.shape}")

    print("\n--- SVM Confusion Matrix (raw) ---")
    print(cm_svm)
    print(f"Shape: {cm_svm.shape}")

    print("\n--- Random Forest Confusion Matrix (raw) ---")
    print(cm_rf)
    print(f"Shape: {cm_rf.shape}")

    cm_cnn = None
    if y_pred_cnn is not None:
        cm_cnn = confusion_matrix(y_test, y_pred_cnn)
        print("\n--- CNN Confusion Matrix (raw) ---")
        print(cm_cnn)
        print(f"Shape: {cm_cnn.shape}")
    else:
        print("\n--- CNN Confusion Matrix ---")
        print("Not computed because CNN model is not available.")

    print("\n✅ Confusion matrices computed successfully for all available models.")

    # --- 8. Visualize Confusion Matrices as Heatmaps ---
    print("\n[8/8] Visualizing confusion matrices as heatmaps...")

    def plot_confusion_matrix(cm, labels, title, ax):
        """
        Plot a confusion matrix as a heatmap.
        """
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Confusion Matrices for All Models", fontsize=16)

    # Top-left: Logistic Regression
    plot_confusion_matrix(cm_log_reg, genre_names, "Logistic Regression", axes[0, 0])

    # Top-right: SVM
    plot_confusion_matrix(cm_svm, genre_names, "Support Vector Machine", axes[0, 1])

    # Bottom-left: Random Forest
    plot_confusion_matrix(cm_rf, genre_names, "Random Forest", axes[1, 0])

    # Bottom-right: CNN or placeholder
    if cm_cnn is not None:
        plot_confusion_matrix(cm_cnn, genre_names, "Convolutional Neural Network", axes[1, 1])
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5, 0.5,
            "CNN not available",
            ha="center",
            va="center",
            fontsize=14
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\n✅ Heatmap visualization complete.")

except FileNotFoundError as e:
    print(f"\n❌ ERROR: A required file was not found: {e.filename}")
    print("Make sure these files exist in your project folder:")
    print("  - features.csv")
    print("  - scaler.joblib")
    print("  - logistic_regression_model.joblib")
    print("  - svm_model.joblib")
    print("  - random_forest_model.joblib")
    print("  - music_genre_cnn.h5 (for CNN)")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")
