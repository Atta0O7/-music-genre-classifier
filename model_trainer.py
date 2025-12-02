# model_trainer.py

# Import the pandas library for data manipulation
import pandas as pd

# Import the function for splitting data into train and test sets
from sklearn.model_selection import train_test_split

# Import the StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

# Import our models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Import the accuracy metric for evaluation
from sklearn.metrics import accuracy_score

# Import joblib for saving our models and scaler
import joblib

# Import LabelEncoder and numpy (if labels are text)
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define the path to our dataset
CSV_PATH = "features.csv"

# Load the dataset from the CSV file into a pandas DataFrame
try:
    print("Loading dataset...")
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")

    # Check if 'genre_label' column exists
    if 'genre_label' not in features_df.columns:
        raise KeyError("Column 'genre_label' not found in features.csv")

    # Separate the features (X) from the target label (y)
    X = features_df.drop('genre_label', axis=1)
    y = features_df['genre_label']

    print("\nShape of X (features):", X.shape)
    print("Shape of y (labels):  ", y.shape)

    # Initialize label_encoder so it always exists
    label_encoder = None

    # --- Label Encoding (only if needed) ---
    # Agar labels text (strings) me hon, to encode them to numeric
    if y.dtype == 'object':
        print("\nLabels are text. Applying LabelEncoder...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print("Labels encoded. Classes:", label_encoder.classes_)

        # Save classes for later use (id -> genre name)
        np.save('label_classes.npy', label_encoder.classes_)
        joblib.dump(label_encoder, 'label_encoder.joblib')

        print("\nLabel mapping (id -> genre):")
        for idx, cls_name in enumerate(label_encoder.classes_):
            print(f"{idx}: {cls_name}")
    else:
        print("\nLabels are already numeric. Skipping LabelEncoder.")
        print("NOTE: These numbers come directly from 'genre_label' in features.csv.")
        print("Make sure you know which number corresponds to which genre.")

    # --- Splitting Data into Training and Testing Sets ---
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
    print("Data split complete.")
    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)

    # --- Scaling Features ---
    print("\nScaling features with StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features have been scaled.")

    # --- Training Models ---

    # 1. Logistic Regression
    print("\n--- Training Logistic Regression Model ---")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    print("Logistic Regression model trained successfully!")

    # 2. Support Vector Machine (SVM)
    print("\n--- Training Support Vector Machine (SVM) Model ---")
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    print("Support Vector Machine model trained successfully!")

    # 3. Random Forest Classifier
    print("\n--- Training Random Forest Classifier Model ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    print("Random Forest Classifier model trained successfully!")

    # --- EVALUATION BLOCK ---

    print("\n--- Evaluating Models on the Test Set ---")

    # Logistic Regression
    y_pred_log_reg = log_reg.predict(X_test_scaled)
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    print(f"Logistic Regression Accuracy: {accuracy_log_reg * 100:.2f}%")

    # Support Vector Machine
    y_pred_svm = svm_model.predict(X_test_scaled)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Support Vector Machine Accuracy: {accuracy_svm * 100:.2f}%")

    # Random Forest Classifier (using .score shortcut)
    accuracy_rf = rf_model.score(X_test_scaled, y_test)
    print(f"Random Forest Classifier Accuracy: {accuracy_rf * 100:.2f}%")

    # --- NEW BLOCK: SAVE SCALER + MODELS WITH JOBLIB ---

    print("\n--- Saving Models and Scaler to Disk ---")

    # Save scaler
    joblib.dump(scaler, 'scaler.joblib')

    # Save models
    joblib.dump(log_reg, 'logistic_regression_model.joblib')
    joblib.dump(svm_model, 'svm_model.joblib')
    joblib.dump(rf_model, 'random_forest_model.joblib')

    print("Scaler and models have been successfully saved to disk.")
    print("The following files have been created in your project directory:")
    print("- scaler.joblib")
    print("- logistic_regression_model.joblib")
    print("- svm_model.joblib")
    print("- random_forest_model.joblib")

    # If we used a LabelEncoder, remind about saved mapping
    if label_encoder is not None:
        print("- label_encoder.joblib")
        print("- label_classes.npy (for id -> genre mapping)")

    print("\nTraining, Evaluation, and Saving complete ✅")

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory as model_trainer.py")
except KeyError as ke:
    print(f"Key Error: {ke}")
    print("Please check that 'genre_label' column exists in features.csv")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
