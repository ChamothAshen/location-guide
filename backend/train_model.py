import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Paths
DATASET_PATH = os.path.join(os.path.dirname(__file__), "sigiriya_dataset.csv")
MODEL_SAVE_PATH = os.path.join("models", "sigiriya_model.pkl")
DESCRIPTIONS_SAVE_PATH = os.path.join("models", "location_descriptions.json")


def add_features(df):
    """Add engineered features to improve separation between close locations."""
    df = df.copy()
    # Interaction feature helps separate locations with similar lat OR lon
    df['lat_lon_interaction'] = df['lat'] * df['lon']
    # Distance from centroid of all points (fixed values from dataset)
    df['dist_from_center'] = np.sqrt(
        (df['lat'] - 7.957417) ** 2 + (df['lon'] - 80.756434) ** 2
    )
    # Ratio feature captures relative position
    df['lat_lon_ratio'] = df['lat'] / df['lon']
    return df


def augment_data(df, augment_factor=3, noise_std=0.00003):
    """Add more training samples with small GPS noise to improve robustness."""
    augmented = [df]
    for _ in range(augment_factor):
        noisy = df.copy()
        noisy['lat'] = noisy['lat'] + np.random.normal(0, noise_std, size=len(noisy))
        noisy['lon'] = noisy['lon'] + np.random.normal(0, noise_std, size=len(noisy))
        augmented.append(noisy)
    return pd.concat(augmented, ignore_index=True)


def train_sigiriya_model():
    print("🚀 Starting Sigiriya ML Model Training (Enhanced)...")

    os.makedirs("models", exist_ok=True)

    if os.path.exists(DATASET_PATH):
        print(f"📂 Found dataset at {DATASET_PATH}. Loading...")
        df = pd.read_csv(DATASET_PATH)
        df = df.rename(columns={
            'latitude': 'lat',
            'longitude': 'lon',
            'location_name': 'location'
        })
        descriptions_mapping = df.drop_duplicates('location').set_index('location')['description'].to_dict()
    else:
        print("❌ Dataset not found!")
        return

    print(f"📊 Original dataset: {len(df)} rows, {df['location'].nunique()} locations")

    # Step 1: Augment data (4x more training data with realistic GPS noise)
    df_augmented = augment_data(df, augment_factor=3, noise_std=0.00003)
    print(f"📊 After augmentation: {len(df_augmented)} rows")

    # Step 2: Feature engineering
    feature_cols = ['lat', 'lon']
    X = df_augmented[feature_cols]
    y = df_augmented['location']

    X = add_features(X)
    print(f"📐 Features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 3: Build an Ensemble of best models
    print("\n🧠 Training Ensemble Model (GBM + SVM + RF)...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ensemble", VotingClassifier(
            estimators=[
                ("gb", GradientBoostingClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42
                )),
                ("svm", SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
                ("rf", RandomForestClassifier(
                    n_estimators=300, max_depth=15, min_samples_leaf=2, random_state=42
                )),
            ],
            voting='soft'  # Use probability-based voting for better accuracy
        ))
    ])

    model.fit(X_train, y_train)

    # Step 4: Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
    print(f"\n📊 Classification Report:\n{classification_report(y_test, y_pred)}")

    # Step 5: Save
    print("💾 Saving model...")
    joblib.dump(model, MODEL_SAVE_PATH)
    with open(DESCRIPTIONS_SAVE_PATH, 'w') as f:
        json.dump(descriptions_mapping, f, indent=4)
    print(f"✨ Saved to {MODEL_SAVE_PATH}")


def test_model():
    print("\n🔍 Running Test...")
    if not os.path.exists(MODEL_SAVE_PATH):
        print("❌ Model not found.")
        return

    model = joblib.load(MODEL_SAVE_PATH)
    with open(DESCRIPTIONS_SAVE_PATH, 'r') as f:
        descriptions = json.load(f)

    test_coords = pd.DataFrame([[7.95772, 80.76027]], columns=['lat', 'lon'])
    test_coords = add_features(test_coords)

    prediction = model.predict(test_coords)[0]
    description = descriptions.get(prediction, "No description available.")

    print(f"  Input: (7.95772, 80.76027)")
    print(f"  Prediction: [{prediction}]: {description[:60]}...")


if __name__ == "__main__":
    train_sigiriya_model()
    test_model()
