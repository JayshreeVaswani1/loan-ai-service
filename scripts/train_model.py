import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from datetime import datetime
import os
import json

print("Isolation Forest Training Script")
print("=" * 50)

def train_isolation_forest():
    """Train Isolation Forest for fraud detection."""

    # Step 1: Load data
    print("\n STEP 1: Loading data...")

    data_path= 'data/fraud_training_data.csv'

    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        print("     Run: python scripts/generate_fraud_data.py first!")
        return

    # Load the CSV
    df = pd.read_csv(data_path)

    print(f"    Loaded {len(df)} transactions")
    print(f"    Fraud rate: {df['is_fraud'].mean()*100:.1f}%")

    # Step 2: Prepare features and labels
    print("\n STEP 2: Preparing features and labels...")

    # Feature columns (inputs to model)
    feature_columns = [
        'transaction_amount',
        'hour_of_day',
        'day_of_week',
        'location_risk_score',
        'customer_age',
        'transaction_velocity'
    ]

    # X = Features (what model uses to predict)
    X = df[feature_columns]

    # y = Labels (what we're predicting: 0=normal, 1=fraud)
    y = df['is_fraud']

    print(f"    Features: {len(feature_columns)} columns")
    print(f"    X shape: {X.shape} ({X.shape[0]} rows * {X.shape[1]} features)")
    print(f"    y shape: {y.shape} (labels)")

    # Show first few rows
    print("\n   Sample data:")
    print(X.head(3))

    # Step 3: Split data
    print("\n STEP 3: Splitting into train and test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% for testing
        random_state=42,    # Same split every time
        stratify=y          # Keep same fraud % in both set
    )

    print(f"    Training set: {len(X_train)} transactions")
    print(f"    - Normal: {(y_train==0).sum()}")
    print(f"    - Fraud: {(y_train==1).sum()}")
    print(f"    - Fraud rate: {y_train.mean()*100:.1f}%")

    print(f"\n  Test set: {len(X_test)} transactions")
    print(f"    - Normal: {(y_test==0).sum()}")
    print(f"    - Fraud: {(y_test==1).sum()}")
    print(f"    - Fraud rate: {y_test.mean()*100:.1f}%")

    # Step 4: Train Isolation Forest
    print("\n   STEP 4: Train Isolation Forest ...")
    print("     Building 100 decision trees...")

    # Create the model
    model = IsolationForest(
        n_estimators=100,       # Build 100 trees
        contamination=0.1,      # Expect 10% anomalies
        random_state=42,        # Reproducible
        max_samples='auto',     # Auto sample size
        n_jobs=1                # Use all CPU cores
    )

    # Train the model (this is where learning happen)
    model.fit(X_train)

    print("     Training complete!")
    print("     Model learned patterns from 800 transactions")

    # Step 5: Predict on test set
    print("\n   STEP 5: Making predictions on test set...")

    # Predict (-1 for anomaly, 1 for normal)
    predictions = model.predict(X_test)

    # Convert to 0/1 (0=normal, 1=fraud)
    y_pred = (predictions == -1).astype(int)

    # Get anomaly scores (lower = more anomalous)
    anomaly_scores = model.score_samples(X_test)

    print(f"   Predicted {len(y_pred)} transactions")
    print(f"   Flagged as fraud: {y_pred.sum()}")
    print(f"   Flagged as normal: {(y_pred==0).sum()}")

    # Step 6: Evaluate performance
    print("\n   STEP 6: Evaluating model performance...")
    print("="*50)

    # Detailed classification report
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Normal', 'Fraud'],
                                digits=3))

    # Calculate individual metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, -anomaly_scores)    # Negative because lower is worse

    # Print key metrics
    print("\n KEY METRICS:")
    print(f" Precision: {precision:.1%} (When we say fraud, are we right?)")
    print(f" Recall: {recall:.1%} (Did we catch all frauds?)")
    print(f" F1-Score: {f1:.3f} (Balance of both)")
    print(f" ROC-AUC: {roc_auc:.3f} (Overall discrimination)")

    print("\n What that means:")
    print(f" Out of {y_pred.sum()} transactions flagged as fraud,")
    print(f"    {int(precision * y_pred.sum())} are actually fraud ({precision:.1%})")
    print(f"   ✅ Out of {y_test.sum()} actual frauds,")
    print(f"      we caught {int(recall * y_test.sum())} of them ({recall:.1%})")

    # Step 7: Save the model
    print("\n💾 STEP 7: Saving trained model...")

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/isolation_forest_v1.0_{timestamp}.pkl'
    joblib.dump(model, model_path)
    print(f"    Saved versioned model: {model_path}")

    # Save as 'latest' for production
    latest_path = 'models/fraud_detector_latest.pkl'
    joblib.dump(model, latest_path)
    print(f"    Saved latest model: {latest_path}")

    # Save metrics to JSON
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_score': float(roc_auc),
        'timestamp': timestamp,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    metrics_path = 'models/metrics_latest.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent = 2)
    print(f"    Saved metrics: {metrics_path}")

    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETE!")
    print(f"🎯 Model achieved:")
    print(f"   Precision: {precision:.1%}")
    print(f"   Recall: {recall:.1%}")
    print(f"   F1-Score: {f1:.3f}")
    print(f"   ROC-AUC: {roc_auc:.3f}")
    print("=" * 50)

    return model, metrics

def test_saved_model():
    """Test the saved model."""
    print("\n\n🧪 TESTING SAVED MODEL")
    print("=" * 50)

    model = joblib.load('models/fraud_detector_latest.pkl')
    print("✅ Model loaded!")

    # Test fraud
    test_fraud = pd.DataFrame([{
        'transaction_amount': 150000,
        'hour_of_day' : 3,
        'day_of_week' : 5,
        'location_risk_score' : 85,
        'customer_age' : 32,
        'transaction_velocity' : 9
    }])
    pred = model.predict(test_fraud)[0]
    score = model.score_samples(test_fraud)[0]
    print(f"\n📋 HIGH RISK: €150K at 3am")
    print(f"   Prediction: {'🚨 FRAUD!' if pred == -1 else '✅ Normal'}")
    print(f"   Score: {score:.3f}")

    # Test normal
    test_normal = pd.DataFrame([{
        'transaction_amount': 800,
        'hour_of_day' : 14,
        'day_of_week' : 2,
        'location_risk_score' : 25,
        'customer_age' : 45,
        'transaction_velocity' : 2
    }])

    pred2 = model.predict(test_normal)[0]
    score2 = model.score_samples(test_normal)[0]
    print(f"\n📋 NORMAL: €8K at 2pm")
    print(f"   Prediction: {'🚨 FRAUD!' if pred2 == -1 else '✅ Normal'}")
    print(f"   Score: {score2:.3f}")

    print("\n✅ Model works!")
    print("=" * 50)

if __name__ == "__main__":
    train_isolation_forest()
    test_saved_model()
