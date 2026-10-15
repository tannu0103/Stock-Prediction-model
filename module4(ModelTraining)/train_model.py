import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_stock_classifier():
    print("Starting Module 4: Machine Learning Model Training & Evaluation...")
    
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "rawData", "processed_feature_matrix.csv")
    models_dir = os.path.join(base_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature dataset not found: {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} rows.")
    
    # Feature columns specification
    feature_cols = [
        "Daily_Return", "SMA_5", "SMA_20", "SMA_Ratio", "Volatility_5d",
        "RSI_14", "MACD", "MACD_Signal", "Volume_Change",
        "sent_compound_mean", "sent_pos_mean", "sent_neg_mean", "sent_polarity", "news_count"
    ]
    
    target_col = "Target_Up"
    
    # Filter valid features
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)
    y = df[target_col].values

    
    # Drop last row if Target is NaN (due to shift)
    if df[target_col].isnull().any():
        valid_idx = ~df[target_col].isnull()
        X = X[valid_idx]
        y = y[valid_idx]
        
    print(f"Final training samples: {len(X)}, Feature count: {X.shape[1]}")
    
    # Sequential time-series train-test split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=4)
    }
    
    best_model = None
    best_name = ""
    best_f1 = -1.0
    best_metrics = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else preds
        
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        try:
            auc = float(roc_auc_score(y_test, probs))
        except Exception:
            auc = 0.5
            
        print(f"\n--- {name} Model Performance ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        
        if f1 > best_f1 or (f1 == best_f1 and acc > best_metrics.get("accuracy", 0)):
            best_f1 = f1
            best_model = model
            best_name = name
            best_metrics = {
                "model_name": name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc
            }
            
    print(f"\nSelecting Best Model: {best_name} (F1 Score: {best_f1:.4f})")
    
    # Feature Importance Analysis
    importances = best_model.feature_importances_
    feat_importance = dict(zip(available_cols, importances.tolist()))
    sorted_importances = dict(sorted(feat_importance.items(), key=lambda item: item[1], reverse=True))
    
    print("\nFeature Importance Breakdown:")
    for feat, imp in sorted_importances.items():
        print(f"  {feat:20s}: {imp:.4f}")
        
    best_metrics["feature_importances"] = sorted_importances
    best_metrics["feature_columns"] = available_cols
    
    # Save Model Artifacts
    model_file = os.path.join(models_dir, "stock_predictor.joblib")
    scaler_file = os.path.join(models_dir, "scaler.joblib")
    metrics_file = os.path.join(models_dir, "metrics.json")
    
    joblib.dump(best_model, model_file)
    joblib.dump(scaler, scaler_file)
    with open(metrics_file, "w") as f:
        json.dump(best_metrics, f, indent=4)
        
    print("\nSuccessfully saved model artifacts:")
    print(f"  Model:   {model_file}")
    print(f"  Scaler:  {scaler_file}")
    print(f"  Metrics: {metrics_file}")

if __name__ == "__main__":
    train_stock_classifier()
