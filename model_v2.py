import json
import numpy as np
import sys
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
import shap
from model import extract_features_extended as extract_features  # now using extended version

def load_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def build_feature_matrix(episodes):
    X, y = [], []
    for ep in episodes:
        try:
            feats = extract_features(ep)
            if all(np.isfinite(feats)) and not np.any(np.isnan(feats)):
                X.append(feats)
                y.append(ep.get("heuristic_used"))
        except Exception as e:
            print(f"Skipping episode due to error: {e}")
            continue
    return np.array(X), np.array(y)

def train_voting_model(json_file="balanced_eval_data.json", model_file="best_model.pkl"):
    print(f"Loading data from {json_file}")
    episodes = load_data(json_file)
    X, y = build_feature_matrix(episodes)
    # Encode target labels as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Dataset loaded with {len(X)} episodes")

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    X_train = np.nan_to_num(X_train).astype(np.float32)
    X_val = np.nan_to_num(X_val).astype(np.float32)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    # 1. Random Forest
    print("Training Random Forest...")
    rf_params = {
        'n_estimators': [100, 150],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(class_weight=class_weight_dict, random_state=42), rf_params, cv=5, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Best RF Parameters: {rf_grid.best_params_}")

    # 2. MLPClassifier
    print("Training MLPClassifier...")
    mlp_params = {
        'hidden_layer_sizes': [(100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'solver': ['adam'],
        'max_iter': [300],
        'early_stopping': [True]
    }
    mlp_grid = GridSearchCV(MLPClassifier(random_state=42), mlp_params, cv=5, n_jobs=-1, error_score='raise')
    mlp_grid.fit(X_train, y_train)
    best_mlp = mlp_grid.best_estimator_
    print(f"Best MLP Parameters: {mlp_grid.best_params_}")

    # 3. XGBoost
    print("Training XGBoost Classifier...")
    xgb_params = {
        'n_estimators': [100, 150],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
    xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), xgb_params, cv=5, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    print(f"Best XGBoost Parameters: {xgb_grid.best_params_}")

    # 4. LightGBM
    print("Training LightGBM Classifier...")
    lgbm_params = {
        'n_estimators': [100, 150],
        'max_depth': [3, 6, -1],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    lgbm_grid = GridSearchCV(
    LGBMClassifier(
        class_weight=class_weight_dict, 
        verbosity=-1
    ),
    lgbm_params,
    cv=5,
    n_jobs=-1
    )
    lgbm_grid.fit(X_train, y_train)
    best_lgbm = lgbm_grid.best_estimator_
    print(f"Best LightGBM Parameters: {lgbm_grid.best_params_}")

    # 5. Base Model Creation 
    print("Creating Base Models...")
    base_models = [
        ('rf', best_rf),
        ('mlp', best_mlp),
        ('xgb', best_xgb),
        ('lgbm', best_lgbm)
    ]

    print("Running SHAP analysis...")
    voting = VotingClassifier(estimators=base_models, voting='soft')
    voting.fit(X_train, y_train)

    meta_model = LogisticRegression(max_iter=300)
    stacking = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    stacking.fit(X_train, y_train)

    explainer = shap.Explainer(stacking.predict, X_train)
    shap_values = explainer(X_train)
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    important_features = np.where(shap_sum > np.percentile(shap_sum, 20))[0]

    X_train_filtered = X_train[:, important_features]
    X_val_filtered = X_val[:, important_features]

    print("Retraining base models on filtered features...")
    best_rf.fit(X_train_filtered, y_train)
    best_mlp.fit(X_train_filtered, y_train)
    best_xgb.fit(X_train_filtered, y_train)
    best_lgbm.fit(X_train_filtered, y_train)

    base_models = [
        ('rf', best_rf),
        ('mlp', best_mlp),
        ('xgb', best_xgb),
        ('lgbm', best_lgbm)
    ]

    stacking = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    stacking.fit(X_train_filtered, y_train)

    y_val_pred = stacking.predict(X_val_filtered)
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    joblib.dump({
        "model": stacking,
        "label_encoder": label_encoder,
        "important_features": important_features
    }, model_file)

    print(f"StackingClassifier model saved to {model_file}")

    print("Running Permutation Importance...")
    result = permutation_importance(stacking, X_val_filtered, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[::-1]

    print("Top 10 Important Features by Permutation:")
    for idx in sorted_idx[:10]:
        print(f"Feature {idx}: Importance = {result.importances_mean[idx]:.4f}")

    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"\n Validation Accuracy: {accuracy:.4f}")


def predict_heuristic(episode_file, model_file="best_model.pkl"):
    with open(episode_file, 'r') as f:
        episode = json.load(f)
    features = extract_features(episode)
    bundle = joblib.load(model_file)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    prediction_encoded = model.predict(np.array(features).reshape(1, -1))[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    print("Predicted Heuristic:", prediction_label)
    return prediction_label

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "predict":
        if len(sys.argv) < 3:
            print("Usage: python model_voting.py predict path/to/episode.json")
        else:
            predict_heuristic(sys.argv[2])
    else:
        training_file = sys.argv[1] if len(sys.argv) >= 2 else r"test_set\synthetic_a_star_data.json"
        train_voting_model(training_file)
