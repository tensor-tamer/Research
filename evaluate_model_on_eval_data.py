import os
import json
import numpy as np
import joblib
from model import extract_features_extended

def predict_heuristic(features, model, important_features):
    features = np.array(features)[important_features].reshape(1, -1)
    return model.predict(features)[0]

def evaluate_model(eval_folder="eval_data", model_path="best_model.pkl"):
    # Load trained model and metadata
    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    important_features = bundle.get("important_features", None)

    if important_features is None:
        print("❌ 'important_features' missing in model. Cannot evaluate.")
        return

    correct = 0
    total = 0

    for file in os.listdir(eval_folder):
        if not file.endswith(".json"):
            continue

        path = os.path.join(eval_folder, file)
        with open(path, 'r') as f:
            episode = json.load(f)

        true_label_str = episode.get("heuristic_used")
        if not true_label_str:
            print(f"⚠️ Missing label in {file}, skipping.")
            continue

        try:
            true_label_encoded = label_encoder.transform([true_label_str])[0]
        except ValueError:
            print(f"⚠️ Unknown label '{true_label_str}' in {file}, skipping.")
            continue

        try:
            features = extract_features_extended(episode)
            predicted_encoded = predict_heuristic(features, model, important_features)
            predicted_label_str = label_encoder.inverse_transform([predicted_encoded])[0]

            total += 1
            if predicted_encoded == true_label_encoded:
                correct += 1

            print(f"{file}: Predicted = {predicted_label_str}, Actual = {true_label_str}")
        except Exception as e:
            print(f"⚠️ Failed on {file}: {e}")

    if total == 0:
        print("❌ No valid labeled episodes evaluated.")
    else:
        accuracy = (correct / total) * 100
        print(f"\n✅ Evaluation Complete: {correct}/{total} correct | Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()