import os
import json
import random
import numpy as np
import joblib
from make_model_input import a_star, generate_random_state, HEURISTICS
from model import extract_features_extended  # make sure this path is correct

def decode_numpy(obj):
    """Recursively decode NumPy data types into native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [decode_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: decode_numpy(v) for k, v in obj.items()}
    else:
        return obj

def generate_eval_data(num_episodes=100, output_dir="eval_data", model_path="best_model.pkl"):
    os.makedirs(output_dir, exist_ok=True)
    solved_state = tuple([1, 2, 3, 4, 5, 6, 7, 8, 0])

    # Load model metadata (label encoder and feature selector)
    model_bundle = joblib.load(model_path)
    label_encoder = model_bundle["label_encoder"]
    important_features = model_bundle.get("important_features", None)

    for i in range(num_episodes):
        goal = generate_random_state(solved_state, moves=random.randint(15, 25))
        start = generate_random_state(goal, moves=random.randint(20, 30))
        heuristic_name, heuristic_fn = random.choice(list(HEURISTICS.items()))

        solution, search_data = a_star(start, goal, heuristic_fn)
        if not solution:
            continue

        episode_data = {
            "goal_state": list(goal),
            "start_state": list(start),
            "solution": [list(state) for state in solution],
            "search_data": search_data,
        }

        features = extract_features_extended(episode_data)
        features = decode_numpy(features)

        if important_features is not None:
            filtered_features = np.array(features)[important_features].tolist()
        else:
            filtered_features = features

        episode = {
            **episode_data,
            "heuristic_used": heuristic_name,
            "heuristic_label": int(label_encoder.transform([heuristic_name])[0]),
            "features": filtered_features
        }

        filename = os.path.join(output_dir, f"episode_{i:03d}.json")
        with open(filename, 'w') as f:
            json.dump(episode, f, indent=2)

        print(f"✅ Saved episode {i+1} using heuristic: {heuristic_name}")

if __name__ == "__main__":
    generate_eval_data(num_episodes=10000)