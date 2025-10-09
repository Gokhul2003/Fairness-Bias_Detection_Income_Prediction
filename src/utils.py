import os, json
def save_experiment(summary, path='results/experiments.json'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = []
    if os.path.exists(path):
        try:
            with open(path,'r') as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(summary)
    with open(path,'w') as f:
        json.dump(data, f, indent=2)
