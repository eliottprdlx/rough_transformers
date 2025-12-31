import os, re
import pandas as pd

LOG_DIR = "results/logs"
ACC_RE = re.compile(r"final_test_acc:\s*([0-9]*\.?[0-9]+)")

DATASETS = [
    "TSC_ArticularyWordRecognition",
    "TSC_HandMovementDirection",
    "TSC_UWaveGestureLibrary",
]
METHODS = ["LSTM", "Transformer", "RFormer-G", "RFormer-L", "RFormer-GL"]

rows = []
for ds in DATASETS:
    for m in METHODS:
        path = os.path.join(LOG_DIR, f"{ds}__{m}.log")
        if not os.path.exists(path):
            rows.append({"dataset": ds, "method": m, "acc": float("nan")})
            continue
        txt = open(path, "r", encoding="utf-8", errors="ignore").read()
        accs = ACC_RE.findall(txt)
        acc = float(accs[-1]) if accs else float("nan")
        rows.append({"dataset": ds, "method": m, "acc": acc})

df = pd.DataFrame(rows)
table = df.pivot(index="dataset", columns="method", values="acc")

os.makedirs("results", exist_ok=True)
table.to_csv("results/table2_partial_now.csv")

print("\n=== Table 2 (partial, current logs) ===")
print(table)
print("\nSaved -> results/table2_partial_now.csv")