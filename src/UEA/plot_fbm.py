import re
import pandas as pd
import matplotlib.pyplot as plt

log_path = "results/logs_fbm/"

models = [
    "LSTM",
    "Transformer",
    "NCDE",
    "RFormer-G",
    "RFormer-L",
    "RFormer-GL",
]

accuracies = {model: [] for model in models}
epoch_re = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)/\d+\s+\|\s+Loss:\s+(?P<loss>[0-9.]+)\s+\|.*?Test Acc:\s+(?P<test_acc>[0-9.]+)%",
    re.MULTILINE
)

for model in models:
    with open(f"{log_path}FBM__{model}__seed42.log", "r") as f:
        log_content = f.read()
        accuracies[model] = [
            (int(m.group("epoch")), float(m.group("loss")), float(m.group("test_acc")))
            for m in epoch_re.finditer(log_content)
        ]


plt.figure(figsize=(10, 6))
for model in models:
    epochs, losses, test_accs = zip(*accuracies[model])
    plt.plot(epochs, test_accs, label=model)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.legend()
plt.savefig("results/fbm_test_accuracy_plot.png")
plt.show()