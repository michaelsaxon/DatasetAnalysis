import numpy as np
import matplotlib.pyplot as plt


# plot pekos

x = np.arange(0,50) / 50

partition = "train"

model = "RB"

for model in ["BERT", "RB"]:
    for ds in ["S", "M", "A1", "A2", "A3"]:
        y = np.load(f"tmp/_{ds}_{partition}_{model}_peco.tmp.npy")
        plt.plot(x, y, label=ds)
    plt.xlabel("Threshold")
    plt.ylabel("No. Clusters exceeding Threshold")
    plt.legend()
    plt.title(model)
    plt.show()
