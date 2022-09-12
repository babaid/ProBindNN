import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd


def comparator(model, dataset,n, filename:str,device="cuda"):
    xs = []
    ys = []
    model.eval()

    for i in tqdm(range(n)):
        x, y, = dataset[i]["mutated"].to(device), dataset[i]["non_mutated"].to(device)
        ddg = dataset[i]["mutated"].ddg
        xs.append(model(x, y).squeeze().item())
        ys.append(ddg)
    xs = np.array(xs)
    ys = np.array(ys)

    df = pd.DataFrame([xs, ys])
    df.to_csv(filename)

    corr = np.corrcoef(xs, ys)[0, 1]

    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys,s=10, marker="x", label = "r = {}".format(corr))
    #ax.set_ylim(-2, 4)
    #ax.set_xlim(-2, 4)
    ax.set_xlabel("Predicted $\Delta \Delta G$")
    ax.set_ylabel("Experimental $\Delta \Delta G$")
    ax.grid(color="grey",alpha=0.8, ls="--")
    #ax.set_aspect('equal')
    ax.legend()
    fig.show()
    
    