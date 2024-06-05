import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from utils import sample
from absorption import trial_absorption_time
from typing import *
from multiprocessing import Pool

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})



def simulate(n: int, p: float):
  print('start', n, p)
  samples = []
  TRIALS = 1_000
  abs_times = []
  for _ in range(TRIALS):
    while not nx.is_connected(G := nx.erdos_renyi_graph(n, p)): ...
    abs_time = trial_absorption_time(G)
    abs_times.append(abs_time)
    # samples.append((n, p, abs_time))
  print('end', n, p)
  return [(n, p, np.mean(abs_times))] # samples

NUM_WORKERS = 16

def draw(N):
  data = []
  with Pool(NUM_WORKERS) as p:
    for datum in p.starmap(simulate, (
      (n, p)
      for n in range(2, N+1)
      for p in (px for px in np.linspace(0, 1, 10*N) if px >= 2*np.log(N)/N)
    )):
      if datum:
        data.extend(datum)

  import matplotlib.cm as cm
  df = pd.DataFrame(data, columns=['number_of_nodes', 'probability', 'absorption_time'])
  plot = sns.lineplot(
    df,
    x='number_of_nodes',
    y='absorption_time',
    hue='probability',
    palette=cm.viridis,
    # style='probability',
    marker='o',
    sort=True,
    markersize=10,
    legend=False
  )


  plt.colorbar(cm.ScalarMappable(cmap=cm.viridis), label=r'Probability, $p$')

  plt.xlabel(r'Number of nodes, $N$')
  plt.ylabel(r'Absorption time, $T$')
  plt.xscale('log')
  plt.yscale('log')
  #get legend and change stuff
  # for h in handles:
  #   h.set_markersize(20)

  # replace legend using handles and labels from above
  # plt.legend()
  # plt.tight_layout()
  dpi = 300
  width, height = 2*np.array([3024, 1964])
  fig = plot.get_figure()
  fig.set_size_inches(*(width/dpi, height/dpi))
  fig.savefig('plots/erdos-renyi.png', dpi=dpi)

if __name__ == '__main__':
  N = 15
  draw(N)