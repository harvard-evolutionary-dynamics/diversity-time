#!/usr/bin/env python3
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from utils import style, sample, trial_cftime

sns.set_theme(font_scale=2, rc={'text.usetex' : True})
sns.set_style("whitegrid", {'axes.grid' : False})


def fan_graph(B):
  G = nx.DiGraph()

  for b in range(1, B+1):
    G.add_edge(0, -b)
    G.add_edge(-b, +b)
    G.add_edge(+b, 0)

  return G

def fan_fixation_time(NN, samples=1000, overwrite=True, use_existing_file=True):
  data = []
  r = 1
  for N in range(2, NN+1):
    print(N)
    G = nx.to_directed(nx.complete_graph(N))
    nodes = set(G.nodes())
    nodes.pop()
    assert len(nodes) == len(G.nodes()) - 1
    for time in sample(lambda: trial_cftime(G, nodes, r), samples):
      data.append((N, r, time))

  df = pd.DataFrame(data, columns=["N", "r", "Fixation time"])

  plot = sns.lineplot(
    # kind="line",
    data=df,
    x="N",
    y="Fixation time",
    hue="r",
    palette="Paired",
    # units="graph idx",
    # estimator=None,
    # facet_kws={"ylim": (23.698, 23.705)},
    # style="counterexample",
    marker="o",
    # style="logic",
    linestyle="--",
    # dashes=True,
  )

  ax = plt.gca()
  handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles, [f'${r = }$' for r in Rs], title='')
  ax.set(xlabel='Population size, $N$', ylabel='Fixation time, $T$')

  style(plot)
  ax = plt.gca()
  # plt.gca().set_xticks([2*B+1 for B in Bs])

  # plt.xscale('log')
  # plt.yscale('log')
  # ax.ticklabel_format(useOffset=False, style='plain')
  from matplotlib.ticker import ScalarFormatter
  plt.gca().xaxis.set_major_formatter(ScalarFormatter()) 
  plt.gca().xaxis.set_minor_formatter(ScalarFormatter())
  # ax.set_xticks([2*B+1 for B in Bs if (2*B+1) % 10 == 0 and ()])
  # plt.savefig(f'charts/fan-graph-estimated-B-vs-ft-{B}.png', dpi=300, bbox_inches="tight")
  plt.show()

if __name__ == '__main__':
  fan_fixation_time(30)