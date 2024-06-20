import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import os
import functools

from utils import sample
from absorption import trial_absorption_time
from classes import *
from typing import *
from multiprocessing import Pool
from dotenv import load_dotenv

load_dotenv()

N = int(os.environ["N"])
NUM_WORKERS = int(os.getenv("NUM_WORKERS", default=1))
NUM_SIMULATIONS = int(os.getenv("NUM_SIMULATIONS", default=100))

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

def samples_info(G: nx.Graph, times: int = 100):
  samples = list(sample(lambda: trial_absorption_time(G), times=times))
  mean = np.mean(samples)
  return mean


GRAPH_GENERATORS = [
  GraphGenerator(cyclically_joined_5_leaf_stars, 'cyclically joined 5 leaf stars'),
  GraphGenerator(cyclically_joined_10_leaf_stars, 'cyclically joined 10 leaf stars'),
  GraphGenerator(cyclically_joined_15_leaf_stars, 'cyclically joined 15 leaf stars'),
  GraphGenerator(cyclically_joined_20_leaf_stars, 'cyclically joined 20 leaf stars'),
  # GraphGenerator(nx.complete_graph, 'complete'),
  # GraphGenerator(complete_with_safe, 'complete with safe'),
  # GraphGenerator(star_graph, 'star'),
  # GraphGenerator(conjoined_star_graph, 'conjoined star'),
  # GraphGenerator(custom_long_conjoined_star_graph, 'long conjoined star'),
  # GraphGenerator(nx.cycle_graph, 'cycle'),
  # GraphGenerator(complete_bipartite_graph, 'complete bipartite'),
  # GraphGenerator(nx.path_graph, 'path'),
  # GraphGenerator(barbell_graph, 'barbell'),
  # GraphGenerator(perfect_binary_tree, 'perfect binary tree'),
  # GraphGenerator(perfect_ternary_tree, 'perfect ternary tree'),
  # GraphGenerator(perfect_quadary_tree, 'perfect quadary tree'),
  # GraphGenerator(perfect_fiveary_tree, 'perfect 5-ary tree'),
]

def simulate(graph_generator: GraphGenerator, n: int):
  print(graph_generator.name, n)
  if G := graph_generator.build_graph(n):
    assert len(G) == n, (graph_generator.name, n)
    abs_time = samples_info(G)
    return (n, abs_time, graph_generator.name)
  return None


def draw(N):
  data = []
  with Pool(NUM_WORKERS) as p:
    for datum in p.starmap(simulate, (
      (graph_generator, n)
      for graph_generator in GRAPH_GENERATORS
      for n in range(2, N+1)
    )):
      if datum:
        data.append(datum)

  df = pd.DataFrame(data, columns=['number_of_nodes', 'absorption_time', 'graph_family'])
  print('drawing')
  plot = sns.lineplot(
    df,
    x='number_of_nodes',
    y='absorption_time',
    hue='graph_family',
    style='graph_family',
    markers=True,
    dashes=False,
    sort=True,
    markersize=20,
  )

  plt.xlabel(r'Number of nodes, $N$')
  plt.ylabel(r'Absorption time, $T$')
  plt.xscale('log')
  plt.yscale('log')
  #get legend and change stuff
  handles, lables = plot.get_legend_handles_labels()
  for h in handles:
    h.set_markersize(20)

  # replace legend using handles and labels from above
  lgnd = plt.legend(handles, lables, loc='upper left', borderaxespad=0.2, title='graph family')
  # plt.legend()
  # plt.tight_layout()
  dpi = 300
  width, height = 2*np.array([3024, 1964])
  fig = plot.get_figure()
  fig.set_size_inches(*(width/dpi, height/dpi))
  fig.savefig('plots/trends.png', dpi=dpi)

if __name__ == '__main__':
  draw(N)