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
from pathlib import Path
from customlogger import logger

load_dotenv()

N = int(os.environ["N"])
NUM_WORKERS = int(os.getenv("NUM_WORKERS", default=1))
NUM_SIMULATIONS = int(os.getenv("NUM_SIMULATIONS", default=100))
USE_EXISTING_DATA = os.getenv("USE_EXISTING_DATA", default='false').lower() not in ('false', '0')
OVERWRITE = os.getenv("OVERWRITE", default='false').lower() not in ('false', '0')
TRENDS_DATA_FILE = Path(os.environ["TRENDS_DATA_FILE"])
DRAW = os.getenv("DRAW", default='false').lower() not in ('false', '0')

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

def samples_info(G: nx.Graph, times: int = 100):
  samples = list(sample(lambda: trial_absorption_time(G), times=times))
  mean = np.mean(samples)
  return mean


GRAPH_GENERATORS = [
  GraphGenerator(barbell_graph, 'barbell'),
  GraphGenerator(nx.complete_graph, 'complete'),
  GraphGenerator(complete_bipartite_graph, 'complete bipartite'),
  GraphGenerator(nx.cycle_graph, 'cycle'),
  GraphGenerator(conjoined_star_graph, 'double star'),
  GraphGenerator(double_leaved_star, 'double-leaved star'),
  GraphGenerator(nx.path_graph, 'path'),
  GraphGenerator(perfect_binary_tree, 'perfect binary tree'),
  GraphGenerator(square_periodic_grid, 'square periodic grid'),
  GraphGenerator(star_graph, 'star'),
  GraphGenerator(cyclically_joined_stars_3_stars, 'triple star'),
  GraphGenerator(triple_leaved_star, 'triple-leaved star'),
  GraphGenerator(nx.wheel_graph, 'wheel'),
]

def simulate(graph_generator: GraphGenerator, n: int):
  logger.info((graph_generator.name, n))
  if G := graph_generator.build_graph(n):
    assert len(G) == n, (graph_generator.name, n)
    abs_time = samples_info(G)
    return (n, abs_time, graph_generator.name)
  return None


def draw(N):
  df = None
  if USE_EXISTING_DATA:
    df = pd.read_pickle(TRENDS_DATA_FILE)
  else:
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
  
  if OVERWRITE:
    pd.to_pickle(df, TRENDS_DATA_FILE)

  if not DRAW: return

  logger.info('drawing')
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
  # lgnd = plt.legend(handles, lables, loc='upper right', borderaxespad=0.2, bbox_to_anchor=(1.05, 1), title='graph family', shadow=True, fancybox=True)
  lgnd = plt.legend(handles, lables, loc='upper left', bbox_to_anchor=(1.05, 1), title='graph family', shadow=True, fancybox=True)
  # plt.legend()
  # plt.tight_layout()
  # plot.tick_params(axis='both', which='both', length=5, width=2, color='grey')
  dpi = 300
  width, height = 2*np.array([3024, 1964])
  fig = plot.get_figure()
  fig.set_size_inches(*(width/dpi, height/dpi))
  fig.savefig('plots/trends.png', dpi=dpi, bbox_inches='tight')

if __name__ == '__main__':
  draw(N)