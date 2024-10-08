import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import os
import functools

from utils import sample
from absorption import trial_absorption_time, dB_trial_absorption_time
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
DYNAMIC = os.getenv("DYNAMIC", default='Birth-death')
SAMPLE_RATE = float(os.getenv("SAMPLE_RATE", default=0))

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

TRIAL_ABSORPTION_TIME_FN = {
  'Birth-death': trial_absorption_time,
  'death-Birth': dB_trial_absorption_time,
}[DYNAMIC]

def samples_info(G: nx.Graph, times: int = NUM_SIMULATIONS):
  samples = list(sample(lambda: TRIAL_ABSORPTION_TIME_FN(G), times=times))
  mean = np.mean(samples)
  return mean


GRAPH_GENERATORS = [
  # GraphGenerator(barbell_graph, 'barbell'),
  GraphGenerator(nx.complete_graph, 'complete'),
  # GraphGenerator(complete_bipartite_graph, 'complete bipartite'),
  GraphGenerator(nx.cycle_graph, 'cycle'),
  GraphGenerator(conjoined_star_graph, 'double star'),
  # GraphGenerator(double_leaved_star, 'double-leaved star'),
  GraphGenerator(nx.path_graph, 'path'),
  # GraphGenerator(perfect_binary_tree, 'perfect binary tree'),
  # GraphGenerator(square_periodic_grid, 'square periodic grid'),
  GraphGenerator(star_graph, 'star'),
  # GraphGenerator(multi_column_graph_2, 'multi column graph 2'),
  # GraphGenerator(multi_column_graph_3, 'multi column graph 3'),
  # GraphGenerator(multi_column_graph_4, 'multi column graph 4'),
  # GraphGenerator(cyclically_joined_stars_3_stars, 'triple star'),
  # GraphGenerator(triple_leaved_star, 'triple-leaved star'),
  # GraphGenerator(nx.wheel_graph, 'wheel'),
]

def simulate(graph_generator: GraphGenerator, n: int):
  logger.info((graph_generator.name, n))
  if G := graph_generator.build_graph(n):
    assert len(G) == n, (graph_generator.name, n)
    abs_time = samples_info(G)
    logger.info(('done', graph_generator.name, n))
    return (n, abs_time, graph_generator.name)
  return None

def should_sample(s):
  return s == np.floor(SAMPLE_RATE ** np.ceil(np.emath.logn(SAMPLE_RATE, s)))

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
        if should_sample(n)
      )):
        if datum:
          data.append(datum)
    df = pd.DataFrame(data, columns=['number_of_nodes', 'absorption_time', 'graph_family'])
  
  if OVERWRITE:
    pd.to_pickle(df, TRENDS_DATA_FILE)

  if not DRAW: return

  logger.info('drawing')
  graph_families = set([gg.name for gg in GRAPH_GENERATORS])
  df = df[df['graph_family'].isin(graph_families)]
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
  plt.ylabel(r'Average absorption time, $\overline{T}$')
  plt.xscale('log')
  plt.yscale('log')
  #get legend and change stuff
  handles, lables = plot.get_legend_handles_labels()
  for h in handles:
    h.set_markersize(20)

  # replace legend using handles and labels from above
  # lgnd = plt.legend(handles, lables, loc='upper right', borderaxespad=0.2, bbox_to_anchor=(1.05, 1), title='graph family', shadow=True, fancybox=True)
  graph_families_legend = plt.legend(handles, lables, loc='upper left', bbox_to_anchor=(1.02, 1), title='graph family', shadow=True, fancybox=True)
  plot.add_artist(graph_families_legend)

  # plt.legend()
  # plt.tight_layout()
  # plot.tick_params(axis='both', which='both', length=5, width=2, color='grey')

  # plot level lines.
  lines = []
  xs = np.array(list(range(10, N+1)))
  # yss = [xs**d-(2**d-1) for d in range(1, 4+1)]
  yss = [xs**d for d in range(1, 4+1)]
  linestyles = [(0, (3,)+(1,)*(2*d+1)) for d in range(1, 4+1)]
  for d in range(1, 4+1):
    lines.append(plt.plot(xs, yss[d-1], linestyle=linestyles[d-1], lw=2, color='grey', alpha=1, label=f'$N^{d}$' if d > 1 else "$N$")[0])
  for d in range(1, 3+1):
    plot.fill_between(xs, yss[d-1], yss[d], alpha=.1)
  plot.legend(lines, [f'$N^{d}$' if d>1 else "$N$" for d in range(1, 4+1)], loc='lower left', bbox_to_anchor=(1.02, 0), title='level lines', shadow=True, fancybox=True)

  dpi = 300
  width, height = 2*np.array([3024, 1964])
  fig = plot.get_figure()
  fig.set_size_inches(*(width/dpi, height/dpi))
  fig.savefig('plots/trends.png', dpi=dpi, bbox_inches='tight', bbox_extra_artists=[graph_families_legend])

if __name__ == '__main__':
  draw(N)