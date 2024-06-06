import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import functools
import os

from utils import sample
from absorption import trial_absorption_time
from classes import *
from typing import *
from multiprocessing import Pool
from dataclasses import dataclass
from dotenv import load_dotenv
from customlogger import logger

load_dotenv()

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

N = int(os.getenv("N"))
assert is_perfect_square(N), N

NUM_WORKERS = int(os.getenv("NUM_WORKERS"))
NUM_SIMULATIONS = int(os.getenv("NUM_SIMULATIONS"))
THRESHOLD = int(os.getenv("THRESHOLD"))
ABSORBED_SIMPSONS_INDEX = 0
GRAPH_GENERATORS = [
  GraphGenerator(nx.complete_graph, 'complete'),
  GraphGenerator(conjoined_star_graph, 'conjoined star'),
  GraphGenerator(nx.cycle_graph, 'cycle'),
  GraphGenerator(square_periodic_grid, 'square periodic grid'),
  GraphGenerator(star_graph, 'star'),
]

def simpsons_index(S: Dict[int, Set[Any]], n: int):
  """complement"""
  return 1 - sum(
    len(locations) * (len(locations) - 1)
    for locations in S.values()
  ) / (n*(n-1)) 


from collections import defaultdict

def simulate(graph_generator: GraphGenerator, n: int) -> Tuple[str, Dict[int, Tuple[float, int]]]:
  logger.info((graph_generator.name, n))
  D_at_steps: DefaultDict[int, List[float]] = defaultdict(list)
  if (G := graph_generator.build_graph(n)) is None: return None

  assert len(G) == n, (graph_generator.name, n)
  for _ in range(NUM_SIMULATIONS):
    for steps, S in trial_absorption_time(G, interactive=True):
      D_at_steps[steps].append(simpsons_index(S, N))

  avg_D_at_steps = {
    steps: (np.mean(Ds + [ABSORBED_SIMPSONS_INDEX] * max(NUM_SIMULATIONS-len(Ds), 0)), len(Ds))
    for steps, Ds in D_at_steps.items()
  }

  return (graph_generator.name, avg_D_at_steps)

def compute() -> pd.DataFrame:
  data = []
  with Pool(NUM_WORKERS) as p:
    for datum in p.starmap(simulate, (
      (graph_generator, N)
      for graph_generator in GRAPH_GENERATORS
    )):
      if not datum: continue
      graph_family_name, avg_D_at_steps = datum
      for steps, (avg_D, num_samples) in avg_D_at_steps.items():
        data.append((graph_family_name, steps, avg_D, num_samples))
  
  df = pd.DataFrame(data, columns=['graph_family', 'time', 'average_simpsons_index', 'num_samples'])
  return df

def draw(df: pd.DataFrame):
  plot = sns.scatterplot(
    df[df['num_samples'] >= THRESHOLD],
    x='time',
    y='average_simpsons_index',
    hue='graph_family',
    linewidth=0,
    alpha=1.0
    # style='graph_family',
    # markers=True,
    # dashes=False,
    # sort=True,
    # markersize=20,
    # opacity='num_samples',
  )
  sns.scatterplot(
    df[df['num_samples'] < THRESHOLD],
    ax=plot,
    x='time',
    y='average_simpsons_index',
    hue='graph_family',
    linewidth=0,
    legend=False,
    alpha=0.05,
    # style='graph_family',
    # markers=True,
    # dashes=False,
    # sort=True,
    # markersize=20,
    # opacity='num_samples',
  )

  plt.xlabel(r'Time, $T$')
  plt.ylabel(r"Average Simpon's Index, $\overline{D}$")
  # plt.xscale('squareroot')
  # plt.yscale('squareroot')
  # plt.xscale('log')
  # plt.yscale('log')
  plt.xlim(left=0)
  # plt.xscale('function', functions=(lambda x: print(x) or np.log(1/(1-np.log((2+x)))-1), lambda x: np.exp(1-(1/(np.exp(x)+1)))-2))
  plt.title(f'{N=} {NUM_SIMULATIONS=} {THRESHOLD=}')
  #get legend and change stuff
  # handles, lables = plot.get_legend_handles_labels()
  # for h in handles:
  #   h.set_markersize(20)

  # replace legend using handles and labels from above
  # lgnd = plt.legend(handles, lables, loc='upper left', borderaxespad=0.2, title='graph family')
  # plt.legend()
  # plt.tight_layout()
  dpi = 300
  width, height = 2*np.array([3024, 1964])
  fig = plot.get_figure()
  fig.set_size_inches(*(width/dpi, height/dpi))
  fig.savefig('plots/characteristic-curve.png', dpi=dpi)


if __name__ == '__main__':
  df = compute() 
  # save_data(df)
  draw(df)
  