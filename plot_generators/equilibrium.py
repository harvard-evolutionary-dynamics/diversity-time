from __future__ import annotations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import functools
import os

from utils import sample
from absorption import _trial_absorption_time
from classes import *
from typing import *
from multiprocessing import Lock, Manager, Pool
from dataclasses import dataclass
from dotenv import load_dotenv
from customlogger import logger
from collections import defaultdict
from pathlib import Path
from diversity import num_types_left

load_dotenv()

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

N = int(os.getenv("N"))
MAX_STEPS = int(os.getenv("MAX_STEPS"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS"))
NUM_SIMULATIONS = int(os.getenv("NUM_SIMULATIONS"))
NUM_INTERVALS = int(os.getenv("NUM_INTERVALS"))
USE_EXISTING_DATA = os.getenv("USE_EXISTING_DATA", default='false').lower() not in ('false', '0')
OVERWRITE = os.getenv("OVERWRITE", default='false').lower() not in ('false', '0')
EQUILIBRIUM_DATA_FILE = Path(os.environ["EQUILIBRIUM_DATA_FILE"])
DRAW = os.getenv("DRAW", default='false').lower() not in ('false', '0')

INTERVALS = [0.01/N, .1/N, 1/N] # np.linspace(0, 1, NUM_INTERVALS)
NUM_INTERVALS = len(INTERVALS)

GRAPH_GENERATORS = [
  GraphGenerator(nx.complete_graph, 'complete'),
  GraphGenerator(nx.cycle_graph, 'cycle'),
  GraphGenerator(conjoined_star_graph, 'double star'),
  GraphGenerator(star_graph, 'star'),
]

@dataclass
class Equilibrium:
  graph_family: str
  mutation_rate: float
  diversity: float
  trial_number: int

@dataclass
class Stats:
  graph_family: str
  mutation_rate: float
  diversity_mean: float
  diversity_variance: float

def collect(it):
  for e in it:
    return e

def calculate(graph_generator: GraphGenerator, mutation_rate: float, trial_number: int):
  if trial_number == 0:
    logger.info((graph_generator.name, mutation_rate, trial_number))

  G = graph_generator.build_graph(N)
  _, S = collect(_trial_absorption_time(
    G,
    max_steps=MAX_STEPS,
    interactive=False,
    mutation_rate=mutation_rate,
    num_initial_types=1,
    full_final_info=True,
    sample_rate=0,
  ))
  logger.info(('done', graph_generator.name, mutation_rate, trial_number))
  return Equilibrium(
    graph_family=graph_generator.name,
    mutation_rate=mutation_rate,
    diversity=num_types_left(S),
    trial_number=trial_number,
  )

def compute() -> pd.DataFrame:
  data: List[Equilibrium] = []
  with Pool(NUM_WORKERS) as p:
    for datum in p.starmap(calculate, (
      (graph_generator, mutation_rate, trial_number)
      for graph_generator in GRAPH_GENERATORS
      for mutation_rate in INTERVALS
      for trial_number in range(NUM_SIMULATIONS)
    )):
      data.append(datum)

  organized_data = [(e.graph_family, e.mutation_rate, e.diversity) for e in data]
  return pd.DataFrame(data=organized_data, columns=['graph family', 'mutation_rate', 'diversity'])

  # organize.
  # equilibriums: DefaultDict[Tuple[str, float], List[Equilibrium]] = defaultdict(list)
  # for equilibrium in data:
  #   equilibriums[(equilibrium.graph_family, equilibrium.mutation_rate)].append(equilibrium)

  # stats: List[Stats] = []
  # for (graph_family, mutation_rate), dataset in equilibriums.items():
  #   values = [e.diversity for e in dataset]
  #   stats.append(Stats(
  #     graph_family=graph_family,
  #     mutation_rate=mutation_rate,
  #     diversity_mean=np.mean(values),
  #     diversity_variance=np.var(values),
  #   ))

  # organized_data = [(s.graph_family, s.mutation_rate, s.diversity_mean, s.diversity_variance) for s in stats]
  # return pd.DataFrame(data=organized_data, columns=['graph_family', 'mutation_rate', 'diversity_mean', 'diversity_variance'])

from matplotlib.ticker import FormatStrFormatter

def draw(df: pd.DataFrame):
  palette = 'tab10'
  g: plt.Axes = sns.boxplot(
    data=df,
    x='mutation_rate',
    y='diversity',
    hue='graph family',
    palette=palette,
    # meanline=True,
    showmeans=True,
    meanprops={'marker':'o','markerfacecolor':'white','markeredgecolor':'black','markersize':'8'},
  )
  colors = sns.color_palette(palette, len(GRAPH_GENERATORS))


  mutation_rates = INTERVALS
  handles, lables = g.get_legend_handles_labels()
  _ = plt.legend(handles, lables, loc='upper left', bbox_to_anchor=(1.05, 1), title='graph family', shadow=True, fancybox=True)
  g.set(xlabel=r"Mutation rate, $\mu$", ylabel=r"Number of types remaining, $D$")
  xlabels = ['{:.3f}'.format(x) for x in mutation_rates]
  g.set_xticklabels(xlabels)

  PLOT_LINES = False
  if PLOT_LINES:
    xs = []
    ys = []
    seen = set()
    for line in g.get_lines():
      xyd = line.get_xydata()
      if xyd.size == 2:
        x, y = xyd.flatten()
        if x not in seen:
          xs.append(x)
          ys.append(y)
          seen.add(x)

    X = np.array(xs).reshape(NUM_INTERVALS, len(GRAPH_GENERATORS)).T
    Y = np.array(ys).reshape(NUM_INTERVALS, len(GRAPH_GENERATORS)).T
    for idx in range(len(GRAPH_GENERATORS)):
      g.plot(X[idx], Y[idx], linestyle='solid', color=colors[idx], linewidth=5)

  dpi = 300
  width, height = 2*np.array([3024, 1964])
  fig = g.get_figure()
  fig.set_size_inches(*(width/dpi, height/dpi))
  fig.savefig('plots/equilibrium.png', dpi=dpi, bbox_inches='tight')



def main():
  df = None
  if USE_EXISTING_DATA:
    df = pd.read_pickle(EQUILIBRIUM_DATA_FILE)
  else:
    df = compute()

  logger.info(f'\n{df}')
  if OVERWRITE:
    pd.to_pickle(df, EQUILIBRIUM_DATA_FILE)

  if DRAW:
    logger.info('drawing')
    draw(df)

if __name__ == '__main__':
  main()