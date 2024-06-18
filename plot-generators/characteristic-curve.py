from __future__ import annotations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import functools
import os

from utils import sample
from absorption import trial_absorption_time, trial_absorption_time_interactive
from classes import *
from typing import *
from multiprocessing import Lock, Manager, Pool
from dataclasses import dataclass
from dotenv import load_dotenv
from customlogger import logger

load_dotenv()

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

N = int(os.getenv("N"))
# assert is_perfect_square(N), N

NUM_WORKERS = int(os.getenv("NUM_WORKERS"))
NUM_SIMULATIONS = int(os.getenv("NUM_SIMULATIONS"))
CHUNKSIZE = int(os.getenv("CHUNKSIZE", default=NUM_SIMULATIONS // NUM_WORKERS))
STAT_TO_CALCULATE = os.getenv("STAT_TO_CALCULATE").strip().lower()
MODE = os.getenv("MODE").strip().lower()
GRAPH_GENERATORS = [
  GraphGenerator(conjoined_star_graph, 'conjoined star'),
  GraphGenerator(star_graph, 'star'),
  GraphGenerator(meta_conjoined_star_graph, 'meta conjoined star graph'),
  # GraphGenerator(cyclically_joined_stars_3_stars, 'cyclically joined three stars'),
  # GraphGenerator(star_joined_stars_3_stars, 'star joined three stars'),
  # GraphGenerator(double_leaved_star, 'double leaved star'),
  # GraphGenerator(triple_leaved_star, 'triple leaved star'),
  # GraphGenerator(random_regular_6, 'random regular d=6'),
  # GraphGenerator(random_regular_7, 'random regular d=7'),
  # GraphGenerator(random_regular_8, 'random regular d=8'),
  # GraphGenerator(random_regular_9, 'random regular d=9'),
  # GraphGenerator(random_regular_10, 'random regular d=10'),
  # GraphGenerator(random_regular_11, 'random regular d=11'),
  # GraphGenerator(random_regular_12, 'random regular d=12'),
  # GraphGenerator(random_regular_13, 'random regular d=13'),
  # GraphGenerator(random_regular_14, 'random regular d=14'),
  # GraphGenerator(random_regular_15, 'random regular d=15'),
  # GraphGenerator(random_regular_16, 'random regular d=16'),
  # GraphGenerator(random_regular_17, 'random regular d=17'),
  # GraphGenerator(random_regular_18, 'random regular d=18'),
  # GraphGenerator(nx.complete_graph, 'complete'),
  # GraphGenerator(nx.cycle_graph, 'cycle'),
  # GraphGenerator(square_periodic_grid, 'square periodic grid'),
]


def simpsons_index(S: Dict[int, Set[Any]], n: int):
  """complement"""
  return 1 - sum(
    len(locations) * (len(locations) - 1)
    for locations in S.values()
  ) / (n*(n-1)) 

def spatial_diversity(S: Dict[int, Set[Any]], G: nx.Graph) -> float:
  # S is type to set of locations.
  S_rev = {
    v: k
    for k, vs in S.items()
    for v in vs
  }
  return sum(
    int(S_rev[u] != S_rev[v])
    for (u, v) in G.edges()
  ) / G.number_of_edges()

from collections import defaultdict

def num_types_left(S: Dict[int, Set[Any]]):
  return sum(
    int(len(locations) > 0)
    for locations in S.values()
  )

@dataclass
class Stats:
  ABSORBED_SIMPSONS_INDEX: ClassVar[int] = 0
  ABSORBED_NUM_TYPES_LEFT: ClassVar[int] = 1
  ABSORBED_SPATIAL_DIVERSITY: ClassVar[int] = 0

  simpsons_index: float = None
  num_types_left: float = None
  spatial_diversity: Optional[float] = None
  num_samples: Optional[float] = None
  trial: Optional[int] = None

T = TypeVar("T")
def calculate_average(dataset: List[T], extractor: Callable[[T], float], pad_value: float):
  return np.mean([extractor(datum) for datum in dataset] + [pad_value] * max(NUM_SIMULATIONS-len(dataset), 0))

def one_simulation(args: Tuple[nx.Graph, int]) -> List[Tuple[int, Stats]]:
  G, trial = args
  logger.info((G, trial))
  return [
    (steps, Stats(
      simpsons_index=simpsons_index(S, N) if STAT_TO_CALCULATE == 'simpsons_index' else None,
      spatial_diversity=spatial_diversity(S, G) if STAT_TO_CALCULATE == 'spatial_diversity' else None,
      num_types_left=num_types_left(S) if STAT_TO_CALCULATE == 'num_types_left' else None,
      trial=trial,
    ))
    for steps, S in trial_absorption_time_interactive(G)
  ]

import itertools

def get_stats_at_steps(graph_generator: GraphGenerator, N: int) -> Dict[int, List[Stats]]:
  total_results: List[Tuple[int, Stats]] = []
  with Pool(NUM_WORKERS) as p:
    for results in p.imap_unordered(one_simulation, zip((graph_generator.build_graph(N) for _ in range(NUM_SIMULATIONS)), range(NUM_SIMULATIONS)), chunksize=CHUNKSIZE):
      total_results.extend(results)
      
  stats_at_steps: Dict[int, List[Stats]] = defaultdict(list)
  for steps, stats in total_results:
    stats_at_steps[steps].append(stats)

  return stats_at_steps

def simulate_multiple(graph_generator: GraphGenerator, n: int) -> Tuple[str, Dict[int, Stats]]:
  logger.info((graph_generator.name, n))
  if (G := graph_generator.build_graph(n)) is None: return None
  assert len(G) == n, (graph_generator.name, n)

  stats_at_steps = get_stats_at_steps(graph_generator, n)
  avg_stat_at_steps = {
    steps: Stats(
      simpsons_index=calculate_average(dataset=stats, extractor=lambda stat: stat.simpsons_index, pad_value=Stats.ABSORBED_SIMPSONS_INDEX) if STAT_TO_CALCULATE == 'simpsons_index' else None,
      num_types_left=calculate_average(dataset=stats, extractor=lambda stat: stat.num_types_left, pad_value=Stats.ABSORBED_NUM_TYPES_LEFT) if STAT_TO_CALCULATE == 'num_types_left' else None,
      spatial_diversity=calculate_average(dataset=stats, extractor=lambda stat: stat.spatial_diversity, pad_value=Stats.ABSORBED_SPATIAL_DIVERSITY) if STAT_TO_CALCULATE == 'spatial_diversity' else None,
      num_samples=len(stats),
    )
    for steps, stats in stats_at_steps.items()
  }

  return (graph_generator.name, avg_stat_at_steps)

def process_multiple(datum: Tuple[str, Dict[int, Stats]]):
  data = []
  graph_family_name, avg_stats_at_steps = datum
  for steps, stat in avg_stats_at_steps.items():
    data.append((graph_family_name, steps, stat.trial, stat.simpsons_index, stat.num_samples, stat.num_types_left, stat.spatial_diversity))
  return data


def simulate_single(graph_generator: GraphGenerator, n: int) -> Tuple[str, Dict[int, List[Stats]]]:
  logger.info((graph_generator.name, n))
  stats_at_steps: DefaultDict[int, List[Stats]] = defaultdict(list)
  if (G := graph_generator.build_graph(n)) is None: return None

  assert len(G) == n, (graph_generator.name, n)

  stats_at_steps = get_stats_at_steps(graph_generator, n)
  return (graph_generator.name, stats_at_steps)

def process_single(datum: Tuple[str, Dict[int, List[Stats]]]):
  data = []
  graph_family_name, stats_at_steps = datum
  for steps, stats in stats_at_steps.items():
    for stat in stats:
      data.append((graph_family_name, steps, stat.trial, stat.simpsons_index, stat.num_samples, stat.num_types_left, stat.spatial_diversity))
  return data


def compute() -> pd.DataFrame:
  process = process_single if MODE == 'single' else process_multiple if MODE == 'multiple' else None
  simulate = simulate_single if MODE == 'single' else simulate_multiple if MODE == 'multiple' else None

  data = []
  for datum in itertools.starmap(simulate, (
    (graph_generator, N)
    for graph_generator in GRAPH_GENERATORS
  )):
    if not datum: continue
    data.extend(process(datum))

  
  df = pd.DataFrame(data, columns=['graph_family', 'time', 'trial', 'simpsons_index', 'num_samples', 'num_types_left', 'spatial_diversity'])
  return df

def draw_multiple(df: pd.DataFrame):
  plot = sns.scatterplot(
    df,
    x='time',
    y=STAT_TO_CALCULATE,
    hue='graph_family',
    linewidth=0,
    alpha=1.0,
    palette='cool',
    # style='graph_family',
    # markers=True,
    # dashes=False,
    # sort=True,
    # markersize=20,
  )

  plt.xlabel(r'Time, $T$')
  plt.ylabel(f"Average {STAT_TO_CALCULATE}, $\\overline{{D}}$")
  plt.xscale('log')
  plt.yscale('log')
  # plt.xlim(left=0)
  plt.title(f'{N=} {NUM_SIMULATIONS=}')
  dpi = 300
  width, height = 2*np.array([3024, 1964])
  fig = plot.get_figure()
  fig.set_size_inches(*(width/dpi, height/dpi))
  fig.savefig('plots/characteristic-curve-multiple.png', dpi=dpi)

def draw_single(df: pd.DataFrame):
  assert len(GRAPH_GENERATORS) == 1, GRAPH_GENERATORS
  graph_family = GRAPH_GENERATORS[0].name
  plot = sns.lineplot(
    df,
    x='time',
    y=STAT_TO_CALCULATE,
    hue='trial',
    # markers=True,
    legend=False,
    linestyle='--',
    markeredgecolor=None,
    alpha=0.25,
    # style='graph_family',
    # dashes=True,
    # sort=True,
    marker="o",
    markersize=7,
  )

  plt.xlabel(r'Time, $T$')
  plt.ylabel(f"{STAT_TO_CALCULATE}, $D$")
  # plt.xscale('log')
  # plt.yscale('log')
  # plt.xlim(left=0)
  plt.title(f'{graph_family=} {N=} {NUM_SIMULATIONS=}')
  dpi = 300
  width, height = 2*np.array([3024, 1964])
  fig = plot.get_figure()
  fig.set_size_inches(*(width/dpi, height/dpi))
  fig.savefig('plots/characteristic-curve-single.png', dpi=dpi)

if __name__ == '__main__':
  if MODE == 'single': assert len(GRAPH_GENERATORS) == 1, [gg.name for gg in GRAPH_GENERATORS]
  df = compute() 
  # save_data(df)
  draw = draw_single if MODE == 'single' else draw_multiple if MODE == 'multiple' else None
  draw(df)