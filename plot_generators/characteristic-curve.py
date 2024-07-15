from __future__ import annotations
import datetime as dt
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
from collections import defaultdict
from pathlib import Path
from diversity import *

load_dotenv()

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

N = int(os.getenv("N"))
# assert is_perfect_square(N), N

NUM_INITIAL_TYPES = int(os.getenv("NUM_INITIAL_TYPES", default=N))
NUM_WORKERS = int(os.getenv("NUM_WORKERS"))
NUM_SIMULATIONS = int(os.getenv("NUM_SIMULATIONS"))
CHUNKSIZE = int(os.getenv("CHUNKSIZE", default=1))
STAT_TO_CALCULATE = os.getenv("STAT_TO_CALCULATE").strip().lower()
MODE = os.getenv("MODE").strip().lower()
MAX_STEPS = int(_max_steps) if (_max_steps := os.getenv("MAX_STEPS")) is not None else None
MUTATION_RATE = float(os.getenv("MUTATION_RATE", default=0))
USE_EXISTING_DATA = os.getenv("USE_EXISTING_DATA", default='false').lower() not in ('false', '0')
OVERWRITE = os.getenv("OVERWRITE", default='false').lower() not in ('false', '0')
CHARACTERISTIC_CURVE_DATA_FILE = Path(os.environ["CHARACTERISTIC_CURVE_DATA_FILE"])
DRAW = os.getenv("DRAW", default='false').lower() not in ('false', '0')
SAMPLE_RATE = float(os.getenv("SAMPLE_RATE", default=0))
TIMESTAMP_STR = dt.datetime.utcnow().strftime("%Y-%m-%d::%H:%M:%S.%f")
USE_TIMESTAMP = str(os.getenv("USE_TIMESTAMP", default=True)).lower() == 'true'

STAT_NAME = {'num_types_left': 'number of types remaining'}
GRAPH_GENERATORS = [
  # GraphGenerator(conjoined_star_graph_N50, 'conjoined star N=50'),
  # GraphGenerator(star_graph_N50, 'star N=50'),
  # GraphGenerator(star_graph_N25, 'star N=25'),
  # GraphGenerator(meta_conjoined_star_graph, 'meta conjoined star graph'),
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
  GraphGenerator(nx.complete_graph, 'complete'),
  GraphGenerator(nx.cycle_graph, 'cycle'),
  GraphGenerator(conjoined_star_graph, 'double star'),
  GraphGenerator(star_graph, 'star'),
  # GraphGenerator(star_joined_stars_3_stars, 'triple star'),
  # GraphGenerator(multi_column_graph_2, 'multi column graph 2'),
  # GraphGenerator(multi_column_graph_3, 'multi column graph 3'),
  # GraphGenerator(multi_column_graph_4, 'multi column graph 4'),
  # GraphGenerator(cyclically_joined_stars_3_stars, 'cyclically joined three stars'),
  # GraphGenerator(square_periodic_grid, 'square periodic grid'),
]


@dataclass
class Stats:
  ABSORBED_SIMPSONS_INDEX: ClassVar[int] = 0
  ABSORBED_NUM_TYPES_LEFT: ClassVar[int] = 1
  ABSORBED_SPATIAL_DIVERSITY: ClassVar[int] = 0
  ABSORBED_SHANNON_INDEX: ClassVar[int] = 0

  simpsons_index: Optional[float] = None
  num_types_left: Optional[float] = None
  spatial_diversity: Optional[float] = None
  shannon_index: Optional[float] = None
  num_samples: Optional[float] = None
  trial: Optional[int] = None

T = TypeVar("T")
def calculate_average(dataset: List[T], extractor: Callable[[T], float], pad_value: float):
  return np.mean([extractor(datum) for datum in dataset] + [pad_value] * max(NUM_SIMULATIONS-len(dataset), 0))

def one_simulation(G: nx.Graph, trial_number: int) -> List[Tuple[int, Stats]]:
  return [
    (steps, Stats(
      simpsons_index=simpsons_index(S, N),
      spatial_diversity=spatial_diversity(S, G),
      num_types_left=num_types_left(S),
      shannon_index=shannon_index(S, N),
      trial=trial_number,
    ))
    for steps, S in trial_absorption_time_interactive(
      G,
      max_steps=MAX_STEPS,
      mutation_rate=MUTATION_RATE,
      num_initial_types=NUM_INITIAL_TYPES,
      sample_rate=SAMPLE_RATE,
    )
  ]

# def get_stats_at_steps(G: nx.Graph, trial_number: int) -> Dict[int, List[Stats]]:
#   results = one_simulation(G, trial_number)
#   return
#   stats_at_steps: Dict[int, List[Stats]] = defaultdict(list)
#   for steps, stats in results:
#     stats_at_steps[steps].append(stats)
#   return stats_at_steps

Run = List[Tuple[int, Stats]]

def simulate_multiple(graph_generator: GraphGenerator, trial_number: int) -> Tuple[str, Run]:
  logger.info((graph_generator.name, trial_number))
  if (G := graph_generator.build_graph(N)) is None: return None
  assert len(G) == N, (graph_generator.name, N, trial_number)
  return (graph_generator.name, one_simulation(G, trial_number))

def process_multiple(transformed_data: Dict[str, Dict[int, Stats]]):
  data: List[Tuple[str, int, Optional[int], Optional[float], Optional[int], Optional[float], Optional[float]]] = []
  for graph_family, stats_at_steps in transformed_data.items():
    for steps, stats in stats_at_steps.items():
      data.append((
        graph_family,
        steps,
        stats.trial,
        stats.num_samples,
        stats.num_types_left,
        stats.simpsons_index,
        stats.spatial_diversity,
        stats.shannon_index,
      ))
  return data

def transform_multiple(data: DefaultDict[str, List[List[Tuple[int, List[Stats]]]]]) -> Dict[str, Dict[int, Stats]]:
  transformed_data: DefaultDict[str, DefaultDict[int, Stats]] = defaultdict(lambda: defaultdict(int))
  for graph_family, runs in data.items():
    stats_at_steps = defaultdict(list)

    for run in runs:
      for steps, stats in run:
        stats_at_steps[steps].append(stats)

    for steps, stats in stats_at_steps.items():
      transformed_data[graph_family][steps] = Stats(
        simpsons_index=calculate_average(dataset=stats, extractor=lambda stat: stat.simpsons_index, pad_value=Stats.ABSORBED_SIMPSONS_INDEX),
        num_types_left=calculate_average(dataset=stats, extractor=lambda stat: stat.num_types_left, pad_value=Stats.ABSORBED_NUM_TYPES_LEFT),
        spatial_diversity=calculate_average(dataset=stats, extractor=lambda stat: stat.spatial_diversity, pad_value=Stats.ABSORBED_SPATIAL_DIVERSITY),
        shannon_index=calculate_average(dataset=stats, extractor=lambda stat: stat.shannon_index, pad_value=Stats.ABSORBED_SHANNON_INDEX),
        num_samples=len(stats),
      )

  return transformed_data

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
      data.append((
        graph_family_name,
        steps,
        stat.trial,
        stat.num_samples,
        stat.num_types_left,
        stat.simpsons_index,
        stat.spatial_diversity,
        stat.shannon_index,
      ))
  return data

def compute() -> pd.DataFrame:
  process = process_single if MODE == 'single' else process_multiple if MODE == 'multiple' else None
  simulate = simulate_single if MODE == 'single' else simulate_multiple if MODE == 'multiple' else None

  data: DefaultDict[str, List[Run]] = defaultdict(list)
  with Pool(NUM_WORKERS) as p:
    for datum in p.starmap(simulate, (
      (graph_generator, trial_number)
      for graph_generator in GRAPH_GENERATORS
      for trial_number in range(NUM_SIMULATIONS)
    )):
      if not datum: continue
      graph_family, run = datum
      data[graph_family].append(run)

  transformed_data = transform_multiple(data)
  processed_data = process(transformed_data)

  df = pd.DataFrame(processed_data, columns=[
    'graph_family',
    'time',
    'trial',
    'num_samples',
    'num_types_left',
    'simpsons_index',
    'spatial_diversity',
    'shannon_index',
  ])
  return df

def draw_multiple(df: pd.DataFrame):
  plot = sns.scatterplot(
    # df[df["time"] > 0],
    data=df,
    x='time',
    # y=df[STAT_TO_CALCULATE],# * df['time'],
    y=STAT_TO_CALCULATE,
    hue='graph_family',
    linewidth=0,
    alpha=1.0,
    # palette='cool',
    # style='graph_family',
    # markers=True,
    # dashes=False,
    # sort=True,
    # markersize=20,
  )

  # plt.xlim(left=1)
  plt.xlabel(r'Time, $T$')
  plt.ylabel(f"Average {STAT_NAME.get(STAT_TO_CALCULATE, STAT_TO_CALCULATE)}, $\\overline{{D}}$")
  # plt.xscale('function', functions=
  #   [lambda x: np.divide(1, x, where=x!=0)]*2
  # )
  # plt.xticks([np.floor(1.1**x) for x in range(int(np.log(df["time"].max())))])
  plt.xscale('log')
  plt.yscale('log')

  # plt.xlim(left=0)
  # plt.title(f'{N=} {NUM_SIMULATIONS=} {MUTATION_RATE=} {NUM_INITIAL_TYPES=}')
  handles, lables = plot.get_legend_handles_labels()
  lgnd = plt.legend(handles, lables, loc='upper left', bbox_to_anchor=(1.05, 1), title='graph family', shadow=True, fancybox=True)
  dpi = 300
  width, height = 2*np.array([3024, 1964])
  fig = plot.get_figure()
  fig.set_size_inches(*(width/dpi, height/dpi))
  fig.savefig(f'plots/characteristic-curve-multiple-{TIMESTAMP_STR}.png', dpi=dpi, bbox_inches='tight')

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
  plt.title(f'{graph_family=} {N=} {NUM_SIMULATIONS=} {MUTATION_RATE=} {NUM_INITIAL_TYPES=}')
  dpi = 300
  width, height = 2*np.array([3024, 1964])
  fig = plot.get_figure()
  fig.set_size_inches(*(width/dpi, height/dpi))
  fig.savefig(f'plots/characteristic-curve-single-{TIMESTAMP_STR}.png', dpi=dpi)

if __name__ == '__main__':
  if MODE == 'single': raise NotImplementedError(MODE)
  if MODE == 'single': assert len(GRAPH_GENERATORS) == 1, [gg.name for gg in GRAPH_GENERATORS]
  draw = draw_single if MODE == 'single' else draw_multiple if MODE == 'multiple' else None

  df = None
  if USE_EXISTING_DATA:
    df = pd.read_pickle(CHARACTERISTIC_CURVE_DATA_FILE)
  else:
    df = compute()

  if OVERWRITE:
    file_name = CHARACTERISTIC_CURVE_DATA_FILE
    if USE_TIMESTAMP:
      file_name = f'{file_name.parent}/{file_name.stem}-{TIMESTAMP_STR}{file_name.suffix}'
    pd.to_pickle(df, file_name)

  if DRAW:
    logger.info('drawing')
    draw(df)