from __future__ import annotations

import argparse
import datetime as dt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pprint
import seaborn as sns

from absorption import trial_absorption_time_interactive
from classes import *
from collections import defaultdict
from customlogger import logger
from dataclasses import dataclass
from diversity import *
from multiprocessing import Pool
from pathlib import Path
from typing import *


T = TypeVar("T")

@dataclass
class Stats:
  ABSORBED_NUM_TYPES_LEFT: ClassVar[int] = 1
  ABSORBED_SHANNON_INDEX: ClassVar[int] = 0
  ABSORBED_SIMPSONS_INDEX: ClassVar[int] = 0
  ABSORBED_SPATIAL_DIVERSITY: ClassVar[int] = 0

  num_types_left: Optional[float] = None
  shannon_index: Optional[float] = None
  simpsons_index: Optional[float] = None
  spatial_diversity: Optional[float] = None

  num_samples: Optional[float] = None
  trial: Optional[int] = None

Run = List[Tuple[int, Stats]]

@dataclass(frozen=True)
class Params:
  CHARACTERISTIC_CURVE_DATA_FILE: Path
  CHUNKSIZE: int
  DRAW: bool
  GRAPH_GENERATORS: List[GraphGenerator]
  MAX_STEPS: int | None
  MUTATION_RATE: float
  N: int
  NUM_INITIAL_TYPES: int
  NUM_SIMULATIONS: int
  NUM_WORKERS: int
  OVERWRITE: bool
  SAMPLE_RATE: float
  STAT_TO_CALCULATE: str
  TIMESTAMP_STR: str
  USE_EXISTING_DATA: bool
  USE_TIMESTAMP: bool

  def __str__(self):
    return pprint.pformat(self, indent=2)


STAT_NAME = {'num_types_left': 'number of types remaining'}

class Simulations:
  def __init__(self, params: Params):
    self.params = params


  def calculate_average(self, dataset: List[T], extractor: Callable[[T], float], pad_value: float):
    return np.mean([extractor(datum) for datum in dataset] + [pad_value] * max(self.params.NUM_SIMULATIONS-len(dataset), 0))

  def one_simulation(self, G: nx.Graph, trial_number: int) -> List[Tuple[int, Stats]]:
    return [
      (steps, Stats(
        simpsons_index=simpsons_index(S, self.params.N),
        spatial_diversity=spatial_diversity(S, G),
        num_types_left=num_types_left(S),
        shannon_index=shannon_index(S, self.params.N),
        trial=trial_number,
      ))
      for steps, S in trial_absorption_time_interactive(
        G,
        max_steps=self.params.MAX_STEPS,
        mutation_rate=self.params.MUTATION_RATE,
        num_initial_types=self.params.NUM_INITIAL_TYPES,
        sample_rate=self.params.SAMPLE_RATE,
      )
    ]


  def simulate_multiple(self, graph_generator: GraphGenerator, trial_number: int) -> Tuple[str, Run]:
    logger.info((graph_generator.name, trial_number))
    if (G := graph_generator.build_graph(self.params.N)) is None: return None
    assert len(G) == self.params.N, (graph_generator.name, self.params.N, trial_number)
    return (graph_generator.name, self.one_simulation(G, trial_number))

  def process_multiple(self, transformed_data: Dict[str, Dict[int, Stats]]):
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

  def transform_multiple(self, data: DefaultDict[str, List[List[Tuple[int, List[Stats]]]]]) -> Dict[str, Dict[int, Stats]]:
    transformed_data: DefaultDict[str, DefaultDict[int, Stats]] = defaultdict(lambda: defaultdict(int))
    for graph_family, runs in data.items():
      stats_at_steps = defaultdict(list)

      for run in runs:
        for steps, stats in run:
          stats_at_steps[steps].append(stats)

      for steps, stats in stats_at_steps.items():
        transformed_data[graph_family][steps] = Stats(
          simpsons_index=self.calculate_average(dataset=stats, extractor=lambda stat: stat.simpsons_index, pad_value=Stats.ABSORBED_SIMPSONS_INDEX),
          num_types_left=self.calculate_average(dataset=stats, extractor=lambda stat: stat.num_types_left, pad_value=Stats.ABSORBED_NUM_TYPES_LEFT),
          spatial_diversity=self.calculate_average(dataset=stats, extractor=lambda stat: stat.spatial_diversity, pad_value=Stats.ABSORBED_SPATIAL_DIVERSITY),
          shannon_index=self.calculate_average(dataset=stats, extractor=lambda stat: stat.shannon_index, pad_value=Stats.ABSORBED_SHANNON_INDEX),
          num_samples=len(stats),
        )

    return transformed_data

  def compute(self) -> pd.DataFrame:
    data: DefaultDict[str, List[Run]] = defaultdict(list)
    with Pool(self.params.NUM_WORKERS) as p:
      for datum in p.starmap(self.simulate_multiple, (
        (graph_generator, trial_number)
        for graph_generator in self.params.GRAPH_GENERATORS
        for trial_number in range(self.params.NUM_SIMULATIONS)
      )):
        if not datum: continue
        graph_family, run = datum
        data[graph_family].append(run)

    transformed_data = self.transform_multiple(data)
    processed_data = self.process_multiple(transformed_data)

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

  def draw_multiple(self, df: pd.DataFrame):
    plot = sns.scatterplot(
      data=df,
      x='time',
      y=df[self.params.STAT_TO_CALCULATE],#/self.params.N,
      hue='graph_family',
      linewidth=0,
      alpha=1.0,
    )

    plt.xlabel(r'Time, $T$')
    plt.ylabel(f"Average {STAT_NAME.get(self.params.STAT_TO_CALCULATE, self.params.STAT_TO_CALCULATE)}, $\\overline{{D}}$")
    plt.xscale('log')
    plt.yscale('log')

    handles, lables = plot.get_legend_handles_labels()
    _ = plt.legend(handles, lables, loc='upper left', bbox_to_anchor=(1.05, 1), title='graph family', shadow=True, fancybox=True)
    dpi = 300
    width, height = 2*np.array([3024, 1964])
    fig = plot.get_figure()
    fig.set_size_inches(*(width/dpi, height/dpi))
    fig.savefig(f'plots/characteristic-curve-multiple-{self.params.TIMESTAMP_STR}.png', dpi=dpi, bbox_inches='tight')

POSSIBLE_GRAPH_GENERATORS = [
  GraphGenerator(nx.complete_graph, 'complete'),
  GraphGenerator(nx.cycle_graph, 'cycle'),
  GraphGenerator(conjoined_star_graph, 'double star'),
  GraphGenerator(star_graph, 'star'),
]

def get_params(args: argparse.Namespace):
  return Params(
    N=args.N, #  or (_n := int(os.getenv("N"))),
    NUM_INITIAL_TYPES=args.num_initial_types, # or int(os.getenv("NUM_INITIAL_TYPES", default=_n)),
    NUM_WORKERS=args.num_workers, # or int(os.getenv("NUM_WORKERS")),
    NUM_SIMULATIONS=args.num_simulations, # or int(os.getenv("NUM_SIMULATIONS")),
    CHUNKSIZE=args.chunksize, # int(os.getenv("CHUNKSIZE", default=1)),
    STAT_TO_CALCULATE=args.stat_to_calculate, # os.getenv("STAT_TO_CALCULATE").strip().lower(),
    MAX_STEPS=args.max_steps, # int(_max_steps) if (_max_steps := os.getenv("MAX_STEPS")) is not None else None,
    MUTATION_RATE=args.mutation_rate, # float(os.getenv("MUTATION_RATE", default=0)),
    USE_EXISTING_DATA=args.use_existing_data,#os.getenv("USE_EXISTING_DATA", default='false').lower() not in ('false', '0'),
    OVERWRITE=args.overwrite,#os.getenv("OVERWRITE", default='false').lower() not in ('false', '0'),
    CHARACTERISTIC_CURVE_DATA_FILE=args.characteristic_curve_data_file, # Path(os.environ["CHARACTERISTIC_CURVE_DATA_FILE"]),
    DRAW=args.draw,#os.getenv("DRAW", default='false').lower() not in ('false', '0'),
    SAMPLE_RATE=args.sample_rate, # float(os.getenv("SAMPLE_RATE", default=0)),
    TIMESTAMP_STR=args.timestamp_str, # dt.datetime.utcnow().strftime("%Y-%m-%d::%H:%M:%S.%f"),
    USE_TIMESTAMP=args.use_timestamp, # str(os.getenv("USE_TIMESTAMP", default=True)).lower() == 'true',
    GRAPH_GENERATORS=[
      g
      for g in POSSIBLE_GRAPH_GENERATORS
      if g.name in args.graph_generators  
    ],
  )

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--N', type=int, required=True)
  parser.add_argument('--num-initial-types', type=int, required=True)
  parser.add_argument('--num-workers', type=int, required=True)
  parser.add_argument('--num-simulations', type=int, required=True)
  parser.add_argument('--max-steps', type=int, required=True)
  parser.add_argument('--sample-rate', type=float, required=True)
  parser.add_argument('--mutation-rate', type=float, required=True)
  parser.add_argument('--chunksize', type=int, required=True)
  parser.add_argument('--characteristic-curve-data-file', type=Path, required=True)
  parser.add_argument('--timestamp-str', type=str, required=True)
  parser.add_argument('--stat-to-calculate', choices=[
      'num_types_left',
      'simpsons_index',
      'spatial_diversity',
      'shannon_index',
    ],
    type=str,
    required=True,
  )
  parser.add_argument('--graph-generators','--list', nargs='+', type=str, required=True)


  parser.add_argument('--use-existing-data', action='store_true')
  parser.add_argument('--draw', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--use-timestamp', action='store_true')

  return parser.parse_args()

def setup_plotting():
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
  })

if __name__ == '__main__':
  params = get_params(get_args())
  logger.info(str(params))
  setup_plotting()
  simulations = Simulations(params)

  df = None
  if params.USE_EXISTING_DATA:
    df = pd.read_pickle(params.CHARACTERISTIC_CURVE_DATA_FILE)
    if 'params' in df._metadata:
      logger.info('attached params differ from given params:')
      logger.info(str(df.params))
    else:
      logger.warn('loaded dataframe has no params metadata')
  else:
    df = simulations.compute()

  if params.OVERWRITE:
    file_name = params.CHARACTERISTIC_CURVE_DATA_FILE
    if params.USE_TIMESTAMP:
      file_name = f'{file_name.parent}/{file_name.stem}-{params.TIMESTAMP_STR}{file_name.suffix}'

    # Add metadata.
    df._metadata.append('params')
    df.params = params
    pd.to_pickle(df, file_name)

  if params.DRAW:
    logger.info('drawing')
    simulations.draw_multiple(df)