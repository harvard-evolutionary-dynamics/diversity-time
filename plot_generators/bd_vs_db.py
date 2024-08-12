from utils import yield_all_graph6
from pathlib import Path
from absorption import get_exact
from dataclasses import dataclass
from multiprocessing import Pool
from typing import *
from halo import Halo
from tqdm.contrib.concurrent import process_map

import argparse
import csv
import matplotlib.pyplot as plt
import networkx as nx
import os

NUM_THREADS = int(os.getenv('NUM_THREADS', 1))

@dataclass
class Stat:
  graph: nx.Graph
  bd_absorption_time: float
  db_absorption_time: float

def compute_stat(G: nx.Graph, idx: int) -> Stat:
  bd_abs_time = get_exact(G, bd=True)
  db_abs_time = get_exact(G, bd=False)
  print('done', idx)
  return Stat(graph=G, bd_absorption_time=bd_abs_time, db_absorption_time=db_abs_time)

def get_stats(N: int) -> List[Stat]:
  # with Halo("collecting graphs"):
  graphs = [(G, idx) for idx, G in enumerate(yield_all_graph6(Path(f"data/connected-n{N}.g6")))]

  with Pool(NUM_THREADS) as pool:
    return pool.starmap(compute_stat, graphs)
  # return process_map(compute_stat, graphs, max_workers=NUM_THREADS)

def store_stats(stats: List[Stat], N: int):
  with Path(f'data/bd-vs-db-{N}.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=[
      'graph6',
      'bd_absorption_time',
      'db_absorption_time',
    ])
    writer.writeheader()
    for stat in stats:
      writer.writerow({
        'graph6': nx.graph6.to_graph6_bytes(stat.graph, header=False).hex(),
        'bd_absorption_time': stat.bd_absorption_time,
        'db_absorption_time': stat.db_absorption_time,
      })

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-N', type=int, required=True)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = get_args()
  N = args.N
  stats = get_stats(N)
  # with Halo("storing stats"):
  store_stats(stats, N)
  # print(stats)