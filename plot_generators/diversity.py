import networkx as nx
import numpy as np

from typing import *

def num_types_left(S: Dict[int, Set[Any]]):
  return sum(
    int(len(locations) > 0)
    for locations in S.values()
  )


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

def shannon_index(S: Dict[int, Set[Any]], n: int) -> float:
  return -sum(
    (len(locs) / n * np.log2(len(locs) / n)) if len(locs) > 0 else 0
    for _, locs in S.items()
  )