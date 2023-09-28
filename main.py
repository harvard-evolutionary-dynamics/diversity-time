#!/usr/bin/env python3.11
import argparse
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path


def walk(n):
  partition = [1] * n
  steps = 0
  while (k := len(partition)) > 1:
    r, *_ = random.choices(population=list(range(k)), weights=partition)
    rp, *_ = random.choices(population=list(range(k)), weights=partition)
    partition[rp] += 1
    partition[r] -= 1
    partition = sorted(partition, reverse=True)
    new_partition = []
    for part in partition:
      if part == 0: break
      new_partition.append(part)
    partition = new_partition
    
    steps += 1

  return steps

def partition_walking(args):
  TRIALS = 1000
  SEED = 2023
  sns.set_theme(font_scale=2, rc={'text.usetex' : False})
  sns.set_style("whitegrid", {'axes.grid' : False})
  random.seed(SEED)
  N = args.n
  data = [(n, walk(n)) for n in range(1, N+1) for _ in range(TRIALS)]
  df = pd.DataFrame(data, columns=["n", "steps"])
  sns.boxplot(df, x="n", y="steps", showmeans=True)
  plt.show()



def yield_all_graph6(path: Path):
  with path.open(mode="rb") as f:
    for line in f.readlines():
      line = line.strip()
      if not len(line):
        continue
      yield nx.from_graph6_bytes(line)


def trial_absorption_time(G: nx.Graph):
  # Map of type -> locations.
  S = {idx: {u} for idx, u in enumerate(G.nodes())}
  S_rev = {v: k
    for k, vs in S.items()
    for v in vs
  }
  N = len(G)
  V = G.nodes()
  steps = 0

  while len(S) > 1:
    population_with_weights = [(type_, len(locations)) for type_, locations in S.items()]
    birther_type_ = random.choices(
      population=[type_ for type_, _ in population_with_weights],
      weights=[type_size for _, type_size in population_with_weights],
    )[0]
    birther = random.choice(list(S[birther_type_]))

    dier = random.choice([w for (_, w) in G.edges(birther)])
    dier_type = S_rev[dier]

    # Our graphs should not have self loops.
    assert birther != dier
    if birther_type_ != dier_type:
      S[birther_type_].add(dier)
      S[dier_type].remove(dier)
      if not S[dier_type]:
        S.pop(dier_type)
      S_rev[dier] = birther_type_
    
    steps += 1
  return steps

def sample(fn, times):
  count = 0
  while count < times:
    if (ans := fn()) is not None:
      yield ans
      count += 1

def partitions(collection):
  """Source: https://stackoverflow.com/questions/19368375/set-partitions-in-python"""
  if len(collection) == 1:
    yield (collection,)
    return

  first = collection[0]
  for smaller in partitions(collection[1:]):
    # Insert `first` in each of the subpartition's subsets.
    for n, subset in enumerate(smaller):
      yield smaller[:n] + (((first,) + subset),) + smaller[n+1:]
    # Put `first` in its own subset.
    yield ((first,),) + smaller

def rec_sort(xs):
  """Helper method to sort partitions."""
  if type(xs) is not list and type(xs) is not tuple:
    # We don't want to sort strings.
    return xs

  new_xs = []
  for x in xs:
    new_xs.append(rec_sort(x))
  new_xs.sort()
  return tuple(new_xs)

import scipy.special as sp
def number_of_partitions(n):
  if n == 0: return 1
  return sum(
    int(sp.comb(n-1, k)) * number_of_partitions(k)
    for k in range(n)
  )

import copy
def birth_event(S, u, v):
  assert u != v
  uloc = None
  vloc = None
  for loc, group in enumerate(S):
    if uloc is not None and vloc is not None:
      break
    if u in group:
      uloc = loc
    if v in group:
      vloc = loc

  assert uloc is not None
  assert vloc is not None

  if uloc == vloc: return S
  Tl = [list(group) for group in S]
  Tl[vloc].remove(v)
  Tl[uloc].append(v)
  return tuple(group for group in rec_sort(Tl) if group)
    
from fractions import Fraction
from decimal import Decimal
def absorption_time(G: nx.Graph):
  N = len(G)
  B = number_of_partitions(N)
  A = np.zeros((B, B))
  b = np.zeros((B,))
  S_to_idx = {rec_sort(S): idx for idx, S in enumerate(partitions(tuple(G.nodes())))}
  for S, idx in S_to_idx.items():
    A[idx, idx] = 1
    if len(S) == 1: continue
    for u in G.nodes():
      for (_, v) in G.edges(u):
        T = birth_event(S, u, v)
        A[idx, S_to_idx[T]] += -(1/N) * (1/G.degree(u))
        b[idx] = 1

  t = np.linalg.solve(A, b)
  return (round(Decimal.from_float(t[-1]), 3),) #, Fraction.from_float(t[-1]).limit_denominator(10**9))

import itertools

def cheegers_constant(G: nx.Graph):
  # Source: GPT-4
  nodes = list(G.nodes())
  min_ratio = float('inf') # Initialize the minimum ratio to infinity.
  
  # The subsets considered are all the subsets of the nodes except the empty set and the set of all nodes.
  for i in range(1, len(nodes)):
    for subset in itertools.combinations(nodes, i):
      S = set(subset)
      Sc = set(nodes) - S # Sc is the complement of S.
      
      # Calculate the boundary size, which is the number of edges between S and Sc.
      boundary_size = sum(1 for u, v in itertools.product(S, Sc) if G.has_edge(u, v))
      
      # Calculate the Cheeger’s constant for the current subset and update the minimum ratio.
      ratio = boundary_size / min(len(S), len(Sc))
      min_ratio = min(min_ratio, ratio)
          
  return min_ratio


def samples_info(G: nx.Graph):
  N = len(G)
  samples = list(sample(lambda: trial_absorption_time(G), times=1000))
  mean = np.mean(samples)
  std = np.std(samples)
  print(f"{mean} +/- {std} steps {'*' if len(G.edges()) == N*(N-1)//2 else ''} {nx.edge_connectivity(G)}")

def get_exact(G: nx.Graph):
  N = len(G)
  is_complete = len(G.edges()) == N*(N-1)//2
  at = absorption_time(G)[0]
  print(f"{at} {'<-- complete' if is_complete else ''}")

def main(args):
  DRAW_GRAPH = False
  N = args.n
  for G in yield_all_graph6(Path(f"data/connected-n{N}.g6")):
    if DRAW_GRAPH:
      nx.draw(G)
      plt.show()

    # get_samples(G)
    get_exact(G)


def parse_args():
  parser = argparse.ArgumentParser(description="compute expected number of steps until absorption in multi-type Moran process on an undirected graph.")
  parser.add_argument('-n', type=int, required=True)
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  main(args)