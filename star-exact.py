#!/usr/bin/env python3
import argparse
import networkx as nx
import numpy as np
import scipy.special as sp

from fractions import Fraction
from sympy import *
from typing import *

# Run the following first:
# > pip3 install networkx numpy scipy sympy

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


def number_of_partitions(n):
  if n == 0: return 1
  return sum(
    int(sp.comb(n-1, k)) * number_of_partitions(k)
    for k in range(n)
  )


def birth_event(S, u, v):
  if u == v: return S
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


def absorption_time_float(G: nx.Graph, full: bool = False):
  """No self loops allowed."""
  N = len(G)
  B = number_of_partitions(N)
  A = np.zeros((B, B))
  b = np.zeros((B,))
  S_to_idx = {rec_sort(S): idx for idx, S in enumerate(partitions(tuple(G.nodes())))}
  # print(S_to_idx.keys())
  for S, idx in S_to_idx.items():
    A[idx, idx] = 1
    if len(S) == 1: continue
    for u in G.nodes():
      for (_, v) in G.edges(u):
        T = birth_event(S, u, v)
        A[idx, S_to_idx[T]] += (-1/N) * (1/G.degree(u))
    b[idx] = 1

  t = np.linalg.solve(A, b)
  if full:
    return {
      S: t[idx]
      for S, idx in S_to_idx.items()
    }
  else:
    return t[-1]


def absorption_time_fraction(G: nx.Graph, full: bool = False):
  """No self loops allowed."""
  N = len(G)
  B = number_of_partitions(N)
  A = np.zeros((B, B), dtype=Fraction)
  b = np.zeros((B,), dtype=Fraction)
  S_to_idx = {rec_sort(S): idx for idx, S in enumerate(partitions(tuple(G.nodes())))}
  for S, idx in S_to_idx.items():
    A[idx, idx] = Fraction(1)
    if len(S) == 1: continue
    for u in G.nodes():
      for (_, v) in G.edges(u):
        T = birth_event(S, u, v)
        A[idx, S_to_idx[T]] += Fraction(-1, N) * Fraction(1, G.degree(u))
    b[idx] = Fraction(1)

  Asp = Matrix(A)
  bsp = Matrix(b)
  t = Asp.LUsolve(bsp)
  if full:
    return {
      S: t[idx]
      for S, idx in S_to_idx.items()
    }
  else:
    return t[-1]


def get_exact(G: nx.Graph, full: bool = False, precise: bool = False):
  """
  Input:
  - G: the undirected graph.
  - full: gives solution to every possible starting configuration if true.
          otherwise gives solution to the maximally diverse starting configuration.
  """
  assert all(type(x) is int for x in G.nodes())
  N = len(G)
  at_fn = absorption_time_fraction if precise else absorption_time_float
  at = at_fn(G, full=full)
  return at 


def main(args):
  n = args.n

  # n leaves, 1 center. so n+1 nodes in total.
  G = nx.star_graph(n)

  # there may be a rounding error if precise=False, but precise=False is faster. 
  # abs_time = get_exact(G, full=False, precise=False)

  # # or use the following to get, for example, the absorption time starting at a configuration with red in the center (0)
  # # and blue on the the leaves (1, 2, ..., n):
  S = ((0,1), tuple(range(2, n+1)))
  abs_time = get_exact(G, full=True, precise=False)[S]

  # S should be a tuple of tuples. each inner tuple is the locations of a particular color. so S is a partition {0,1,...,n+1}
  # in the case of a star with n leaves.

  print(abs_time)

def parse_args():
  parser = argparse.ArgumentParser(description="compute expected number of steps until absorption in multi-type Moran process on an undirected graph.")
  parser.add_argument('-n', type=int, required=True)
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  main(args)