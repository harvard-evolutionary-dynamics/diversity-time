import copy
import networkx as nx
import numpy as np
import random
import scipy.special as sp

from decimal import Decimal
from fractions import Fraction
from functools import lru_cache

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

def absorption_time(G: nx.Graph, SS=None):
  """No self loops allowed."""
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
        A[idx, S_to_idx[T]] += (-1/N) * (1/G.degree(u))
    b[idx] = 1

  t = np.linalg.solve(A, b)
  idx = -1 if SS is None else S_to_idx[rec_sort(SS)]
  return (t[idx],)

@lru_cache(maxsize=None)
def get_exact(G: nx.Graph, S=None):
  assert all(type(x) is int for x in G.nodes())
  N = len(G)
  at = absorption_time(G, S)[0]
  return at