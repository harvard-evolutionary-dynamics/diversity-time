import copy
import networkx as nx
import numpy as np
import random
import scipy.special as sp

from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from typing import *
from collections import defaultdict

def trial_absorption_time_interactive(
  G: nx.Graph,
  max_steps: Optional[int] = None,
  mutation_rate: float = 0,
  num_initial_types: Optional[int] = None,
  sample_rate: int = 0,
):
  return _trial_absorption_time(
    G,
    max_steps=max_steps,
    interactive=True,
    mutation_rate=mutation_rate,
    num_initial_types=num_initial_types or len(G),
    full_final_info=False,
    sample_rate=sample_rate,
  )

def trial_absorption_time(G: nx.Graph):
  for e in _trial_absorption_time(
    G,
    max_steps=None,
    interactive=False,
    num_initial_types=len(G),
    mutation_rate=0,
    full_final_info=False,
    sample_rate=0,
  ):
    return e
  return None

def _trial_absorption_time(
  G: nx.Graph,
  *,
  max_steps: Optional[int],
  interactive: bool,
  mutation_rate: float,
  num_initial_types: int,
  full_final_info: bool,
  sample_rate: int,
):
  """birth-death"""
  assert 0 <= mutation_rate <= 1, mutation_rate
  assert sample_rate >= 0, sample_rate

  if not isinstance(G, nx.DiGraph):
    G: nx.DiGraph = nx.to_directed(G)

  assert nx.is_strongly_connected(G)

  # Map of type -> locations.
  initial_types = [
    idx
    for idx in range(num_initial_types)
    for _ in range(len(G) // num_initial_types)
  ]
  assert len(initial_types) <= len(G)
  for idx in range(num_initial_types):
    if len(initial_types) == len(G): break
    initial_types.append(idx)

  assert len(initial_types) == len(G)
  random.shuffle(initial_types)

  S: DefaultDict[int, Set[Any]] = defaultdict(set)
  for idx, u in zip(initial_types, G.nodes()):
    S[idx].add(u)

  max_type = num_initial_types-1
  S_rev = {
    v: k
    for k, vs in S.items()
    for v in vs
  }
  should_sample = lambda s: (sample_rate == 0) or (s == np.floor(sample_rate ** np.ceil(np.emath.logn(sample_rate, s))))

  steps = 0
  if interactive: yield (steps, S)
  while (mutation_rate > 0 or len(S) > 1) and (max_steps is None or steps < max_steps):
    population_with_weights = [(type_, len(locations)) for type_, locations in S.items()]
    birther_type_ = random.choices(
      population=[type_ for type_, _ in population_with_weights],
      weights=[type_size for _, type_size in population_with_weights],
    )[0]
    birther = random.choice(list(S[birther_type_]))

    dier = random.choice([w for (_, w) in G.out_edges(birther)])
    dier_type = S_rev[dier]

    # Our graphs should not have self loops.
    assert birther != dier

    # possible mutation.
    if random.random() < mutation_rate:
      max_type += 1 # introduce new type.
      birther_type_ = max_type
      S[birther_type_] = set()

    if birther_type_ != dier_type:
      S[birther_type_].add(dier)
      S[dier_type].remove(dier)
      if not S[dier_type]:
        S.pop(dier_type)
      S_rev[dier] = birther_type_
    
    steps += 1
    
    if interactive and (sample_rate == 0 or should_sample(steps)):
      yield (steps, S)

  if not interactive:
    yield (steps, S) if full_final_info else steps

def dB_trial_absorption_time(G: nx.Graph):
  for e in _dB_trial_absorption_time(G, max_steps=None, interactive=False, num_initial_types=len(G), mutation_rate=0, full_final_info=False):
    return e
  return None

def _dB_trial_absorption_time(G: nx.Graph, *, max_steps: Optional[int], interactive: bool, mutation_rate: float, num_initial_types: int, full_final_info: bool):
  """death-Birth"""
  assert 0 <= mutation_rate <= 1, mutation_rate
  initial_types = [
    idx
    for idx in range(num_initial_types)
    for _ in range(len(G) // num_initial_types)
  ]
  assert len(initial_types) <= len(G)
  for idx in range(num_initial_types):
    if len(initial_types) == len(G): break
    initial_types.append(idx)

  assert len(initial_types) == len(G)
  random.shuffle(initial_types)

  # Map of type -> locations.
  S: DefaultDict[int, Set[Any]] = defaultdict(set)
  for idx, u in zip(initial_types, G.nodes()):
    S[idx].add(u)

  max_type = num_initial_types-1
  S_rev = {v: k
    for k, vs in S.items()
    for v in vs
  }

  N = len(G)
  V = G.nodes()
  steps = 0

  if interactive: yield (steps, S)
  while (mutation_rate > 0 or len(S) > 1) and (max_steps is None or steps < max_steps):
    population_with_weights = [(type_, len(locations)) for type_, locations in S.items()]
    dier_type = random.choices(
      population=[type_ for type_, _ in population_with_weights],
      weights=[type_size for _, type_size in population_with_weights],
    )[0]
    dier = random.choice(list(S[dier_type]))

    birther = random.choice([v for _, v in G.edges(dier)])
    birther_type_ = S_rev[birther]

    # Our graphs should not have self loops.
    assert birther != dier

    # possible mutation.
    if random.random() < mutation_rate:
      max_type += 1 # introduce new type.
      birther_type_ = max_type
      S[birther_type_] = set()

    if birther_type_ != dier_type:
      S[birther_type_].add(dier)
      S[dier_type].remove(dier)
      if not S[dier_type]:
        S.pop(dier_type)
      S_rev[dier] = birther_type_
    
    steps += 1
    if interactive: yield (steps, S)

  if not interactive:
    yield (steps, S) if full_final_info else steps


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

@lru_cache(maxsize=None)
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


def list_partitions(N):
  return [rec_sort(S) for S in partitions(tuple(range(N)))]

@lru_cache(maxsize=None)
def get_indices(N: int):
  return {rec_sort(S): idx for idx, S in enumerate(partitions(tuple(range(N))))}

def absorption_time(G: nx.Graph, SS=None, full=False, bd=True):
  """No self loops allowed. birth-death by default."""
  N = len(G)
  B = number_of_partitions(N)
  A = np.zeros((B, B))
  b = np.zeros((B,))
  S_to_idx = {rec_sort(S): idx for idx, S in enumerate(partitions(tuple(G.nodes())))}
  normalize = (lambda u,_: 1./G.degree(u)) if bd else (lambda _,v: 1./G.degree(v)) 
  for S, idx in S_to_idx.items():
    A[idx, idx] = 1
    if len(S) == 1: continue
    for u in G.nodes():
      for (_, v) in G.edges(u):
        T = birth_event(S, u, v)
        A[idx, S_to_idx[T]] += (-1/N) * normalize(u, v)
    b[idx] = 1

  t = np.linalg.solve(A, b)
  if full and SS is None:
    return {
      S: t[idx]
      for S, idx in S_to_idx.items()
    }
  else:
    idx = -1 if SS is None else S_to_idx[rec_sort(SS)]
    return (t[idx],)

from sympy import *
def absorption_time_frac(G: nx.Graph, SS=None, full=False):
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
  if full and SS is None:
    return {
      S: t[idx]
      for S, idx in S_to_idx.items()
    }
  else:
    idx = -1 if SS is None else S_to_idx[rec_sort(SS)]
    return (t[idx],)


# @lru_cache(maxsize=None)
def get_exact(G: nx.Graph, S=None, full=False, bd=True):
  assert all(type(x) is int for x in G.nodes())
  N = len(G)
  at = absorption_time(G, S, full=full, bd=bd)
  return at if full else at[-1]