#!/usr/bin/env python3.11
import argparse
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

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




from functools import lru_cache




def H(N):
  return sum(1/i for i in range(1, N+1))

def compute_grid(N):
  G = nx.Graph()
  for i in range(N):
    for j in range(N):
      G.add_edge((i, j), ((i+1)%N, j))
      G.add_edge((i, j), ((i-1)%N, j))
      G.add_edge((i, j), (i, (j+1)%N))
      G.add_edge((i, j), (i, (j-1)%N))

  print(G)
  print(list(partitions(tuple(G.nodes()))))
  steps = get_exact(G)
  print(steps)



def main(args):
  slowest(args.N) 




def main4(args):
  xs = []
  ys = []
  for n in range(1, args.n+1):
    G = slowest(n)
    abs_time = get_exact(G)
    nx.draw(G)
    plt.show()
    print(n, abs_time)
    xs.append(n)
    ys.append(abs_time)

  plt.plot(xs, ys, 'o-', label='Slowest')
  plt.xlabel(r'Number of nodes, $N$')
  plt.ylabel(r'Expected absorption time, $T$')
  plt.title('Expected absorption time, neutral drift, exact')
  plt.legend()
  plt.show()

@lru_cache(maxsize=None)
def h(parts):
  if any(x < 0 for x in parts): return 0
  parts = tuple(sorted(x for x in parts if x > 0))
  n = sum(parts)
  l = len(parts)
  if parts == (n,): return 0
  return 1 + sum(
    parts[r]*parts[s]/n**2 * h(tuple(x + (-1 if i==r else +1 if i==s else 0) for i, x in enumerate(parts)))
    for r in range(l)
    for s in range(l)
  )

def main2(args):
  for n in range(1, args.n+1):
    G = nx.star_graph(n)
    # for v in G.nodes():
    #   G.add_edge(v, v)
    S = None # (tuple(range(n-1)), (n-1,),) + ()*(n-2) if n > 1 else ((0,),)
    # S = tuple((i,) for i in range(n))
    # guess = n**2-n - sum(
    # # guess = (n-1)**2 - sum(
    #   sum(k*(n+len(S[i])-2*k)/(n-k) for k in range(1, len(S[i])))
    #   for i in range(len(S))
    # )
    abs_time = get_exact(G, S)
    print(abs_time)#, guess)
  # search(G)

def parse_args():
  parser = argparse.ArgumentParser(description="compute expected number of steps until absorption in multi-type Moran process on an undirected graph.")
  parser.add_argument('-n', type=int, required=True)
  return parser.parse_args()


def main5(args):
  for n in range(1, args.n+1):
    print(h((1,)*n))

if __name__ == '__main__':
  args = parse_args()
  main2(args)