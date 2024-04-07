import networkx as nx
import numpy as np
import random

from fractions import Fraction
from typing import *

def simulate_absorption_time(G: nx.DiGraph):
  N = len(G)
  types = set(range(N))
  node_to_type = {u: type for u, type in zip(G.nodes(), types)}
  active_edges = {u: {v for _, v in G.out_edges(u)} for u in G.nodes()}
  prob_active = Fraction(1)
  total_steps = 0
  while prob_active > 0:
    # wait for active step.
    num_inactive_steps = np.random.geometric(p=prob_active)
    total_steps += num_inactive_steps
    # pick birth node.
    u = random.choices(
      list(G.nodes()),
      weights=np.array([len(active_edges[u])/G.out_degree(u) for u in G.nodes()]) / (N*prob_active)
    )[0]
    # pick death node.
    v = random.choice(list(active_edges[u]))
    # update.
    node_to_type[v] = node_to_type[u]
    for a, _ in G.in_edges(v):
      if node_to_type[a] == node_to_type[v] and v in active_edges[a]:
        active_edges[a].remove(v)
        prob_active -= Fraction(1, N*G.out_degree(a))
      elif node_to_type[a] != node_to_type[v] and v not in active_edges[a]:
        active_edges[a].add(v)
        prob_active += Fraction(1, N*G.out_degree(a))
    for _, b in G.out_edges(v):
      if node_to_type[v] == node_to_type[b] and b in active_edges[v]:
        active_edges[v].remove(b)
        prob_active -= Fraction(1, N*G.out_degree(v))
      elif node_to_type[v] != node_to_type[b] and b not in active_edges[v]:
        active_edges[v].add(b)
        prob_active += Fraction(1, N*G.out_degree(v))

  return total_steps


if __name__ == '__main__':
  from time import time
  import matplotlib.pyplot as plt
  N = 50
  L = nx.star_graph(N)
  R = nx.star_graph(N)
  G: nx.Graph = nx.union(L, R, rename=("L", "R"))
  G.add_edge(f"L0", f"R0")
  # nx.draw(G)
  # plt.show()
  from main import trial_absorption_time
  start = time()
  print(trial_absorption_time(G), time()-start)
  start = time()
  print(simulate_absorption_time(nx.to_directed(G)), time()-start)
  


