import networkx as nx
from dataclasses import dataclass
from typing import *
import numpy as np

def conjoined_star_graph(N: int):
  G = nx.Graph()
  if N == 1:
    G.add_node(0)
    return G

  a = (N-2) // 2
  b = a + (N % 2)

  L = nx.star_graph(a)
  R = nx.star_graph(b)
  G: nx.Graph = nx.union(L, R, rename=("L", "R"))
  G.add_edge("L0", "R0")

  return nx.convert_node_labels_to_integers(G)

def long_conjoined_star_graph(N: int, k: int):
  if k == 0: return conjoined_star_graph(N)
  G = nx.Graph()
  M = N-k
  if M == 1:
    G.add_node(0)
    return G

  a = (M-2) // 2
  b = a + (M % 2)

  L = nx.star_graph(a)
  R = nx.star_graph(b)
  C = nx.path_graph(k)
  G: nx.Graph = nx.union(L, R, rename=("L", "R"))
  G = nx.union(G, C)
  G.add_edge("L0", 0)
  G.add_edge(k-1, "R0")

  return nx.convert_node_labels_to_integers(G)

def complete_with_safe(N: int):
  G: nx.Graph = nx.union(nx.complete_graph(N-1), nx.path_graph(1), rename=("C", "S"))
  G.add_edge("C0", "S0")
  return G


@dataclass
class GraphGenerator:
  build_graph: Callable[[int], nx.Graph]
  name: str

def star_graph(n): return nx.star_graph(n-1)
def complete_bipartite_graph(n): return nx.complete_bipartite_graph(n//2+(n%2), n//2)
def barbell_graph(n):
  if n % 2 == 1: return None
  if n // 2 <= 2: return nx.path_graph(n)
  return nx.barbell_graph(n//2, 0)

def perfect_kary_tree(n, k): return nx.balanced_tree(k, int(h)) if np.isclose(h := np.emath.logn(k, n*(k-1)+1)-1, np.round(h)) else None
def perfect_binary_tree(n): return perfect_kary_tree(n, 2)
def perfect_ternary_tree(n): return perfect_kary_tree(n, 3)
def perfect_quadary_tree(n): return perfect_kary_tree(n, 4)
def perfect_fiveary_tree(n): return perfect_kary_tree(n, 5)
def custom_long_conjoined_star_graph(n):
  return long_conjoined_star_graph(n, k=n//3)

def is_perfect_square(n: int) -> bool: return np.isclose(np.floor(np.sqrt(n))**2, n)
def isqrt(n: int) -> int: return int(np.round(np.sqrt(n)))

def square_periodic_grid(n):
  if not is_perfect_square(n): return None
  return nx.grid_2d_graph(isqrt(n), isqrt(n), periodic=True)

import matplotlib.pyplot as plt
if __name__ == '__main__':
  # G = long_conjoined_star_graph(N=20, k=7)
  G = complete_with_safe(N=10)
  nx.draw(G)
  plt.show()