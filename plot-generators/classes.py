import networkx as nx
from dataclasses import dataclass
from typing import *
import numpy as np
import itertools


def double_leaved_star(n):
  if n % 2 == 0: return None
  num_leaves = (n-1) // 2
  G: nx.Graph = nx.star_graph(num_leaves)
  for i in range(1, num_leaves+1):
    G.add_node(f"outer-{i}")
    G.add_edge(f"outer-{i}", i)

  return G

def triple_leaved_star(n):
  if n % 3 != 1: return None
  num_leaves = (n-1) // 3
  G: nx.Graph = double_leaved_star(n-num_leaves)
  for i in range(1, num_leaves+1):
    G.add_node(f"outer-outer-{i}")
    G.add_edge(f"outer-outer-{i}", f"outer-{i}")

  return G

def random_regular(n, d):
  while not nx.is_connected(G := nx.random_regular_graph(d, n)): ...
  return G

def random_regular_2(n): return random_regular(n, 2)
def random_regular_3(n): return random_regular(n, 3)
def random_regular_4(n): return random_regular(n, 4)
def random_regular_5(n): return random_regular(n, 5)
def random_regular_6(n): return random_regular(n, 6)
def random_regular_7(n): return random_regular(n, 7)
def random_regular_8(n): return random_regular(n, 8)
def random_regular_9(n): return random_regular(n, 9)
def random_regular_10(n): return random_regular(n, 10)
def random_regular_11(n): return random_regular(n, 11)
def random_regular_12(n): return random_regular(n, 12)
def random_regular_13(n): return random_regular(n, 13)
def random_regular_14(n): return random_regular(n, 14)
def random_regular_15(n): return random_regular(n, 15)
def random_regular_16(n): return random_regular(n, 16)
def random_regular_17(n): return random_regular(n, 17)
def random_regular_18(n): return random_regular(n, 18)


def erdos_renyi(n, p):
  assert 0 < p <= 1
  while not nx.is_connected(G := nx.erdos_renyi_graph(n, p)): ...
  return G

def erdos_renyi_threshold(n): return erdos_renyi(n, np.log(n)/n)
def erdos_renyi_10(n): return erdos_renyi(n, .1)
def erdos_renyi_20(n): return erdos_renyi(n, .2)
def erdos_renyi_30(n): return erdos_renyi(n, .3)
def erdos_renyi_40(n): return erdos_renyi(n, .4)
def erdos_renyi_50(n): return erdos_renyi(n, .5)
def erdos_renyi_60(n): return erdos_renyi(n, .6)
def erdos_renyi_70(n): return erdos_renyi(n, .7)
def erdos_renyi_80(n): return erdos_renyi(n, .8)
def erdos_renyi_90(n): return erdos_renyi(n, .9)
def erdos_renyi_100(n): return erdos_renyi(n, 1)

def joined_constant_stars(num_stars: int, num_leaves_in_star: int, join_method: str):
  stars = [nx.star_graph(num_leaves_in_star) for _ in range(num_stars)]  
  G: nx.Graph = nx.Graph()
  for i in range(num_stars):
    G = nx.union(G, stars[i], rename=("", f"G{i}-"))

  if join_method == 'cycle':
    for i in range(num_stars):
      G.add_edge(f"G{i}-0", f"G{(i+1) % num_stars}-0")
  elif join_method == 'complete':
    for i, j in itertools.combinations(range(num_stars), r=2):
      G.add_edge(f"G{i}-0", f"G{j}-0")
  elif join_method == 'star':
    nx.union(G, nx.complete_graph(1), rename=("", "center"))
    for i in range(num_stars):
      G.add_edge(f"G{i}-0", "center")
  else:
    raise NotImplementedError(f"{join_method} is not a supported join method")

  G.remove_edges_from(nx.selfloop_edges(G))
  return G

def joined_stars_given_num_stars(N: int, num_stars: int, join_method: str):
  num_leaves_in_star=(N-int(join_method=='star')) // num_stars - 1
  if num_leaves_in_star < 0: return None
  G = joined_constant_stars(num_stars=num_stars, num_leaves_in_star=num_leaves_in_star, join_method=join_method)
  assert N-num_stars < len(G) <= N, len(G)
  for i in range(num_stars):
    if G.number_of_nodes() == N: break
    G.add_node(f"G{i}-{num_leaves_in_star+1}")
    G.add_edge(f"G{i}-0", f"G{i}-{num_leaves_in_star+1}")

  assert len(G) == N, len(G)
  return G

def joined_stars_given_num_leaves_in_star(N: int, num_leaves_in_star: int, join_method: str):
  num_stars = (N-int(join_method=='star')) // (num_leaves_in_star+1)
  if num_stars < 1: return None
  return joined_stars_given_num_stars(N, num_stars=num_stars, join_method=join_method)

def cyclically_joined_stars_3_stars(N: int): return joined_stars_given_num_stars(N, num_stars=3, join_method='cycle')
def star_joined_stars_3_stars(N: int): return joined_stars_given_num_stars(N, num_stars=3, join_method='star')
def cyclically_joined_stars_sqrt_num_stars(N: int): return joined_stars_given_num_stars(N, num_stars=int(np.sqrt(N)), join_method='cycle')
def completely_joined_stars_sqrt_num_stars(N: int): return joined_stars_given_num_stars(N, num_stars=int(np.sqrt(N)), join_method='complete')
def starly_joined_stars_sqrt_num_stars(N: int): return joined_stars_given_num_stars(N, num_stars=int(np.sqrt(N)), join_method='star')

def cyclically_joined_stars_constant_star_size(N: int, num_leaves_in_star: int): return joined_stars_given_num_leaves_in_star(N, num_leaves_in_star=num_leaves_in_star, join_method='cycle')
def cyclically_joined_5_leaf_stars(N: int): return joined_stars_given_num_leaves_in_star(N, num_leaves_in_star=5, join_method='cycle')
def cyclically_joined_10_leaf_stars(N: int): return joined_stars_given_num_leaves_in_star(N, num_leaves_in_star=10, join_method='cycle')
def cyclically_joined_15_leaf_stars(N: int): return joined_stars_given_num_leaves_in_star(N, num_leaves_in_star=15, join_method='cycle')
def cyclically_joined_20_leaf_stars(N: int): return joined_stars_given_num_leaves_in_star(N, num_leaves_in_star=20, join_method='cycle')
def cyclically_joined_stars_sqrt_star_size(N: int): return joined_stars_given_num_leaves_in_star(N, num_leaves_in_star=int(np.sqrt(N)), join_method='cycle')

def meta_conjoined_star_graph(N: int):
  assert N >= 10
  num_leaves = (N-6) // 4
  stars = [nx.star_graph(num_leaves) for _ in range(4)]
  L = nx.union(stars[0], stars[1], rename=("top-", "bottom-"))
  R = nx.union(stars[2], stars[3], rename=("top-", "bottom-"))
  G: nx.Graph = nx.union(L, R, rename=("L-", "R-"))
  G.add_node("L-center")
  G.add_node("R-center")
  G.add_edge("L-center", "L-top-0")
  G.add_edge("L-center", "L-bottom-0")
  G.add_edge("R-center", "R-top-0")
  G.add_edge("R-center", "R-bottom-0")
  G.add_edge("L-center", "R-center")

  assert len(G) <= N, len(G)
  for star_prefix in ("L-top", "R-top", "L-bottom", "R-bottom"):
    if len(G) == N: break
    G.add_node(f"{star_prefix}-{num_leaves+1}")
    G.add_edge(f"{star_prefix}-{num_leaves+1}", f"{star_prefix}-0")


  return G


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
  G = meta_conjoined_star_graph(25)
  nx.draw(G, nx.spring_layout(G))
  plt.show()