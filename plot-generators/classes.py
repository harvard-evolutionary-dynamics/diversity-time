import networkx as nx

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
  G = nx.union(nx.complete_graph(N-1), nx.path_graph(1), rename=("C", "S"))
  G.add_edge("C0", "S0")
  return G

import matplotlib.pyplot as plt
if __name__ == '__main__':
  # G = long_conjoined_star_graph(N=20, k=7)
  G = complete_with_safe(N=10)
  nx.draw(G)
  plt.show()