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