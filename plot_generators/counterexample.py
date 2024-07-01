import matplotlib.pyplot as plt
import networkx as nx

from absorption import get_exact

def search(init_G: nx.Graph, recursive: bool = False):
  conjecture_pts, conjecture_total = 0, 0
  graphs = [init_G]
  while graphs:
    new_graphs = []
    seen = set()
    for G in graphs:
      abs_time_G = get_exact(G)
      for u, v in G.edges():
        H = G.copy()
        H.remove_edge(u, v)
        if not nx.is_connected(H): continue
        abs_time_H = get_exact(H)
        if abs_time_H in seen: continue
        seen.add(abs_time_H)
        if not (abs_time_H >= abs_time_G):
          plt.subplot(1, 2, 1)
          nx.draw(G)
          plt.subplot(1, 2, 2)
          nx.draw(H)
          print(abs_time_G, abs_time_H)
          plt.show()
        else:
          conjecture_pts += 1
        conjecture_total += 1
        if recursive:
          new_graphs.append(H)
    graphs = new_graphs
  
  print(conjecture_pts / conjecture_total)

if __name__ == '__main__':
  N = 4
  G: nx.Graph = nx.complete_graph(N)
  search(G, recursive=True)