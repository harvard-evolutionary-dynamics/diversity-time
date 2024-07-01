from utils import yield_all_graph6
from pathlib import Path
from absorption import get_exact
from classes import conjoined_star_graph

import matplotlib.pyplot as plt
import networkx as nx


def slowest(N, draw_update: bool = False):
  slowest_G, most_steps = None, -1
  for _, G in enumerate(yield_all_graph6(Path(f"data/connected-n{N}.g6"))):
    steps = get_exact(G)
    if steps > most_steps:
      slowest_G, most_steps = G, steps
      if draw_update:
        print(f'slowest steps: {most_steps}')
        nx.draw(slowest_G)
        plt.show()

  print('done!')
  if draw_update:
    print(f'slowest steps: {most_steps}')
    nx.draw(slowest_G)
    plt.show()
  return slowest_G


def draw_slowest(N):
  for n in range(1, N+1):
    slowest(n, draw_update=False)

def draw(N):
  xs = []
  ys = []
  for n in range(1, N+1):
    G = conjoined_star_graph(n) # G = slowest(n) # the slowest for N <= 8 is the conjoined star!
    abs_time = get_exact(G)
    nx.draw(G)
    plt.show()
    print(n, abs_time)
    xs.append(n)
    ys.append(abs_time)

  plt.clf()
  plt.cla()

  plt.plot(xs, ys, 'o-')
  plt.xlabel(r'Number of nodes, $N$')
  plt.ylabel(r'Absorption time, $T$')
  plt.savefig(f'plots/slowest.png', dpi=300, bbox_inches="tight")
  plt.show()

if __name__ == '__main__':
  N = 8
  draw(N)