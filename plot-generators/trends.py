import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils import sample
from absorption import trial_absorption_time

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})


def conjoined_star(N):
  G = nx.Graph()
  if N == 1:
    G = G.add_node(1)
    return G

  a = (N-2) // 2
  b = a + (N % 2)

  L = nx.star_graph(a)
  R = nx.star_graph(b)
  G: nx.Graph = nx.union(L, R, rename=("L", "R"))
  G.add_edge("L0", "R0")

  # for x in range(1, a+1):
  #   G.add_edge(-1, -x-1)
  # for x in range(1, b+1):
  #   G.add_edge(+1, x+1)
  # G.add_edge(-1, +1)

  return G

def samples_info(G: nx.Graph, times: int = 100):
  samples = list(sample(lambda: trial_absorption_time(G), times=times))
  mean = np.mean(samples)
  return mean

def draw(N):
  xs = []
  ys = []
  for n in range(3, N+1):
    G = conjoined_star(n)
    abs_time = samples_info(G) # get_exact(G)
    print(n, abs_time)
    xs.append(n)
    ys.append(abs_time)

  ks = []
  for n in xs:
    ks.append(samples_info(nx.complete_graph(n)))

  plt.plot(xs, ks, 'o-', label=r'$K_N$')
  plt.plot(xs, ys, 'o-', label='conjoined stars')
  for d in range(1, 7):
    plt.plot(xs, [x**d for x in xs], '--', label=f'$N^{d}$', alpha=0.3)
  plt.xlabel(r'Number of nodes, $N$')
  plt.ylabel(r'Absorption time, $T$')
  # plt.title('Expected absorption time, neutral drift, simulations')
  plt.xscale('log')
  plt.yscale('log')
  plt.legend()
  plt.show()

if __name__ == '__main__':
  N = 10
  draw(N)