import matplotlib.pyplot as plt
import networkx as nx
import random
import seaborn as sns

from typing import Set, Optional

import networkx as nx
from typing import Callable, Iterable, Generator, TypeVar
from pathlib import Path

from networkx.readwrite.graph6 import data_to_n

sns.set_theme(font_scale=2, rc={'text.usetex' : True})
sns.set_style("whitegrid", {'axes.grid' : False})

def is_undirected(G: nx.DiGraph):
  return all((v, u) in G.edges() for (u, v) in G.edges())

def is_oriented(G: nx.DiGraph):
  return all(
    not (
      (u, v) in G.edges() and (v, u) in G.edges()
    )
    for (u, v) in G.edges()
  )


def sample(fn, times):
  count = 0
  while count < times:
    if (ans := fn()) is not None:
      yield ans
      count += 1

def trial_cftime(G: nx.DiGraph, S: Optional[Set], r: float):
  if S is None:
    S = {random.choice(list(G.nodes()))}

  N = len(G)
  V = G.nodes()
  mutants = set()
  mutants |= S
  steps = 0

  while V - mutants:
    if not mutants: return None
    k = len(mutants)
    if random.random() < r*k/(N + (r-1)*k):
      birther = random.choice(list(mutants))
    else:
      birther = random.choice(list(V - mutants))

    dier = random.choice([w for (_, w) in G.out_edges(birther)])
    assert birther != dier
    if birther in mutants:
      mutants.add(dier)
    elif dier in mutants:
      mutants.remove(dier)
    
    steps += 1
  return steps

def style(plot):
  fig = plt.gcf()
  # fig.patch.set_alpha(0)
  # Add a border around the plot
  # ax = plt.gca()
  for i, ax in enumerate(fig.get_axes()):
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Customize the border color and thickness
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.grid(False)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, [['$r = 1.1$', '$r = 2$'][i]], title='')

  # Remove legend title.
  # handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles=handles[1:], labels=labels[1:])


_T = TypeVar("_T")
_Predicate = Callable[[_T], object]

def takewhile_inclusive(predicate: _Predicate[_T], it: Iterable[_T]) -> Generator[_T, None, None]:
  for x in it:
    yield x
    if not predicate(x): break

def digraph6_to_data(string):
  """Convert digraph6 character sequence to 6-bit integers."""
  v = [ord(c)-63 for c in string]
  if len(v) > 0 and (min(v) < 0 or max(v) > 63):
    return None
  return v

# def data_to_n(data):
#   """Read initial one-, four- or eight-unit value from graph6
#   integer sequence.
# 
#   Return (value, rest of seq.)"""
#   if data[0] <= 62:
#     return data[0], data[1:]
#   if data[1] <= 62:
#     return (data[1]<<12) + (data[2]<<6) + data[3], data[4:]
#   return ((data[2]<<30) + (data[3]<<24) + (data[4]<<18) +
#         (data[5]<<12) + (data[6]<<6) + data[7], data[8:])

def parse_digraph6(string):
  """Read a simple directed graph in digraph6 format from string.

  Parameters
  ----------
  string : string
    Data in digraph6 format

  Returns
  -------
  G : Graph

  Raises
  ------
  NetworkXError
      If the string is unable to be parsed in digraph6 format

  References
  ----------
  Graph6 specification:
  http://cs.anu.edu.au/~bdm/data/formats.txt for details.
  """
  def bits():
      """Return sequence of individual bits from 6-bit-per-value
      list of data values."""
      for d in data:
          for i in [5,4,3,2,1,0]:
              yield (d>>i)&1

  HEADER = '>>digraph6<<'
  if string.startswith(HEADER):
    string = string[len(HEADER):]
  
  assert string[0] == "&", string
  string = string[1:]
  data = digraph6_to_data(string)
  n, data = data_to_n(data)
  nd = (n**2 + 5) // 6
  if len(data) != nd:
    raise nx.NetworkXError(\
      'Expected %d bits but got %d in digraph6' % (n**2, len(data)*6))

  G=nx.DiGraph()
  G.add_nodes_from(range(n))
  for (i,j),b in zip([(i,j) for i in range(n) for j in range(n)], bits()):
    if b:
      G.add_edge(i,j)

  return G


def yield_all_digraph6(path: Path):
    """Read simple directed graphs in digraph6 format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.

    Returns
    -------
    G : generator of nx.DiGraphs

    Raises
    ------
    NetworkXError
        If the string is unable to be parsed in digraph6 format

    References
    ----------
    Digraph6 specification:
    http://cs.anu.edu.au/~bdm/data/formats.txt for details.
    """
    with path.open() as f:
      for line in f.readlines():
        line = line.strip()
        if not len(line):
          continue
        yield parse_digraph6(line)

def networkx_to_pepa_format(G: nx.Graph):
  N, M = G.number_of_nodes(), G.number_of_edges()
  return [(N, M), *G.edges()]

def yield_all_graph6(path: Path):
  with path.open(mode="rb") as f:
    for line in f.readlines():
      line = line.strip()
      if not len(line):
        continue
      yield nx.from_graph6_bytes(line)