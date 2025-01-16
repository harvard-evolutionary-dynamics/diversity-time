import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
import matplotlib.cm as cm

from pathlib import Path
from matplotlib.backend_bases import PickEvent
from enum import Enum, auto
from time import time_ns

N = 8
STATS_FILE = Path(f"/Users/david/Dropbox/experiments/diversity/raw-data/bd-vs-db-{N}-v3.csv")

with open(STATS_FILE, "r") as f:
  df = pd.read_csv(f)

palette = sns.color_palette("tab10")

class GraphType(Enum):
  REGULAR = auto()
  Y_EQUALS_X = auto()
  UNCLASSIFIED = auto()


def classify(x):
  G: nx.Graph = nx.graph6.from_graph6_bytes(bytes.fromhex(x).strip())
  entry = df[df['graph6'] == x]
  # print(entry)

  if nx.is_regular(G): return GraphType.REGULAR
  if np.isclose(entry.bd_absorption_time, entry.db_absorption_time): return GraphType.Y_EQUALS_X

  return GraphType.UNCLASSIFIED

df['type'] = df['graph6'].apply(classify)

def onpick(event: PickEvent):
  ind = event.ind[0]
  graph6 = df.iloc[ind]['graph6']
  print(graph6)
  print(event.artist.get_offsets()[event.ind[0]])
  G = nx.graph6.from_graph6_bytes(bytes.fromhex(graph6).strip())
  plt.figure()

  pos = nx.layout.spring_layout(G) # nx.layout.kamada_kawai_layout(G)
  connectionstyle = "arc3,rad=.2"
  # if nx.is_planar(G):
  #   pos = nx.layout.planar_layout(G)
  #   connectionstyle = "arc3,rad=0"

  nx.draw(G, pos=pos, connectionstyle=connectionstyle, arrows=True, node_color='#bcbd22', width=10, node_size=1000)
  plt.savefig(f'plots/a-{time_ns()}.png', transparent=True, dpi=300, bbox_inches='tight')
  # plt.show()



with sns.plotting_context("notebook", font_scale=1.5):
  fig, ax = plt.subplots(figsize=(20,10))
  g = sns.scatterplot(
    data=df,
    ax=ax,
    x="bd_absorption_time",
    y="db_absorption_time",
    s=60,
    alpha=0.5,
    picker=4,
    palette=palette,
    # hue='type',
    legend=False,
  )

  # fig.colorbar(cm.ScalarMappable(cmap=palette), fraction=0.0191, pad=0.04, label='$\Delta/\delta$')
  mi = min(df['bd_absorption_time'].min(), df['db_absorption_time'].min())
  ma = max(df['bd_absorption_time'].max(), df['db_absorption_time'].max())
  PADDING = 10
  g.set(
    xlabel='bd absorption time',
    ylabel='db absorption time',
    aspect='equal',
    adjustable='box',
    xlim=(mi, ma+PADDING),
    ylim=(mi-PADDING, 150),
  )
  # 49, 49
  plo = 20
  phi = 200
  g.axes.plot([plo, phi], [plo, phi], linestyle='--', color='gray')

  nlo = 20
  nhi = 350
  h = lambda x: -0.19751631091802763*(x-49) + 49
  g.axes.plot([nlo, nhi], np.array([h(nlo), h(nhi)]), linestyle='--', color='gray')
  pxs = np.array([plo, phi])
  g.axes.fill_between(pxs, 200+pxs, pxs, alpha=.1)
  nxs = np.array([nlo, nhi])
  g.axes.fill_between(nxs, h(nxs)-200, h(nxs), alpha=.1)
  fig.canvas.mpl_connect("pick_event", onpick)

# plt.savefig(f'plots/bd-vs-db-n8.png', dpi=300, bbox_inches="tight")
plt.show()
