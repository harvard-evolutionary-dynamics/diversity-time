from absorption import absorption_time, trial_absorption_time
from utils import sample
import numpy as np

import networkx as nx
from typing import Tuple, Dict
from classes import conjoined_star_graph

Center = int
SignedPartition = Tuple[int, ...]
State = SignedPartition

def accel_asc(n):
  "generate integer partitions, source: https://jeromekelleher.net/generating-integer-partitions.html"
  a = [0 for i in range(n + 1)]
  k = 1
  y = n - 1
  while k != 0:
    x = a[k - 1] + 1
    k -= 1
    while 2 * x <= y:
      a[k] = x
      y -= x
      k += 1
    l = k + 1
    while x <= y:
      a[k] = x
      a[l] = y
      yield a[:k + 2]
      x += 1
      y -= 1
    a[k] = x + y
    y = x + y - 1
    yield a[:k + 1]

def signed_partitions(n):
  for partition in accel_asc(n):
    m_partition = list(partition)
    yield m_partition
    for i in range(len(partition)):
      m_partition[i] *= -1
      yield m_partition
      m_partition[i] *= -1

def neg_loc(partition: SignedPartition):
  for i, part in enumerate(partition):
    if part < 0: return i
  return None

def main():
  """
  N = 3
  transitions: Dict[Tuple[State, State], float] = {}
  for partition in signed_partitions(N-1):
    l = len(partition)
    # leaf reproduces to center
    for i in range(l):
      new_partition = list(partition)
      if loc := neg_loc(partition) is not None:
        new_partition[loc] *= -1
      new_partition[i] *= -1

      transitions[(partition, new_partition)] = abs(partition[i]) / (N-1)

      # center reproduces to leaf
      for i in range(l):
        new_partition = list(partition)
        new_partition[i] -= 1
        if new_center < l:
          new_partition[new_center]
        else:
          ...
        transitions[((partition, center), (partition, new_center))] = partition[new_center] / (N-1)
  """



from math import gcd
def H(n, rp=False):
  return sum(1/i * int((not rp) or gcd(i, n)==1) for i in range(1, n+1))

if __name__ == '__main__':
  for N in range(1, 10):
    # G = conjoined_star_graph(N)
    # n = N//2
    G = nx.star_graph(N)
    at = absorption_time(G, SS=(tuple(set(range(N+1))-{1}), (1,)))[0]
    print(f"{N} --> {at-H(N,rp=True)} {H(N,rp=True)}")
    # at = np.mean(list(sample(lambda: trial_absorption_time(G), 100)))
    # print(f"{N} --> {at}, {2*n**2*(n**2-n-1/2) + 2*n*H(n)}")
    # print(f"{N} --> {(N)**2*((N)-1) + (N)*H(N)}")
    # print(f"{N} --> {at}, {at[0]-(N)*H(N)}, {(N)**2*((N)-1) + (N)*H(N)}")