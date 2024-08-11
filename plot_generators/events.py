import csv
from pathlib import Path
from absorption import rec_sort, partitions, birth_event

if __name__ == '__main__':
  N = 7
  S_to_idx = {rec_sort(S): idx for idx, S in enumerate(partitions(tuple(range(N))))}
  with Path(f'data/events-{N}.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['S_idx', 'u', 'v', 'T_idx'])
    writer.writeheader()
    for S, idx in S_to_idx.items():
      for u in range(N):
        for v in range(N):
          T = birth_event(S, u, v)
          writer.writerow({'S_idx': idx, 'u': u, 'v': v, 'T_idx': S_to_idx[T]})