from typing import *

def num_types_left(S: Dict[int, Set[Any]]):
  return sum(
    int(len(locations) > 0)
    for locations in S.values()
  )