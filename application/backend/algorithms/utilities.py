from typing import Tuple

def pos_to_str(pos: Tuple[int, int]) -> str:
    return str(list(pos))

def str_to_pos(pos_str: str) -> Tuple[int, int]:
    pos_list = eval(pos_str)
    return tuple(pos_list)

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]