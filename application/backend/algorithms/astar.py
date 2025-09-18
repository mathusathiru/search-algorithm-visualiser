from typing import List, Tuple, Dict, Optional
import heapq
from algorithms.utilities import pos_to_str, str_to_pos, DIRECTIONS

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_count = 0
        self.entries = {}
        
    def push(self, item, priority):
        if item in self.entries:
            self.remove(item)
        entry = [priority, self.entry_count, item]
        self.entries[item] = entry
        heapq.heappush(self.heap, entry)
        self.entry_count += 1
        
    def remove(self, item):
        entry = self.entries.pop(item)
        entry[-1] = None
        
    def pop(self):
        while self.heap:
            priority, _, item = heapq.heappop(self.heap)
            if item is not None:
                del self.entries[item]
                return priority, item
        raise IndexError

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def generate_astar(start: Tuple[int, int], end: Tuple[int, int], grid_size: Dict[str, int], 
                  walls: Optional[List[List[bool]]] = None, initial_altitudes: Optional[List[List[int]]] = None) -> List[dict]:
    
    rows, cols = grid_size["rows"], grid_size["cols"]
    altitudes = initial_altitudes or [[1] * cols for _ in range(rows)]
    walls = walls or [[False] * cols for _ in range(rows)]
    
    start_str = pos_to_str(start)
    g_score = {start_str: 0}
    f_score = {start_str: manhattan_distance(start, end)}
    parent = {start_str: "null"}
    visit_counts = {}
    visited = []
    considered = set()
    
    open_set = PriorityQueue()
    open_set.push(start, f_score[start_str])
    steps = []

    while True:
        try:
            _, current = open_set.pop()
        except IndexError:
            break

        row, col = current
        current_str = pos_to_str(current)
        visited.append(current)
        visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
        
        path = None
        if current == end:
            path = []
            curr_str = current_str
            while curr_str != "null":
                curr = str_to_pos(curr_str)
                path.append(curr)
                curr_str = parent[curr_str]
            path.reverse()
        
        steps.append({
            "position": current,
            "visited": visited.copy(),
            "visitCounts": visit_counts.copy(),
            "considered": list(considered),
            "isGoalReached": current == end,
            "stepNumber": len(steps),
            "path": path,
            "altitudes": altitudes,
            "parent": parent.copy()
        })
        
        if current == end:
            break

        for dr, dc in DIRECTIONS:
            new_row, new_col = row + dr, col + dc
            if not (0 <= new_row < rows and 0 <= new_col < cols) or walls[new_row][new_col]:
                continue
                
            new_pos = (new_row, new_col)
            new_pos_str = pos_to_str(new_pos)
            considered.add(new_pos)
            
            tentative_g_score = g_score[current_str] + altitudes[new_row][new_col]
            
            if new_pos_str not in g_score or tentative_g_score < g_score[new_pos_str]:
                parent[new_pos_str] = current_str
                g_score[new_pos_str] = tentative_g_score
                f_score[new_pos_str] = tentative_g_score + manhattan_distance(new_pos, end)
                open_set.push(new_pos, f_score[new_pos_str])
    
    return steps