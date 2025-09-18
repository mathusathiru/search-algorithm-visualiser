import heapq
from typing import List, Tuple, Dict, Optional
from algorithms.utilities import pos_to_str, str_to_pos, DIRECTIONS

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entries = {}
        
    def push(self, item, priority):
        if item in self.entries:
            self.remove(item)
        entry = [priority, item]
        self.entries[item] = entry
        heapq.heappush(self.heap, entry)
        
    def remove(self, item):
        entry = self.entries.pop(item)
        entry[-1] = None
        
    def pop(self):
        while self.heap:
            priority, item = heapq.heappop(self.heap)
            if item is not None:
                del self.entries[item]
                return priority, item
        raise IndexError

def generate_dijkstra(start: Tuple[int, int], 
                     end: Tuple[int, int], 
                     grid_size: Dict[str, int], 
                     walls: Optional[List[List[bool]]] = None,
                     initial_altitudes: Optional[List[List[int]]] = None) -> List[dict]:
    
    rows, cols = grid_size["rows"], grid_size["cols"]
    altitudes = initial_altitudes if initial_altitudes is not None else [[1] * cols for _ in range(rows)]
    walls = walls if walls is not None else [[False] * cols for _ in range(rows)]
    
    solved = [[False] * cols for _ in range(rows)]
    distances = {}
    parent = {pos_to_str(start): None}
    visit_counts = {}
    visited = []
    
    pq = PriorityQueue()
    pq.push(start, 0)
    distances[pos_to_str(start)] = 0
    steps = []
    
    while True:
        try:
            current_dist, current = pq.pop()
        except IndexError:
            break
            
        row, col = current
        current_str = pos_to_str(current)
        
        if solved[row][col]:
            continue
            
        solved[row][col] = True
        
        visited.append(current)
        visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
        
        path = None
        if current == end:
            path = []
            curr_str = current_str
            while curr_str is not None:
                curr = str_to_pos(curr_str)
                path.append(curr)
                curr_str = parent[curr_str]
            path.reverse()
            
        steps.append({
            "position": current,
            "visited": visited.copy(),
            "visitCounts": visit_counts.copy(),
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
            if (0 <= new_row < rows and 
                0 <= new_col < cols and 
                not walls[new_row][new_col] and 
                not solved[new_row][new_col]):
                
                new_pos = (new_row, new_col)
                new_pos_str = pos_to_str(new_pos)
                new_dist = current_dist + altitudes[new_row][new_col]
                
                if new_pos_str not in distances or new_dist < distances[new_pos_str]:
                    distances[new_pos_str] = new_dist
                    parent[new_pos_str] = current_str
                    pq.push(new_pos, new_dist)
    
    return steps