from typing import List, Tuple, Dict, Optional
import heapq
import logging
from algorithms.utilities import pos_to_str, str_to_pos, euclidean_distance

DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

logger = logging.getLogger(__name__)

def generate_gbfs(start: Tuple[int, int], 
                 end: Tuple[int, int], 
                 grid_size: Dict[str, int],
                 walls: Optional[List[List[bool]]] = None,
                 initial_altitudes: Optional[List[List[int]]] = None) -> List[dict]:
        
    rows, cols = grid_size["rows"], grid_size["cols"]
    altitudes = initial_altitudes or [[1] * cols for _ in range(rows)]
    walls = walls or [[False] * cols for _ in range(rows)]
    
    frontier = []
    count = 0
    heapq.heappush(frontier, (euclidean_distance(start, end), count, start))
    
    visited = set()
    start_str = pos_to_str(start)
    parent = {start_str: "null"}
    visit_counts = {}
    considered = set()
    visited_list = []
    
    steps = []
    
    while frontier:
        try:
            _, _, current = heapq.heappop(frontier)

            if current in visited:
                continue
                
            visited.add(current)
            visited_list.append(current)
            current_str = pos_to_str(current)
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
                "visited": visited_list.copy(),
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
                
            row, col = current
            for dr, dc in DIRECTIONS:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < rows and 
                    0 <= new_col < cols and 
                    not walls[new_row][new_col]):
                    
                    new_pos = (new_row, new_col)
                    if new_pos not in visited:
                        considered.add(new_pos)
                        count += 1
                        h = euclidean_distance(new_pos, end)
                        heapq.heappush(frontier, (h, count, new_pos))
                        parent[pos_to_str(new_pos)] = current_str
                        
        except Exception:
            raise
    
    return steps