from algorithms.utilities import pos_to_str, str_to_pos, manhattan_distance
from typing import List, Tuple, Dict, Optional
import heapq

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
        raise IndexError("PriorityQueue is empty")

def get_neighbors(pos: Tuple[int, int], walls: List[List[bool]], rows: int, cols: int) -> List[Tuple[int, int]]:
    x, y = pos
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < rows and 0 <= new_y < cols and not walls[new_x][new_y]:
            neighbors.append((new_x, new_y))
    return neighbors

def jump(current: Tuple[int, int], direction: Tuple[int, int], end: Tuple[int, int],
         walls: List[List[bool]], rows: int, cols: int, depth: int = 0) -> Optional[Tuple[int, int]]:
    x, y = current
    dx, dy = direction
    
    next_x, next_y = x + dx, y + dy
        
    if not (0 <= next_x < rows and 0 <= next_y < cols) or walls[next_x][next_y]:
        return None
        
    next_pos = (next_x, next_y)
    
    if next_pos == end:
        return next_pos
    
    if dx != 0 and dy != 0:
        has_forced = False
        if 0 <= next_x < rows and 0 <= y < cols and walls[next_x][y]:
            has_forced = True
        if 0 <= x < rows and 0 <= next_y < cols and walls[x][next_y]:
            has_forced = True
            
        if has_forced:
            return next_pos
            
        h_jump = jump(next_pos, (dx, 0), end, walls, rows, cols, depth + 1)
        v_jump = jump(next_pos, (0, dy), end, walls, rows, cols, depth + 1)
        
        if h_jump is not None or v_jump is not None:
            return next_pos
            
    else:
        if dx != 0:
            has_forced = False
            if (0 <= next_x < rows and 0 <= next_y + 1 < cols and
                not walls[next_x][next_y + 1] and walls[x][next_y + 1]):
                has_forced = True
            if (0 <= next_x < rows and 0 <= next_y - 1 < cols and
                not walls[next_x][next_y - 1] and walls[x][next_y - 1]):
                has_forced = True
            if has_forced:
                return next_pos
        else:
            has_forced = False
            if (0 <= next_x + 1 < rows and 0 <= next_y < cols and
                not walls[next_x + 1][next_y] and walls[next_x + 1][y]):
                has_forced = True
            if (0 <= next_x - 1 < rows and 0 <= next_y < cols and
                not walls[next_x - 1][next_y] and walls[next_x - 1][y]):
                has_forced = True
            if has_forced:
                return next_pos
    
    return jump(next_pos, direction, end, walls, rows, cols, depth + 1)

def generate_jps(start: Tuple[int, int], end: Tuple[int, int], grid_size: Dict[str, int],
                walls: Optional[List[List[bool]]] = None) -> List[dict]:
    
    rows, cols = grid_size["rows"], grid_size["cols"]
    walls = walls or [[False] * cols for _ in range(rows)]
        
    open_set = PriorityQueue()
    start_str = pos_to_str(start)
    open_set.push(start, manhattan_distance(start, end))
    
    visited = []
    visit_counts = {}
    considered = set()
    g_score = {start_str: 0}
    parent = {start_str: "null"}
    steps = []
    
    def get_path(current):
        path = []
        curr_str = pos_to_str(current)
        while curr_str != "null":
            curr = str_to_pos(curr_str)
            path.append(curr)
            curr_str = parent[curr_str]
        path.reverse()
        return path
    
    def record_step(current):
        current_str = pos_to_str(current)
        if current not in visited:
            visited.append(current)
        visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
        
        path = get_path(current) if current == end else None
        
        step_data = {
            "position": current,
            "visited": visited.copy(),
            "visitCounts": visit_counts.copy(),
            "considered": list(considered),
            "isGoalReached": current == end,
            "stepNumber": len(steps),
            "path": path,
            "parent": parent.copy()
        }
        steps.append(step_data)
    
    record_step(start)
    nodes_expanded = 0
    
    while True:
        try:
            _, current = open_set.pop()
            nodes_expanded += 1
        except IndexError:
            break
            
        current_str = pos_to_str(current)
        
        if current == end:
            record_step(current)
            break
            
        if parent[current_str] == "null":
            neighbors = get_neighbors(current, walls, rows, cols)
        else:
            neighbors = []
            
            for next_dx, next_dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                next_pos = jump(current, (next_dx, next_dy), end, walls, rows, cols)
                if next_pos:
                    neighbors.append(next_pos)
                    considered.add(next_pos)
        
        for neighbor in neighbors:
            neighbor_str = pos_to_str(neighbor)
            tentative_g_score = g_score[current_str] + manhattan_distance(current, neighbor)
            
            if neighbor_str not in g_score or tentative_g_score < g_score[neighbor_str]:
                parent[neighbor_str] = current_str
                g_score[neighbor_str] = tentative_g_score
                f_score = tentative_g_score + manhattan_distance(neighbor, end)
                open_set.push(neighbor, f_score)
                
        record_step(current)

    return steps