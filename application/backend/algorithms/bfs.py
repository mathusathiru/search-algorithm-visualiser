from collections import deque
from algorithms.utilities import pos_to_str, str_to_pos, DIRECTIONS

class Queue:    
    def __init__(self, initial_items=None):
        self.items = deque(initial_items if initial_items else [])
        
    def enqueue(self, item):
        self.items.append(item)
        
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError
        
    def is_empty(self):
        return not self.items
        
    def __len__(self):
        return len(self.items)

def generate_bfs(start, end, grid_size, walls=None):
    queue = Queue([start])
    steps = []
    visited = []
    visit_counts = {}
    parent = {pos_to_str(start): None}
    found_path = False

    rows = grid_size["rows"]
    cols = grid_size["cols"]
    walls = walls or [[False] * cols for _ in range(rows)]
    
    def get_neighbors(pos):
        row, col = pos
        return [(row + dr, col + dc) for dr, dc in DIRECTIONS
                if 0 <= row + dr < rows 
                and 0 <= col + dc < cols 
                and not walls[row + dr][col + dc]]
    
    def reconstruct_path(end_pos):
        path = []
        current = pos_to_str(end_pos)
        while current is not None:
            pos = str_to_pos(current)
            path.append(pos)
            current = parent[current]
        return path[::-1]
    
    while not queue.is_empty() and not found_path:
        current = queue.dequeue()
        current_str = pos_to_str(current)
        
        visited.append(current)
        visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
        
        if current == end:
            found_path = True
        
        steps.append({
            "position": current,
            "visited": visited.copy(), 
            "visitCounts": visit_counts.copy(),
            "isGoalReached": current == end,
            "stepNumber": len(steps),
            "path": reconstruct_path(current) if found_path else None
        })
            
        if found_path:
            break
            
        for neighbor in get_neighbors(current):
            neighbor_str = pos_to_str(neighbor)
            if neighbor_str not in parent:
                queue.enqueue(neighbor)
                parent[neighbor_str] = current_str
    
    return steps