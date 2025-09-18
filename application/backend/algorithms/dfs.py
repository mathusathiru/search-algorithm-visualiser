from collections import deque
from algorithms.utilities import pos_to_str, str_to_pos, DIRECTIONS

def generate_dfs(start, end, grid_size, walls=None):
    stack = deque([start])
    steps = []
    visited = []
    visit_counts = {}
    considered = set()
    parent = {pos_to_str(start): None}
    found_path = False
    
    rows = grid_size["rows"]
    cols = grid_size["cols"]
    walls = walls or [[False] * cols for _ in range(rows)]
    
    def get_neighbors(pos):
        row, col = pos
        neighbors = []
        new_considered = []
        
        for delta_row, delta_col in DIRECTIONS:
            new_row, new_col = row + delta_row, col + delta_col
            if 0 <= new_row < rows and 0 <= new_col < cols:
                new_pos = (new_row, new_col)
                new_considered.append(new_pos)
                if not walls[new_row][new_col] and pos_to_str(new_pos) not in parent:
                    neighbors.append(new_pos)
                    
        considered.update(new_considered)
        return neighbors
    
    def reconstruct_path(end_pos):
        path = []
        current = pos_to_str(end_pos)
        while current is not None:
            pos = str_to_pos(current)
            path.append(pos)
            current = parent[current]
        return path[::-1]
    
    while stack and not found_path:
        current = stack[-1]
        current_str = pos_to_str(current)
        
        visited.append(current)
        visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
        
        if current == end:
            found_path = True
            path = reconstruct_path(current)
            steps.append({
                "position": current,
                "visited": visited.copy(),
                "visitCounts": visit_counts.copy(),
                "considered": list(considered),
                "isGoalReached": True,
                "stepNumber": len(steps),
                "path": path
            })
            break
            
        neighbors = get_neighbors(current)
        if neighbors:
            next_node = neighbors[0]
            stack.append(next_node)
            parent[pos_to_str(next_node)] = current_str
        else:
            stack.pop()
        
        steps.append({
            "position": current,
            "visited": visited.copy(),
            "visitCounts": visit_counts.copy(),
            "considered": list(considered),
            "isGoalReached": False,
            "stepNumber": len(steps),
            "path": None
        })
    
    return steps