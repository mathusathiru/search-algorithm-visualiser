import random
from algorithms.utilities import pos_to_str, DIRECTIONS

def generate_random_walk(start, end, grid_size, walls=None, max_steps=None):
    if max_steps is None:
        grid_area = grid_size["rows"] * grid_size["cols"]
        max_steps = grid_area * 2

    rows = grid_size["rows"]
    cols = grid_size["cols"]
    walls = walls or [[False] * cols for _ in range(rows)]

    visited = []
    visit_counts = {}
    current = start
    steps = []
    
    def get_moves(current_pos):
        valid_moves = []
        row, col = current_pos
        
        for dr, dc in DIRECTIONS:
            new_row = row + dr
            new_col = col + dc
            
            if (0 <= new_row < rows and 
                0 <= new_col < cols and 
                not walls[new_row][new_col]):
                valid_moves.append((new_row, new_col))
        
        return valid_moves if valid_moves else [current_pos]

    for step in range(max_steps):
        visited.append(current)
        current_str = pos_to_str(current)
        visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
        
        steps.append({
            "position": current,
            "visited": visited.copy(),
            "visitCounts": visit_counts.copy(),
            "isGoalReached": current == end,
            "stepNumber": step
        })
        
        if current == end:
            break
            
        valid_moves = get_moves(current)
        if valid_moves == [current]:
            break
            
        current = random.choice(valid_moves)
    
    return steps