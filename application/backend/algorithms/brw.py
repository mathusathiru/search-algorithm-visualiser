import random
from algorithms.utilities import pos_to_str, DIRECTIONS, manhattan_distance

def generate_biased_random_walk(start, end, grid_size, walls=None, max_steps=None):
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
    
    def get_weighted_moves(current_pos):
        weighted_moves = []
        row, col = current_pos
        
        for dr, dc in DIRECTIONS:
            new_row = row + dr
            new_col = col + dc
            
            if (0 <= new_row < rows and 
                0 <= new_col < cols and 
                not walls[new_row][new_col]):
                
                new_pos = (new_row, new_col)
                visits = visit_counts.get(pos_to_str(new_pos), 0)
                
                dist_to_goal = manhattan_distance(new_pos, end)
                max_dist = rows + cols
                direction_weight = 1 - (dist_to_goal / max_dist)
                visit_weight = 1.0 / (1.0 + visits)
                weight = direction_weight * visit_weight
                
                weighted_moves.append((new_pos, weight))
        
        return weighted_moves if weighted_moves else [(current_pos, 1.0)]
    
    def choose_next_move(moves_with_weights):
        positions = [pos for pos, weight in moves_with_weights]
        weights = [weight for pos, weight in moves_with_weights]
        weight_sum = sum(weights)
        
        if weight_sum == 0:
            return current
            
        probabilities = [w/weight_sum for w in weights]
        return random.choices(positions, probabilities)[0]

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
            
        moves_with_weights = get_weighted_moves(current)
        current = choose_next_move(moves_with_weights)
    
    return steps