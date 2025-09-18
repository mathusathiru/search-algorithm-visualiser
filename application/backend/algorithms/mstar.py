import heapq
from algorithms.utilities import manhattan_distance, pos_to_str, DIRECTIONS

class MStarNode:
    def __init__(self, positions, g_score, h_score, conflict_set=None, parent=None):
        self.positions = positions
        self.g_score = g_score
        self.h_score = h_score
        self.f_score = g_score + h_score
        self.conflict_set = conflict_set or set()
        self.parent = parent
    
    def __lt__(self, other):
        return self.f_score < other.f_score
    
    def __eq__(self, other):
        return self.positions == other.positions
    
    def __hash__(self):
        return hash(tuple(tuple(pos) for pos in self.positions))

def find_individual_paths(starts, goals, grid_size, walls, visited=None, considered=None, visit_counts=None):
    if visited is None:
        visited = {i: [] for i in range(len(starts))}
    if considered is None:
        considered = {i: set() for i in range(len(starts))}
    if visit_counts is None:
        visit_counts = {i: {} for i in range(len(starts))}
        
    paths = []
    
    for i, (start, goal) in enumerate(zip(starts, goals)):
        # Simple A* implementation for individual path finding
        open_list = []
        closed_set = set()
        
        heapq.heappush(open_list, (manhattan_distance(start, goal), 0, start, 0))
        
        g_scores = {(start, 0): 0}
        parents = {}
        
        found_path = False
        
        while open_list and not found_path:
            _, _, current, time = heapq.heappop(open_list)
            
            if (current, time) in closed_set:
                continue
                
            closed_set.add((current, time))
            
            # Record visited node - store as list for JavaScript compatibility
            visited[i].append(list(current))
            current_str = pos_to_str(current)
            visit_counts[i][current_str] = visit_counts[i].get(current_str, 0) + 1
            
            if current == goal:
                path = [current]
                curr = (current, time)
                while curr in parents:
                    curr = parents[curr]
                    path.append(curr[0])
                path.reverse()
                paths.append(path)
                found_path = True
                break
            
            row, col = current
            
            for dr, dc in DIRECTIONS:
                new_row, new_col = row + dr, col + dc
                
                if not (0 <= new_row < grid_size["rows"] and 0 <= new_col < grid_size["cols"]) or walls[new_row][new_col]:
                    continue
                    
                new_pos = (new_row, new_col)
                considered[i].add(tuple(new_pos))  # Record considered node as tuple in Python
                new_time = time + 1
                
                tentative_g = g_scores[(current, time)] + 1
                
                if ((new_pos, new_time) not in g_scores or tentative_g < g_scores[(new_pos, new_time)]) and (new_pos, new_time) not in closed_set:
                    g_scores[(new_pos, new_time)] = tentative_g
                    f_score = tentative_g + manhattan_distance(new_pos, goal)
                    heapq.heappush(open_list, (f_score, tentative_g, new_pos, new_time))
                    parents[(new_pos, new_time)] = (current, time)
        
        if not found_path:
            paths.append([start])
            
    return paths, visited, considered, visit_counts

def detect_conflicts(positions, next_positions):
    conflicts = set()
    
    for i in range(len(next_positions)):
        for j in range(i + 1, len(next_positions)):
            if next_positions[i] == next_positions[j]:
                conflicts.add(i)
                conflicts.add(j)
    
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[i] == next_positions[j] and positions[j] == next_positions[i]:
                conflicts.add(i)
                conflicts.add(j)
    
    return conflicts

def generate_mstar(starts, goals, grid_size, walls=None):
    rows, cols = grid_size["rows"], grid_size["cols"]
    walls = walls or [[False] * cols for _ in range(rows)]
    
    visited = {i: [] for i in range(len(starts))}
    considered = {i: set() for i in range(len(starts))}
    visit_counts = {i: {} for i in range(len(starts))}
    
    individual_paths, path_visited, path_considered, path_counts = find_individual_paths(
        starts, goals, grid_size, walls, visited, considered, visit_counts
    )
    
    visited.update(path_visited)
    for i, nodes in path_considered.items():
        considered[i].update(nodes)
    for i, counts in path_counts.items():
        for pos, count in counts.items():
            visit_counts[i][pos] = visit_counts[i].get(pos, 0) + count
    
    start_node = MStarNode(
        positions=starts,
        g_score=0,
        h_score=sum(manhattan_distance(s, g) for s, g in zip(starts, goals)),
        conflict_set=set()
    )
    
    open_list = [start_node]
    closed_set = set()
    
    steps = []
    step_count = 0
    
    while open_list and step_count < 1000:
        current_node = heapq.heappop(open_list)
        
        # Skip if this node has been visited
        positions_tuple = tuple(tuple(pos) for pos in current_node.positions)
        if positions_tuple in closed_set:
            continue
            
        # Mark as visited
        closed_set.add(positions_tuple)
        
        # Record positions and visited cells for each agent
        for i, pos in enumerate(current_node.positions):
            visited[i].append(list(pos))  # Convert to list for JavaScript compatibility
            pos_str = pos_to_str(pos)
            visit_counts[i][pos_str] = visit_counts[i].get(pos_str, 0) + 1
        
        # Check if goal reached
        all_at_goal = all(pos == goal for pos, goal in zip(current_node.positions, goals))
        
        # Generate step data
        paths = {}
        for i, path in enumerate(find_paths_from_node(current_node)):
            paths[i] = path
            
        step_data = {
            "positions": {i: pos for i, pos in enumerate(current_node.positions)},
            "paths": paths,
            "atGoal": {i: pos == goals[i] for i, pos in enumerate(current_node.positions)},
            "visited": {i: list(v) for i, v in visited.items()},
            "considered": {i: [list(pos) for pos in c] for i, c in considered.items()},  # Ensure positions are lists, not tuples
            "visitCounts": visit_counts,
            "stepNumber": step_count,
            "isGoalReached": all_at_goal,
            "conflictSet": list(current_node.conflict_set)
        }
        
        steps.append(step_data)
        step_count += 1
        
        if all_at_goal:
            break
            
        # Generate next nodes based on individual paths or conflict set
        next_node_positions = []
        for i, pos in enumerate(current_node.positions):
            if i in current_node.conflict_set:
                # Generate all possible moves for agents in conflict set
                possible_moves = []
                row, col = pos
                
                # Stay in place is always an option
                possible_moves.append(pos)
                
                # Try all directions
                for dr, dc in DIRECTIONS:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < rows and 0 <= new_col < cols and not walls[new_row][new_col]:
                        new_pos = (new_row, new_col)
                        possible_moves.append(new_pos)
                        considered[i].add(tuple(new_pos))  # Record considered node as tuple in Python
                        
                next_node_positions.append(possible_moves)
            else:
                # Get next position from individual path
                path_index = min(current_node.g_score + 1, len(individual_paths[i]) - 1)
                if path_index < len(individual_paths[i]):
                    next_pos = individual_paths[i][path_index]
                    next_node_positions.append([next_pos])
                else:
                    # Stay at the last position if path ended
                    next_node_positions.append([individual_paths[i][-1]])
        
        # Generate all combinations of next positions
        import itertools
        for next_positions in itertools.product(*next_node_positions):
            
            # Skip if any position is occupied by a wall
            if any(walls[r][c] for r, c in next_positions):
                continue
                
            # Detect conflicts
            conflicts = detect_conflicts(current_node.positions, next_positions)
            
            # Create a new conflict set by merging current conflicts and new ones
            new_conflict_set = current_node.conflict_set.union(conflicts)
            
            # Calculate heuristic
            h_score = sum(manhattan_distance(pos, goal) for pos, goal in zip(next_positions, goals))
            
            # Create new node
            new_node = MStarNode(
                positions=list(next_positions),
                g_score=current_node.g_score + 1,
                h_score=h_score,
                conflict_set=new_conflict_set,
                parent=current_node
            )
            
            # Skip if this node has already been visited
            positions_tuple = tuple(tuple(pos) for pos in new_node.positions)
            if positions_tuple in closed_set:
                continue
                
            heapq.heappush(open_list, new_node)
    
    if not steps or not steps[-1]["isGoalReached"]:
        return [{
            "actionType": "failed",
            "stepNumber": 0,
            "message": "No valid solution found after exploring the search space"
        }]
    
    return steps

def find_paths_from_node(node):
    paths = [[] for _ in range(len(node.positions))]
    
    current = node
    while current:
        for i, pos in enumerate(current.positions):
            paths[i].insert(0, pos)
        current = current.parent
        
    return paths