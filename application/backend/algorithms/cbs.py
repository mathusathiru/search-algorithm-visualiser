from typing import List, Tuple, Dict, Set, Optional
import heapq
from algorithms.utilities import manhattan_distance, pos_to_str, str_to_pos, DIRECTIONS

class ConstraintTreeNode:
    """Node in the Constraint Tree used by CBS algorithm."""
    def __init__(self):
        self.constraints = {}
        self.solution = {} 
        self.cost = 0
    
    def __lt__(self, other):
        return self.cost < other.cost

def find_path_with_constraints(agent_id, start, goal, grid_size, walls, constraints, other_agents_start_goal=None):
    """A* search respecting time-based vertex and edge constraints plus other agents' start/goal positions."""
    rows, cols = grid_size["rows"], grid_size["cols"]
    
    open_list = []
    closed_set = set()
    
    count = 0
    heapq.heappush(open_list, (manhattan_distance(start, goal), 0, start, 0, count))
    count += 1
    
    came_from = {}
    
    g_score = {(start, 0): 0}
    
    f_score = {(start, 0): manhattan_distance(start, goal)}
    
    visited = []
    visit_counts = {}
    considered = set()
    
    while open_list:
        _, _, current, time, _ = heapq.heappop(open_list)
        current_state = (current, time)
        
        if current_state in closed_set:
            continue
        
        closed_set.add(current_state)
        visited.append(current)
        
        current_str = pos_to_str(current)
        visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
        
        if current == goal:
            path = []
            curr_state = current_state
            while curr_state in came_from:
                path.append(curr_state[0])
                curr_state = came_from[curr_state]
            path.append(start)
            path.reverse()
            return path, visited, visit_counts, considered
        
        row, col = current
        
        neighbors = [(current, time + 1)]
        
        for dr, dc in DIRECTIONS:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)
            
            if not (0 <= new_row < rows and 0 <= new_col < cols) or walls[new_row][new_col]:
                continue
                
            if other_agents_start_goal:
                is_blocked = False
                for other_id, (other_start, other_goal) in other_agents_start_goal.items():
                    if other_id != agent_id:
                        if new_pos == other_goal:
                            is_blocked = True
                            break
                        
                        if new_pos == other_start:
                            is_blocked = True
                            break
                            
                if is_blocked:
                    continue
            
            neighbors.append((new_pos, time + 1))
            considered.add(new_pos)
        
        for next_pos, next_time in neighbors:
            next_state = (next_pos, next_time)
            
            vertex_constraint = (agent_id, next_pos, next_time)
            if vertex_constraint in constraints.get("vertex", []):
                continue
            
            if current != next_pos:
                edge_constraint = (agent_id, current, next_pos, next_time - 1)
                if edge_constraint in constraints.get("edge", []):
                    continue
            
            tentative_g_score = g_score[current_state] + 1
            
            if next_state not in g_score or tentative_g_score < g_score[next_state]:
                came_from[next_state] = current_state
                g_score[next_state] = tentative_g_score
                f_score[next_state] = tentative_g_score + manhattan_distance(next_pos, goal)
                
                if next_state not in closed_set:
                    heapq.heappush(open_list, (f_score[next_state], g_score[next_state], next_pos, next_time, count))
                    count += 1
    
    return None, visited, visit_counts, considered

def detect_conflicts(solution):
    """Find conflicts between agents' paths."""
    conflicts = []
    
    agents = list(solution.keys())
    
    goal_times = {}
    for agent_id, path in solution.items():
        for t, pos in enumerate(path):
            if t < len(path) - 1 and path[t+1:] and all(p == pos for p in path[t+1:]):
                goal_times[agent_id] = t
                break
    
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent1 = agents[i]
            agent2 = agents[j]
            path1 = solution[agent1]
            path2 = solution[agent2]
            
            min_len = min(len(path1), len(path2))
            for t in range(min_len):
                if (agent1 in goal_times and t > goal_times[agent1]) or \
                   (agent2 in goal_times and t > goal_times[agent2]):
                    continue
                
                if path1[t] == path2[t]:
                    conflicts.append({
                        "type": "vertex",
                        "agents": (agent1, agent2),
                        "location": path1[t],
                        "time": t
                    })
                    return conflicts
            
            for t in range(min_len - 1):
                if (agent1 in goal_times and t > goal_times[agent1]) or \
                   (agent2 in goal_times and t > goal_times[agent2]):
                    continue
                
                if path1[t] == path2[t+1] and path1[t+1] == path2[t]:
                    conflicts.append({
                        "type": "edge",
                        "agents": (agent1, agent2),
                        "locations": (path1[t], path1[t+1]),
                        "time": t
                    })
                    return conflicts
    
    return conflicts

def generate_cbs(starts, goals, grid_size, walls=None):
    """Generate CBS solution and visualization steps."""
    rows, cols = grid_size["rows"], grid_size["cols"]
    walls = walls or [[False] * cols for _ in range(rows)]
    
    root = ConstraintTreeNode()
    root.constraints = {"vertex": [], "edge": []}
    
    agents_start_goal = {i: (start, goal) for i, (start, goal) in enumerate(zip(starts, goals))}
    
    all_visited = {}
    all_visit_counts = {}
    all_considered = {}
    
    for i, (start, goal) in enumerate(zip(starts, goals)):
        result = find_path_with_constraints(i, start, goal, grid_size, walls, root.constraints, agents_start_goal)
        
        if result is None or result[0] is None:
            return [{
                "actionType": "failed",
                "stepNumber": 0,
                "message": f"No valid path found for agent {i} in the root node"
            }]
            
        path, visited, visit_counts, considered = result
        root.solution[i] = path
        root.cost += len(path) - 1
        all_visited[i] = visited
        all_visit_counts[i] = visit_counts
        all_considered[i] = considered
    
    conflicts = detect_conflicts(root.solution)
    
    open_list = [root]
    
    steps = []
    constraint_tree_nodes = [root]
    
    high_level_steps = []
    
    while open_list and len(open_list) < 1000:
        node = heapq.heappop(open_list)
        node_idx = constraint_tree_nodes.index(node)
        
        conflicts = detect_conflicts(node.solution)
        
        high_level_step = {
            "actionType": "high-level-search",
            "nodeId": node_idx,
            "conflicts": conflicts,
            "cost": node.cost
        }
        high_level_steps.append(high_level_step)
        
        if not conflicts:
            max_path_length = max(len(path) for path in node.solution.values())
            visualization_steps = []
            
            for t in range(max_path_length):
                positions = {}
                at_goal = {}
                paths = {}
                visited = {}
                considered = {}
                
                for agent_id, path in node.solution.items():
                    pos_idx = min(t, len(path) - 1)
                    positions[agent_id] = path[pos_idx]
                    
                    paths[agent_id] = path[:pos_idx + 1]
                    
                    at_goal[agent_id] = positions[agent_id] == goals[agent_id]
                    
                    visited[agent_id] = all_visited.get(agent_id, [])
                    considered[agent_id] = list(all_considered.get(agent_id, set()))
                
                step = {
                    "positions": positions,
                    "paths": paths,
                    "atGoal": at_goal,
                    "visited": visited,
                    "considered": considered,
                    "visitCounts": {i: dict(vc) for i, vc in all_visit_counts.items()},
                    "stepNumber": t,
                    "isGoalReached": all(at_goal.values())
                }
                
                visualization_steps.append(step)
            
            return visualization_steps
        
        conflict = conflicts[0]
        
        for agent_idx, agent_id in enumerate(conflict["agents"]):
            new_node = ConstraintTreeNode()
            
            new_node.constraints = {
                "vertex": node.constraints.get("vertex", []).copy(),
                "edge": node.constraints.get("edge", []).copy()
            }
            
            if conflict["type"] == "vertex":
                new_node.constraints["vertex"].append((agent_id, conflict["location"], conflict["time"]))
            else: 
                loc1, loc2 = conflict["locations"]
                new_node.constraints["edge"].append((agent_id, loc1, loc2, conflict["time"]))
            
            for a, p in node.solution.items():
                if a != agent_id:
                    new_node.solution[a] = p
            
            result = find_path_with_constraints(
                agent_id, starts[agent_id], goals[agent_id], grid_size, walls, new_node.constraints, agents_start_goal
            )
            
            if result is not None and result[0] is not None:
                path, visited, visit_counts, considered = result
                new_node.solution[agent_id] = path
                new_node.cost = sum(max(1, len(p) - 1) for p in new_node.solution.values())
                all_visited[agent_id] = visited
                all_visit_counts[agent_id] = visit_counts
                all_considered[agent_id] = considered
                
                constraint_tree_nodes.append(new_node)
                heapq.heappush(open_list, new_node)
    
    return [{
        "actionType": "failed",
        "stepNumber": 0,
        "message": "No valid solution found after searching all constraint combinations"
    }]