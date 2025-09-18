from collections import deque
from algorithms.utilities import DIRECTIONS, pos_to_str

class MDD:
    def __init__(self, agent_id, start, goal, level, grid_size, walls, visited=None, considered=None, visit_counts=None):
        self.agent_id = agent_id
        self.start = start
        self.goal = goal
        self.level = level
        self.grid_size = grid_size
        self.walls = walls
        
        self.visited = visited if visited is not None else []
        self.considered = considered if considered is not None else set()
        self.visit_counts = visit_counts if visit_counts is not None else {}
        
        self.nodes = set()
        self.edges = set()
        
        self._build_mdd()
    
    def _build_mdd(self):
        rows, cols = self.grid_size["rows"], self.grid_size["cols"]
        
        queue = deque([(self.start, 0)])
        visited_states = set([(self.start, 0)])
        
        self.visited.append(self.start)
        start_str = pos_to_str(self.start)
        self.visit_counts[start_str] = self.visit_counts.get(start_str, 0) + 1
        
        parents = {}
        
        while queue:
            (current_pos, time) = queue.popleft()
            
            if current_pos == self.goal and time == self.level:
                self.nodes.add((current_pos, time))
                continue
            
            if current_pos == self.goal and time < self.level:
                next_state = (current_pos, time + 1)
                if next_state not in visited_states:
                    queue.append(next_state)
                    visited_states.add(next_state)
                    parents[next_state] = (current_pos, time)
                
            if time < self.level:
                row, col = current_pos
                
                for dr, dc in DIRECTIONS:
                    new_row, new_col = row + dr, col + dc
                    
                    if (0 <= new_row < rows and 
                        0 <= new_col < cols and 
                        not self.walls[new_row][new_col]):
                        
                        new_pos = (new_row, new_col)
                        
                        self.considered.add(new_pos)
                        
                        next_state = (new_pos, time + 1)
                        
                        if next_state not in visited_states:
                            self.visited.append(new_pos)
                            new_pos_str = pos_to_str(new_pos)
                            self.visit_counts[new_pos_str] = self.visit_counts.get(new_pos_str, 0) + 1
                            
                            queue.append(next_state)
                            visited_states.add(next_state)
                            parents[next_state] = (current_pos, time)
        
        for node in visited_states:
            pos, time = node
            if time == self.level and pos == self.goal:
                current = node
                self.nodes.add(current)
                
                while current in parents:
                    parent = parents[current]
                    self.nodes.add(parent)
                    self.edges.add((parent, current))
                    current = parent
    
    def get_states_at_time(self, time):
        return [pos for pos, t in self.nodes if t == time]
    
    def get_transitions_at_time(self, time):
        return [(from_pos, to_pos) for (from_pos, from_time), (to_pos, to_time) in self.edges 
                if from_time == time and to_time == time + 1]

def has_conflict(path1, path2):
    min_len = min(len(path1), len(path2))
    
    for t in range(min_len):
        if path1[t] == path2[t]:
            return True
    
    for t in range(min_len - 1):
        if path1[t] == path2[t+1] and path1[t+1] == path2[t]:
            return True
    
    return False

def is_path_valid(paths):
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if has_conflict(paths[i], paths[j]):
                return False
    return True

def extract_paths_from_joint_solution(joint_solution, path_lengths):
    paths = []
    for agent_id, path_length in enumerate(path_lengths):
        path = []
        for t in range(path_length + 1):
            path.append(joint_solution[agent_id][t])
        paths.append(path)
    return paths

def search_joint_mdd(mdds, agent_ids, visited, considered, visit_counts):
    if not mdds:
        return None
    
    if len(mdds) == 1:
        mdd = mdds[agent_ids[0]]
        path = []
        for t in range(mdd.level + 1):
            states = mdd.get_states_at_time(t)
            if states:
                path.append(states[0])
        return {agent_ids[0]: path}
    
    joint_solution = {}
    for agent_id in agent_ids:
        joint_solution[agent_id] = []
    
    for agent_id in agent_ids:
        mdd = mdds[agent_id]
        visited[agent_id].extend(mdd.visited)
        considered[agent_id].update(mdd.considered)
        
        for pos_str, count in mdd.visit_counts.items():
            visit_counts[agent_id][pos_str] = visit_counts[agent_id].get(pos_str, 0) + count
    
    def dfs(time):
        if time > max(mdds[agent_id].level for agent_id in agent_ids):
            return True
        
        valid_positions = {}
        for agent_id in agent_ids:
            if time <= mdds[agent_id].level:
                valid_positions[agent_id] = mdds[agent_id].get_states_at_time(time)
            else:
                valid_positions[agent_id] = [mdds[agent_id].goal]
        
        positions_to_try = [{}]
        for agent_id in agent_ids:
            new_positions = []
            for pos in valid_positions[agent_id]:
                for existing in positions_to_try:
                    new_dict = existing.copy()
                    new_dict[agent_id] = pos
                    new_positions.append(new_dict)
            positions_to_try = new_positions
        
        for position_combo in positions_to_try:
            positions = list(position_combo.values())
            if len(positions) != len(set(positions)):
                continue  
            
            if time > 0:
                has_edge_conflict = False
                for i, agent_id_i in enumerate(agent_ids):
                    for j, agent_id_j in enumerate(agent_ids):
                        if i < j:
                            if (position_combo[agent_id_i] == joint_solution[agent_id_j][time-1] and 
                                position_combo[agent_id_j] == joint_solution[agent_id_i][time-1]):
                                has_edge_conflict = True
                                break
                if has_edge_conflict:
                    continue
            
            for agent_id, pos in position_combo.items():
                if pos not in visited[agent_id]:
                    considered[agent_id].add(pos)
            
            for agent_id, pos in position_combo.items():
                if time < len(joint_solution[agent_id]):
                    joint_solution[agent_id][time] = pos
                else:
                    joint_solution[agent_id].append(pos)
            
            if dfs(time + 1):
                return True
            
            if time < len(joint_solution[agent_id]):
                for agent_id in agent_ids:
                    joint_solution[agent_id][time] = None
        
        return False
    
    if dfs(0):
        return joint_solution
    
    return None

def generate_icts(starts, goals, grid_size, walls=None):
    rows, cols = grid_size["rows"], grid_size["cols"]
    walls = walls or [[False] * cols for _ in range(rows)]
    
    num_agents = len(starts)
    agent_ids = list(range(num_agents))
    
    visited = {i: [] for i in range(num_agents)}
    considered = {i: set() for i in range(num_agents)}
    visit_counts = {i: {} for i in range(num_agents)}
    
    optimal_paths = []
    optimal_costs = []
    
    for i, (start, goal) in enumerate(zip(starts, goals)):
        queue = deque([(start, 0)])
        path_visited = {start}
        
        visited[i].append(start)
        start_str = pos_to_str(start)
        visit_counts[i][start_str] = 1
        
        found_path = False
        path_length = 0
        parent = {start: None}
        
        while queue and not found_path:
            pos, steps = queue.popleft()
            
            if pos == goal:
                path_length = steps
                found_path = True
                
                path = []
                current = pos
                while current is not None:
                    path.append(current)
                    current = parent.get(current)
                path.reverse()
                optimal_paths.append(path)
                break
                
            row, col = pos
            for dr, dc in DIRECTIONS:
                new_row, new_col = row + dr, col + dc
                new_pos = (new_row, new_col)
                
                if (0 <= new_row < rows and 
                    0 <= new_col < cols and 
                    not walls[new_row][new_col] and
                    new_pos not in path_visited):
                    
                    considered[i].add(new_pos)
                    
                    visited[i].append(new_pos)
                    new_pos_str = pos_to_str(new_pos)
                    visit_counts[i][new_pos_str] = visit_counts[i].get(new_pos_str, 0) + 1
                    
                    queue.append((new_pos, steps + 1))
                    path_visited.add(new_pos)
                    parent[new_pos] = pos
        
        if not found_path:
            return [{
                "actionType": "failed",
                "stepNumber": 0,
                "message": f"No valid path found for agent {i}"
            }]
        
        optimal_costs.append(path_length)
    
    ict_root = tuple(optimal_costs)
    
    ict_queue = deque([ict_root])
    ict_visited = {ict_root}
    
    high_level_steps = []
    visited_mdds = {}
    
    ict_node_visited = {agent: [] for agent in range(num_agents)}
    ict_node_considered = {agent: set() for agent in range(num_agents)}
    
    while ict_queue:
        current_costs = ict_queue.popleft()
        
        high_level_step = {
            "actionType": "high-level-search",
            "nodeId": len(high_level_steps),
            "costs": current_costs,
            "totalCost": sum(current_costs)
        }
        high_level_steps.append(high_level_step)
        
        mdds = {}
        for i, (start, goal, cost) in enumerate(zip(starts, goals, current_costs)):
            if (i, cost) not in visited_mdds:
                mdd = MDD(i, start, goal, cost, grid_size, walls, 
                         visited=visited[i].copy(), 
                         considered=considered[i].copy(), 
                         visit_counts=visit_counts[i].copy())
                visited_mdds[(i, cost)] = mdd
                
                visited[i].extend(mdd.visited)
                considered[i].update(mdd.considered)
                for pos_str, count in mdd.visit_counts.items():
                    visit_counts[i][pos_str] = visit_counts[i].get(pos_str, 0) + count
                
            mdds[i] = visited_mdds[(i, cost)]
        
        joint_solution = search_joint_mdd(mdds, agent_ids, visited, considered, visit_counts)
        
        if joint_solution:
            paths = extract_paths_from_joint_solution(joint_solution, current_costs)
            
            max_path_length = max(len(path) for path in paths)
            visualization_steps = []
            
            for t in range(max_path_length):
                positions = {}
                at_goal = {}
                paths_so_far = {}
                
                for agent_id, path in enumerate(paths):
                    pos_idx = min(t, len(path) - 1)
                    positions[agent_id] = path[pos_idx]
                    
                    paths_so_far[agent_id] = path[:pos_idx + 1]
                    
                    at_goal[agent_id] = positions[agent_id] == goals[agent_id]
                
                step = {
                    "positions": positions,
                    "paths": paths_so_far,
                    "atGoal": at_goal,
                    "visited": {i: [list(pos) for pos in visited[i]] for i in range(num_agents)},
                    "considered": {i: [list(pos) for pos in considered[i]] for i in range(num_agents)},
                    "visitCounts": {i: dict(visit_counts[i]) for i in range(num_agents)},
                    "stepNumber": t,
                    "isGoalReached": all(at_goal.values()),
                    "ictsInfo": {
                        "finalCosts": current_costs,
                        "highLevelSteps": len(high_level_steps)
                    }
                }
                
                visualization_steps.append(step)
            
            return visualization_steps
        
        for i in range(num_agents):
            new_costs = list(current_costs)
            new_costs[i] += 1
            new_costs_tuple = tuple(new_costs)
            
            if new_costs_tuple not in ict_visited:
                ict_queue.append(new_costs_tuple)
                ict_visited.add(new_costs_tuple)
    
    return [{
        "actionType": "failed",
        "stepNumber": 0,
        "message": "No valid solution found in the Increasing Cost Tree"
    }]