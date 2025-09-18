from typing import List, Tuple, Dict, Set, Optional
from collections import deque

class Agent:
    def __init__(self, id, start, goal):
        self.id = id
        self.start = start
        self.goal = goal
        self.current = start
        self.at_goal = start == goal

def create_graph(grid_size, walls):
    rows, cols = grid_size["rows"], grid_size["cols"]
    graph = {}
    
    for r in range(rows):
        for c in range(cols):
            if walls[r][c]:
                continue
            
            node = (r, c)
            graph[node] = []
            
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not walls[nr][nc]:
                    graph[node].append((nr, nc))
    
    return graph

def decompose_graph(graph):
    subproblems = []
    visited = set()
    
    for node in graph:
        if node in visited:
            continue
            
        if len(graph[node]) >= 3:
            subproblem = {
                'id': len(subproblems),
                'nodes': [node],
                'degree3_nodes': [node]
            }
            
            queue = deque([node])
            subproblem_visited = {node}
            
            while queue:
                current = queue.popleft()
                
                for neighbor in graph[current]:
                    if neighbor not in subproblem_visited:
                        subproblem['nodes'].append(neighbor)
                        subproblem_visited.add(neighbor)
                        
                        if len(graph[neighbor]) >= 3:
                            subproblem['degree3_nodes'].append(neighbor)
                            
                        queue.append(neighbor)
            
            subproblems.append(subproblem)
            visited.update(subproblem_visited)
    
    for node in graph:
        if node not in visited:
            subproblems.append({
                'id': len(subproblems),
                'nodes': [node],
                'degree3_nodes': []
            })
            visited.add(node)
    
    return subproblems

def assign_agents_to_subproblems(agents, subproblems, graph):
    for agent in agents:
        for subproblem in subproblems:
            if agent.current in subproblem['nodes'] or agent.goal in subproblem['nodes']:
                agent.subproblem = subproblem['id']
                break
        else:
            agent.subproblem = 0

def determine_priority(agents, subproblems):
    return sorted([a.id for a in agents], key=lambda agent_id: 
                 next(a.subproblem for a in agents if a.id == agent_id))

def find_path(start, goal, graph, walls, avoid_agents=None):
    if start == goal:
        return [start]
    
    avoid_positions = set()
    if avoid_agents:
        avoid_positions = {a.goal for a in avoid_agents if a.at_goal}
        
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        for neighbor in graph[current]:
            if neighbor == goal:
                return path + [neighbor]
                
            if neighbor not in visited and neighbor not in avoid_positions:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None

def get_agent_at_position(agents, position):
    for agent in agents:
        if agent.current == position:
            return agent
    return None

def move_agent(agent, position, agents, steps, step_number):
    old_pos = agent.current
    agent.current = position
    agent.at_goal = position == agent.goal
    
    steps.append({
        "positions": {a.id: a.current for a in agents},
        "paths": {a.id: [a.current] for a in agents},
        "atGoal": {a.id: a.current == a.goal for a in agents},
        "stepNumber": step_number,
        "isGoalReached": all(a.current == a.goal for a in agents),
        "operation": "move",
        "movedAgent": agent.id,
        "from": old_pos,
        "to": position
    })

def try_push(agent, blocking_agent, agents, graph, walls, steps, step_number):
    for neighbor in graph[blocking_agent.current]:
        if not get_agent_at_position(agents, neighbor):
            old_pos = blocking_agent.current
            blocking_agent.current = neighbor
            blocking_agent.at_goal = neighbor == blocking_agent.goal
            
            steps.append({
                "positions": {a.id: a.current for a in agents},
                "paths": {a.id: [a.current] for a in agents},
                "atGoal": {a.id: a.current == a.goal for a in agents},
                "stepNumber": step_number,
                "isGoalReached": all(a.current == a.goal for a in agents),
                "operation": "push",
                "movedAgent": blocking_agent.id,
                "from": old_pos,
                "to": neighbor
            })
            
            return True
    
    return False

def try_swap(agent, blocking_agent, agents, graph, walls, steps, step_number):
    for node in graph:
        if len(graph[node]) >= 3:
            path1 = find_path(agent.current, node, graph, walls)
            path2 = find_path(blocking_agent.current, node, graph, walls)
            
            if path1 and path2:
                for p in path1[1:]:
                    move_agent(agent, p, agents, steps, step_number)
                    step_number += 1
                
                for p in path2[1:]:
                    move_agent(blocking_agent, p, agents, steps, step_number)
                    step_number += 1
                
                temp = agent.current
                agent.current = blocking_agent.current
                blocking_agent.current = temp
                
                agent.at_goal = agent.current == agent.goal
                blocking_agent.at_goal = blocking_agent.current == blocking_agent.goal
                
                steps.append({
                    "positions": {a.id: a.current for a in agents},
                    "paths": {a.id: [a.current] for a in agents},
                    "atGoal": {a.id: a.current == a.goal for a in agents},
                    "stepNumber": step_number,
                    "isGoalReached": all(a.current == a.goal for a in agents),
                    "operation": "swap",
                    "swappedAgents": [agent.id, blocking_agent.id],
                    "swapVertex": node
                })
                
                return True
    
    return False

def detect_cycle(agent, blocking_agent, agents, path, graph):
    visited = set()
    cycle_path = []
    
    def dfs(node, parent=None):
        visited.add(node)
        cycle_path.append(node)
        
        for neighbor in graph[node]:
            if neighbor == blocking_agent.current:
                cycle_path.append(neighbor)
                return True
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
        
        cycle_path.pop()
        return False
    
    return dfs(agent.current)

def try_rotate(agent, blocking_agent, agents, graph, walls, path, steps, step_number):
    cycle_path = detect_cycle(agent, blocking_agent, agents, path, graph)
    if not cycle_path:
        return False
    
    for i in range(len(cycle_path) - 1):
        current_agent = get_agent_at_position(agents, cycle_path[i])
        next_pos = cycle_path[i + 1]
        
        move_agent(current_agent, next_pos, agents, steps, step_number)
        step_number += 1
    
    return True

def resolve_displaced_agents(agents, graph, walls, steps, step_number, displaced_agents):
    while displaced_agents:
        displaced_list = list(displaced_agents)
        progress_made = False
        
        for agent_id in displaced_list:
            agent = next(a for a in agents if a.id == agent_id)
            
            if agent.current == agent.goal:
                agent.at_goal = True
                displaced_agents.remove(agent_id)
                progress_made = True
                continue
                
            occupant = get_agent_at_position(agents, agent.goal)
            if not occupant:
                path = find_path(agent.current, agent.goal, graph, walls)
                if path:
                    for pos in path[1:]:
                        move_agent(agent, pos, agents, steps, step_number)
                        step_number += 1
                    
                    agent.at_goal = True
                    displaced_agents.remove(agent_id)
                    progress_made = True
            else:
                if try_push(agent, occupant, agents, graph, walls, steps, step_number):
                    path = find_path(agent.current, agent.goal, graph, walls)
                    if path:
                        for pos in path[1:]:
                            move_agent(agent, pos, agents, steps, step_number)
                            step_number += 1
                        
                        agent.at_goal = True
                        displaced_agents.remove(agent_id)
                        progress_made = True
                
        if not progress_made and displaced_agents:
            break

def final_resolve_agents(agents, graph, walls, steps, step_number):
    for agent in agents:
        if agent.current != agent.goal:
            path = find_path(agent.current, agent.goal, graph, walls)
            if path:
                for pos in path[1:]:
                    occupant = get_agent_at_position(agents, pos)
                    if occupant:
                        if try_push(agent, occupant, agents, graph, walls, steps, step_number):
                            move_agent(agent, pos, agents, steps, step_number)
                            step_number += 1
                        else:
                            if try_swap(agent, occupant, agents, graph, walls, steps, step_number):
                                step_number += 1
                            else:
                                break
                    else:
                        move_agent(agent, pos, agents, steps, step_number)
                        step_number += 1

def record_step(steps, agents, step_number, subproblems=None, is_final=False):
    step = {
        "positions": {a.id: a.current for a in agents},
        "paths": {a.id: [a.current] for a in agents},
        "atGoal": {a.id: a.current == a.goal for a in agents},
        "stepNumber": step_number,
        "isGoalReached": is_final and all(a.current == a.goal for a in agents)
    }
    
    if subproblems:
        step["subproblems"] = [
            {
                "id": s['id'],
                "nodes": s['nodes']
            } for s in subproblems
        ]
    
    steps.append(step)

def generate_push_and_rotate(starts, goals, grid_size, walls=None):
    rows, cols = grid_size["rows"], grid_size["cols"]
    walls = walls or [[False] * cols for _ in range(rows)]
    
    graph = create_graph(grid_size, walls)
    
    agents = [Agent(i, start, goal) for i, (start, goal) in enumerate(zip(starts, goals))]
    
    steps = []
    
    subproblems = decompose_graph(graph)
    
    assign_agents_to_subproblems(agents, subproblems, graph)
    
    priority_agents = determine_priority(agents, subproblems)
    
    displaced_agents = set()
    
    record_step(steps, agents, 0, subproblems=subproblems)
    
    for agent_id in priority_agents:
        agent = next(a for a in agents if a.id == agent_id)
        
        if agent.at_goal:
            continue
            
        path = find_path(agent.current, agent.goal, graph, walls)
        
        if not path:
            return [{
                "actionType": "failed",
                "stepNumber": len(steps),
                "message": f"No valid path found for agent {agent_id}"
            }]
            
        for next_pos in path[1:]: 
            occupant = get_agent_at_position(agents, next_pos)
            
            if occupant is None:
                move_agent(agent, next_pos, agents, steps, len(steps))
            else:
                if occupant.at_goal:
                    displaced_agents.add(occupant.id)
                
                if try_push(agent, occupant, agents, graph, walls, steps, len(steps)):
                    move_agent(agent, next_pos, agents, steps, len(steps))
                else:
                    if try_swap(agent, occupant, agents, graph, walls, steps, len(steps)):
                        pass
                    else:
                        if try_rotate(agent, occupant, agents, graph, walls, path, steps, len(steps)):
                            pass
                        else:
                            return [{
                                "actionType": "failed",
                                "stepNumber": len(steps),
                                "message": f"No valid rotation found for agent {agent_id}"
                            }]
        
        agent.at_goal = True
        
        resolve_displaced_agents(agents, graph, walls, steps, len(steps), displaced_agents)

    final_resolve_agents(agents, graph, walls, steps, len(steps))
    
    record_step(steps, agents, len(steps), is_final=True)
    
    return steps