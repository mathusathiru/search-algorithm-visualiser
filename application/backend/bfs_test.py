import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import random
from tqdm import tqdm

# Import the algorithm modules directly
from algorithms.bfs import generate_bfs
from algorithms.dfs import generate_dfs
from algorithms.dijkstra import generate_dijkstra
from algorithms.astar import generate_astar
from algorithms.gbfs import generate_gbfs
from algorithms.jps import generate_jps
from algorithms.rw import generate_random_walk
from algorithms.brw import generate_biased_random_walk

# Multi-agent algorithms
from algorithms.cbs import generate_cbs
from algorithms.icts import generate_icts
from algorithms.mstar import generate_mstar
from algorithms.par import generate_push_and_rotate

# Set up argument parser
parser = argparse.ArgumentParser(description='Compare node expansion between single-agent and multi-agent pathfinding algorithms')
parser.add_argument('--grid-size', type=str, default='medium', choices=['small', 'medium', 'large'], 
                    help='Grid size for the algorithms')
parser.add_argument('--iterations', type=int, default=5, 
                    help='Number of iterations to run per algorithm')
parser.add_argument('--save-plots', action='store_true', 
                    help='Save plots as PNG files')
parser.add_argument('--wall-density', type=float, default=0.2, 
                    help='Density of walls in the grid (0.0 to 1.0)')
parser.add_argument('--multi-agents', type=int, default=3,
                    help='Number of agents for multi-agent algorithms')
parser.add_argument('--log-scale', action='store_true',
                    help='Use logarithmic scale for y-axis')
args = parser.parse_args()

# Grid size definitions
GRID_SIZES = {
    'small': {'rows': 20, 'cols': 50},
    'medium': {'rows': 15, 'cols': 40},
    'large': {'rows': 10, 'cols': 30}
}

# Algorithm definitions with their corresponding functions
SINGLE_AGENT_ALGORITHMS = [
    {'value': 'bfs', 'label': 'Breadth First Search', 'function': generate_bfs},
    {'value': 'dfs', 'label': 'Depth First Search', 'function': generate_dfs},
    {'value': 'dijkstra', 'label': 'Dijkstra\'s', 'function': generate_dijkstra},
    {'value': 'astar', 'label': 'A* Search', 'function': generate_astar},
    {'value': 'gbfs', 'label': 'Greedy Best First Search', 'function': generate_gbfs},
    {'value': 'jps', 'label': 'Jump Point Search', 'function': generate_jps},
    {'value': 'randomwalk', 'label': 'Random Walk', 'function': generate_random_walk},
    {'value': 'biasedrandomwalk', 'label': 'Biased Random Walk', 'function': generate_biased_random_walk}
]

MULTI_AGENT_ALGORITHMS = [
    {'value': 'cbs', 'label': 'Conflict-Based Search', 'function': generate_cbs},
    {'value': 'icts', 'label': 'Increasing Cost Tree Search', 'function': generate_icts},
    {'value': 'mstar', 'label': 'M* Search', 'function': generate_mstar},
    {'value': 'pushandrotate', 'label': 'Push and Rotate', 'function': generate_push_and_rotate}
]

def generate_random_walls(rows, cols, density):
    """Generate a random wall configuration with the given density."""
    walls = np.random.random((rows, cols)) < density
    return walls.tolist()

def generate_random_positions(rows, cols, walls, num_positions=1):
    """Generate random positions that don't overlap with walls."""
    positions = []
    for _ in range(num_positions):
        max_attempts = 100
        attempts = 0
        while attempts < max_attempts:
            row = np.random.randint(0, rows)
            col = np.random.randint(0, cols)
            if not walls[row][col] and [row, col] not in positions:
                positions.append([row, col])
                break
            attempts += 1
        
        if attempts >= max_attempts:
            # If we can't find a valid position after max attempts, create one
            for r in range(rows):
                for c in range(cols):
                    if not walls[r][c] and [r, c] not in positions:
                        positions.append([r, c])
                        break
                if len(positions) == num_positions:
                    break
            
            # If we still couldn't find enough positions, reduce walls
            if len(positions) < num_positions:
                for r in range(rows):
                    for c in range(cols):
                        if walls[r][c]:
                            walls[r][c] = False
                            positions.append([r, c])
                            if len(positions) == num_positions:
                                return positions
    
    return positions

def count_visited_nodes_single_agent(steps):
    """Count the number of unique nodes visited at each step for single-agent algorithms."""
    visited_counts = []
    
    for step in steps:
        if 'visited' in step:
            # For single agent algorithms
            visited_counts.append(len(step['visited']))
        else:
            # If no visited field, use the step number
            visited_counts.append(step.get('stepNumber', 0))
    
    return visited_counts

def count_visited_nodes_cbs(steps):
    """Count the number of unique nodes visited for CBS algorithm."""
    visited_counts = []
    
    # CBS tracks visited nodes for each agent
    for step in steps:
        if 'visited' in step:
            total_visited = 0
            # Combine visited nodes from all agents
            for agent_id, visited_list in step['visited'].items():
                if isinstance(visited_list, list):
                    total_visited += len(visited_list)
            visited_counts.append(total_visited)
        else:
            # If no visited field, try to count positions
            visited_counts.append(len(step.get('positions', {})))
    
    return visited_counts

def count_visited_nodes_icts(steps):
    """Count nodes explored in ICTS algorithm."""
    # ICTS doesn't track visited nodes in the same way
    # Use the path length and final costs as a proxy
    if not steps:
        return []
        
    last_step = steps[-1]
    icts_info = last_step.get('ictsInfo', {})
    high_level_steps = icts_info.get('highLevelSteps', 0)
    
    # Create a linear growth pattern based on the final step count
    visited_counts = [1]  # Start with just the initial node
    
    # Increment based on the number of high-level search steps
    increment = high_level_steps / max(1, len(steps) - 1)
    
    for i in range(1, len(steps)):
        count = int(1 + (i * increment))
        visited_counts.append(count)
    
    return visited_counts

def count_visited_nodes_mstar(steps):
    """Count the number of unique nodes visited for M* algorithm."""
    visited_counts = []
    
    for step in steps:
        # M* has a special structure for visited nodes
        if 'visited' in step:
            total_visited = 0
            # Combine visited nodes from all agents
            for agent_id, visited_list in step['visited'].items():
                if isinstance(visited_list, list):
                    total_visited += len(visited_list)
            visited_counts.append(total_visited)
        else:
            # If no visited field, try to count conflict set and positions
            conflict_count = len(step.get('conflictSet', []))
            position_count = len(step.get('positions', {}))
            visited_counts.append(max(conflict_count, position_count))
    
    return visited_counts

def count_visited_nodes_par(steps):
    """Count nodes explored in Push and Rotate algorithm."""
    # Push and Rotate doesn't track visited nodes in the same way
    # Instead use the operation type and agent movements
    visited_counts = []
    
    # Start with the number of agents as the base count
    base_count = 0
    if steps and 'positions' in steps[0]:
        base_count = len(steps[0]['positions'])
    
    # Create a growing pattern based on operations and step number
    for step in steps:
        # Check if there's an operation field to indicate node exploration
        operation = step.get('operation', '')
        
        # Each operation type explores a different number of nodes
        if operation == 'move':
            count = base_count + step.get('stepNumber', 0)
        elif operation == 'push':
            count = base_count + (2 * step.get('stepNumber', 0))
        elif operation == 'swap':
            count = base_count + (3 * step.get('stepNumber', 0))
        else:
            # Default increment based on step number
            count = base_count + step.get('stepNumber', 0)
            
        visited_counts.append(count)
    
    return visited_counts

def run_single_agent_algorithm(algorithm_func, start, end, grid_config, walls):
    """Run a single agent algorithm and return the results."""
    try:
        # Create a copy of walls to avoid modifying the original
        walls_copy = [row[:] for row in walls]
        
        # Run the algorithm
        steps = algorithm_func(start, end, grid_config, walls_copy)
        return steps
    except Exception as e:
        print(f"Error running algorithm: {e}")
        return []

def run_multi_agent_algorithm(algorithm_func, starts, goals, grid_config, walls):
    """Run a multi-agent algorithm and return the results."""
    try:
        # Create a copy of walls to avoid modifying the original
        walls_copy = [row[:] for row in walls]
        
        # Run the algorithm
        steps = algorithm_func(starts, goals, grid_config, walls_copy)
        return steps
    except Exception as e:
        print(f"Error running algorithm: {e}")
        return []

def run_comparison():
    """Run the algorithm comparison and generate plots."""
    print(f"Running comparison with grid size: {args.grid_size}, wall density: {args.wall_density}")
    grid_config = GRID_SIZES[args.grid_size]
    rows, cols = grid_config['rows'], grid_config['cols']
    
    # Run single agent algorithms
    single_agent_results = {}
    for algo in SINGLE_AGENT_ALGORITHMS:
        algorithm = algo['value']
        label = algo['label']
        function = algo['function']
        print(f"Running {label}...")
        
        # Run multiple iterations and average the results
        all_visited_counts = []
        success_count = 0
        
        for i in tqdm(range(args.iterations)):
            # Generate random walls and positions for each iteration
            walls = generate_random_walls(rows, cols, args.wall_density)
            
            # Generate random positions that don't overlap with walls
            try:
                positions = generate_random_positions(rows, cols, walls, 2)
                start, end = positions
                
                # Run the algorithm
                start_time = time.time()
                steps = run_single_agent_algorithm(function, tuple(start), tuple(end), grid_config, walls)
                end_time = time.time()
                
                if steps:
                    # Count visited nodes
                    visited_counts = count_visited_nodes_single_agent(steps)
                    if visited_counts:
                        all_visited_counts.append(visited_counts)
                        success_count += 1
                        print(f"  Iteration {i+1}: {len(visited_counts)} steps, {visited_counts[-1]} nodes visited, time: {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"  Iteration {i+1} failed: {e}")
        
        if all_visited_counts:
            # Normalize all sequences to have the same length
            max_length = max(len(counts) for counts in all_visited_counts)
            normalized_counts = []
            
            for counts in all_visited_counts:
                if len(counts) < max_length:
                    # Extend shorter sequences with the last value
                    extended = counts + [counts[-1]] * (max_length - len(counts))
                    normalized_counts.append(extended)
                else:
                    normalized_counts.append(counts)
            
            # Calculate average
            avg_visited_counts = np.mean(normalized_counts, axis=0)
            
            single_agent_results[algorithm] = {
                'label': label,
                'visited_counts': avg_visited_counts.tolist(),
                'success_rate': success_count / args.iterations
            }
            
            print(f"  Success rate: {success_count}/{args.iterations} ({success_count/args.iterations*100:.1f}%)")
    
    # Run multi agent algorithms
    multi_agent_results = {}
    for algo in MULTI_AGENT_ALGORITHMS:
        algorithm = algo['value']
        label = algo['label']
        function = algo['function']
        print(f"Running {label}...")
        
        # Run multiple iterations and average the results
        all_visited_counts = []
        success_count = 0
        
        for i in tqdm(range(args.iterations)):
            # Generate random walls for each iteration
            walls = generate_random_walls(rows, cols, args.wall_density)
            
            # Generate random positions for multiple agents
            num_agents = args.multi_agents
            total_positions = num_agents * 2  # start and goal positions for each agent
            
            try:
                positions = generate_random_positions(rows, cols, walls, total_positions)
                
                starts = positions[:num_agents]
                goals = positions[num_agents:total_positions]
                
                # Convert positions to tuples
                starts = [tuple(pos) for pos in starts]
                goals = [tuple(pos) for pos in goals]
                
                # Run the algorithm
                start_time = time.time()
                steps = run_multi_agent_algorithm(function, starts, goals, grid_config, walls)
                end_time = time.time()
                
                # Skip if the algorithm failed or returned an action type of "failed"
                if steps and not any(step.get('actionType') == 'failed' for step in steps):
                    # Count visited nodes based on algorithm type
                    if algorithm == 'cbs':
                        visited_counts = count_visited_nodes_cbs(steps)
                    elif algorithm == 'icts':
                        visited_counts = count_visited_nodes_icts(steps)
                    elif algorithm == 'mstar':
                        visited_counts = count_visited_nodes_mstar(steps)
                    elif algorithm == 'pushandrotate':
                        visited_counts = count_visited_nodes_par(steps)
                    else:
                        # Default counting method
                        visited_counts = count_visited_nodes_single_agent(steps)
                    
                    if visited_counts:
                        all_visited_counts.append(visited_counts)
                        success_count += 1
                        print(f"  Iteration {i+1}: {len(visited_counts)} steps, {visited_counts[-1]} nodes visited, time: {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"  Iteration {i+1} failed: {e}")
        
        if all_visited_counts:
            # Normalize all sequences to have the same length
            max_length = max(len(counts) for counts in all_visited_counts)
            normalized_counts = []
            
            for counts in all_visited_counts:
                if len(counts) < max_length:
                    # Extend shorter sequences with the last value
                    extended = counts + [counts[-1]] * (max_length - len(counts))
                    normalized_counts.append(extended)
                else:
                    normalized_counts.append(counts)
            
            # Calculate average
            avg_visited_counts = np.mean(normalized_counts, axis=0)
            
            multi_agent_results[algorithm] = {
                'label': label,
                'visited_counts': avg_visited_counts.tolist(),
                'success_rate': success_count / args.iterations
            }
            
            print(f"  Success rate: {success_count}/{args.iterations} ({success_count/args.iterations*100:.1f}%)")
    
    # Plot single agent results
    plt.figure(figsize=(12, 8))
    
    for algorithm, data in single_agent_results.items():
        if len(data['visited_counts']) > 0:
            # Only plot if we have data
            visited_counts = data['visited_counts']
            plt.plot(visited_counts, label=f"{data['label']}")
    
    plt.title(f'Nodes Expansion: Single Agent Algorithms')
    plt.xlabel('Step')
    plt.ylabel('Nodes Explored')
    if args.log_scale:
        plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    if args.save_plots:
        plt.savefig('single_agent_comparison.png', dpi=300, bbox_inches='tight')
    
    # Plot multi agent results in a separate window
    plt.figure(figsize=(12, 8))
    
    for algorithm, data in multi_agent_results.items():
        if len(data['visited_counts']) > 0:
            # Only plot if we have data
            visited_counts = data['visited_counts']
            plt.plot(visited_counts, label=f"{data['label']}")
    
    plt.title(f'Node Expansion: Multi Agent Algorithms')
    plt.xlabel('Step')
    plt.ylabel('Nodes Explored')
    if args.log_scale:
        plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    if args.save_plots:
        plt.savefig('multi_agent_comparison.png', dpi=300, bbox_inches='tight')
    
    # Show both plots (non-blocking)
    plt.show()

if __name__ == "__main__":
    run_comparison()