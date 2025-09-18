import time
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import tracemalloc
import pandas as pd
import threading
import platform
from typing import Dict, List, Tuple, Optional, Any, Callable

# Import algorithm generators
from algorithms.rw import generate_random_walk
from algorithms.brw import generate_biased_random_walk
from algorithms.bfs import generate_bfs
from algorithms.dfs import generate_dfs
from algorithms.dijkstra import generate_dijkstra
from algorithms.astar import generate_astar
from algorithms.gbfs import generate_gbfs
from algorithms.jps import generate_jps
from algorithms.cbs import generate_cbs
from algorithms.icts import generate_icts
from algorithms.mstar import generate_mstar
from algorithms.par import generate_push_and_rotate

# Utility functions to fix NaN and infinity warnings
def fix_numpy_warnings(df_to_fix):
    """Filter out NaN and infinity values from dataframes used for visualization."""
    # Replace infinity values with NaN
    for col in df_to_fix.select_dtypes(include=[np.number]).columns:
        df_to_fix[col] = df_to_fix[col].replace([np.inf, -np.inf], np.nan)
    
    return df_to_fix

def safe_mean(values):
    """Safely calculate mean ignoring NaN values."""
    values = np.array(values)
    if len(values) == 0 or np.all(np.isnan(values)):
        return 0
    return np.nanmean(values)

def safe_std(values):
    """Safely calculate standard deviation ignoring NaN values."""
    values = np.array(values)
    if len(values) == 0 or np.all(np.isnan(values)) or len(values) < 2:
        return 0
    return np.nanstd(values)

class TimeoutException(Exception):
    """Exception raised when an algorithm execution times out."""
    pass

# Cross-platform timeout mechanism
def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """Run a function with a timeout limit in a cross-platform way."""
    result = [None]
    timed_out = [False]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        timed_out[0] = True
        # Thread is still running, but we're moving on
        return None, True
    
    if exception[0] is not None:
        raise exception[0]
        
    return result[0], False

class PathfindingTester:
    """Framework for testing and comparing pathfinding algorithms."""
    
    def __init__(self):
        # Define algorithm mappings
        self.single_agent_algorithms = {
            "BFS": generate_bfs,
            "DFS": generate_dfs,
            "Dijkstra": generate_dijkstra,
            "A*": generate_astar,
            "GBFS": generate_gbfs,
            "JPS": generate_jps
        }

        self.walk_algorithms = {
            "RandomWalk": generate_random_walk,
            "BiasedRandomWalk": generate_biased_random_walk
        }
        
        self.multi_agent_algorithms = {
            "CBS": generate_cbs,
            "ICTS": generate_icts,
            "M*": generate_mstar,
            "PushAndRotate": generate_push_and_rotate
        }
        
        # Define grid sizes for testing
        self.grid_sizes = {
            "small": {"rows": 10, "cols": 30},
            "medium": {"rows": 15, "cols": 40},
            "large": {"rows": 20, "cols": 50}
        }
        
        # Define wall densities
        self.wall_densities = {
            "sparse": 0.1,
            "medium": 0.3,
            "dense": 0.5
        }
        
        # Results storage
        self.results = {}
        
        # Create output directory
        os.makedirs("test_results", exist_ok=True)
    
    # Map generation methods
    def generate_empty_grid(self, grid_size: Dict[str, int]) -> List[List[bool]]:
        """Generate an empty grid with no obstacles."""
        rows, cols = grid_size["rows"], grid_size["cols"]
        return [[False for _ in range(cols)] for _ in range(rows)]
    
    def generate_random_obstacles(self, grid_size: Dict[str, int], density: float) -> List[List[bool]]:
        """Generate a grid with random obstacles based on density."""
        rows, cols = grid_size["rows"], grid_size["cols"]
        walls = [[False for _ in range(cols)] for _ in range(rows)]
        
        # Add border walls
        for r in range(rows):
            walls[r][0] = True
            walls[r][cols-1] = True
        for c in range(cols):
            walls[0][c] = True
            walls[rows-1][c] = True
        
        # Add random internal walls
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                if random.random() < density:
                    walls[r][c] = True
        
        return walls
    
    def generate_maze(self, grid_size: Dict[str, int]) -> List[List[bool]]:
        """Generate a maze-like grid using recursive division."""
        rows, cols = grid_size["rows"], grid_size["cols"]
        walls = [[False for _ in range(cols)] for _ in range(rows)]
        
        # Add border walls
        for r in range(rows):
            walls[r][0] = True
            walls[r][cols-1] = True
        for c in range(cols):
            walls[0][c] = True
            walls[rows-1][c] = True
        
        # Recursive division maze generation
        def divide(x, y, width, height, orientation):
            if width <= 2 or height <= 2:
                return
            
            horizontal = orientation if orientation is not None else random.random() < height / (height + width)
            
            if horizontal:
                wall_y = y + random.randint(1, height - 1)
                passage_x = x + random.randint(0, width - 1)
                
                for i in range(width):
                    if x + i != passage_x:
                        walls[wall_y][x + i] = True
                
                divide(x, y, width, wall_y - y, horizontal)
                divide(x, wall_y + 1, width, y + height - wall_y - 1, horizontal)
            else:
                wall_x = x + random.randint(1, width - 1)
                passage_y = y + random.randint(0, height - 1)
                
                for i in range(height):
                    if y + i != passage_y:
                        walls[y + i][wall_x] = True
                
                divide(x, y, wall_x - x, height, not horizontal)
                divide(wall_x + 1, y, x + width - wall_x - 1, height, not horizontal)
        
        # Start the recursive division
        divide(1, 1, cols - 2, rows - 2, None)
        
        return walls
    
    def generate_bottleneck(self, grid_size: Dict[str, int]) -> List[List[bool]]:
        """Generate a grid with a bottleneck in the middle."""
        rows, cols = grid_size["rows"], grid_size["cols"]
        walls = [[False for _ in range(cols)] for _ in range(rows)]
        
        # Add border walls
        for r in range(rows):
            walls[r][0] = True
            walls[r][cols-1] = True
        for c in range(cols):
            walls[0][c] = True
            walls[rows-1][c] = True
        
        # Create a bottleneck in the middle
        bottleneck_col = cols // 2
        bottleneck_size = max(1, rows // 8)
        bottleneck_start = (rows - bottleneck_size) // 2
        
        for r in range(1, rows-1):
            if r < bottleneck_start or r >= bottleneck_start + bottleneck_size:
                walls[r][bottleneck_col] = True
        
        return walls
    
    def generate_multi_agent_scenario(self, grid_size: Dict[str, int], 
                                     agent_count: int = 3) -> Tuple[List[List[bool]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Generate a scenario specifically for multi-agent testing."""
        rows, cols = grid_size["rows"], grid_size["cols"]
        walls = self.generate_random_obstacles(grid_size, 0.2)
        
        starts = []
        goals = []
        positions = set()
        
        for _ in range(agent_count):
            # Find valid start position (left side)
            attempts = 0
            while attempts < 100:
                start_r = random.randint(1, rows - 2)
                start_c = random.randint(1, cols // 3)
                if not walls[start_r][start_c] and (start_r, start_c) not in positions:
                    starts.append((start_r, start_c))
                    positions.add((start_r, start_c))
                    break
                attempts += 1
            
            # Find valid goal position (right side)
            attempts = 0
            while attempts < 100:
                goal_r = random.randint(1, rows - 2)
                goal_c = random.randint(2 * cols // 3, cols - 2)
                if not walls[goal_r][goal_c] and (goal_r, goal_c) not in positions:
                    goals.append((goal_r, goal_c))
                    positions.add((goal_r, goal_c))
                    break
                attempts += 1
        
        return walls, starts, goals
    
    # Testing methods
    def test_single_agent_algorithm(self, algorithm_name: str, start: Tuple[int, int], 
                                   end: Tuple[int, int], grid_size: Dict[str, int], 
                                   walls: List[List[bool]], initial_altitudes: Optional[List[List[int]]] = None,
                                   timeout_seconds: int = 30) -> Dict[str, Any]:
        """Test a single-agent algorithm and return performance metrics."""
        algorithm_func = self.single_agent_algorithms[algorithm_name]
        
        # Start memory tracking
        tracemalloc.start()
        
        # Measure execution time
        start_time = time.time()
        
        # Run with timeout
        args = (start, end, grid_size, walls)
        if algorithm_name in ["Dijkstra", "A*"]:
            args += (initial_altitudes,)
            
        steps, timed_out = run_with_timeout(
            algorithm_func, 
            args=args,
            timeout_seconds=timeout_seconds
        )
        
        execution_time = time.time() - start_time
        
        # Get memory usage
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if timed_out or steps is None:
            return {
                "algorithm": algorithm_name,
                "execution_time": timeout_seconds,
                "peak_memory_kb": peak_memory / 1024,
                "path_length": float('inf'),
                "nodes_visited": 0,
                "nodes_considered": 0,
                "is_goal_reached": False,
                "timed_out": True,
                "total_steps": 0
            }
        
        # Extract metrics from the steps
        last_step = steps[-1] if steps else None
        
        path_length = len(last_step["path"]) - 1 if last_step and last_step.get("path") else float('inf')
        nodes_visited = len(last_step["visited"]) if last_step else 0
        nodes_considered = len(last_step.get("considered", [])) if last_step else 0
        is_goal_reached = last_step.get("isGoalReached", False) if last_step else False
        
        return {
            "algorithm": algorithm_name,
            "execution_time": execution_time,
            "peak_memory_kb": peak_memory / 1024,
            "path_length": path_length,
            "nodes_visited": nodes_visited,
            "nodes_considered": nodes_considered,
            "is_goal_reached": is_goal_reached,
            "timed_out": False,
            "total_steps": len(steps),
            "manhattan_distance": abs(start[0] - end[0]) + abs(start[1] - end[1])
        }
    
    def test_random_walk_algorithm(self, algorithm_name: str, start: Tuple[int, int], 
                               end: Tuple[int, int], grid_size: Dict[str, int], 
                               walls: List[List[bool]], timeout_seconds: int = 30) -> Dict[str, Any]:
        """Test a random walk algorithm and return performance metrics."""
        if algorithm_name not in self.walk_algorithms:
            return {
                "algorithm": algorithm_name,
                "error": "Algorithm not found"
            }
            
        algorithm_func = self.walk_algorithms[algorithm_name]
        
        # Start memory tracking
        tracemalloc.start()
        
        # Measure execution time
        start_time = time.time()
        
        # Run with timeout
        steps, timed_out = run_with_timeout(
            algorithm_func, 
            args=(start, end, grid_size, walls),
            timeout_seconds=timeout_seconds
        )
        
        execution_time = time.time() - start_time
        
        # Get memory usage
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if timed_out or steps is None:
            return {
                "algorithm": algorithm_name,
                "execution_time": timeout_seconds,
                "peak_memory_kb": peak_memory / 1024,
                "path_length": float('inf'),
                "nodes_visited": 0,
                "is_goal_reached": False,
                "timed_out": True,
                "total_steps": 0
            }
        
        # Extract metrics from the steps
        last_step = steps[-1] if steps else None
        
        is_goal_reached = last_step.get("isGoalReached", False) if last_step else False
        nodes_visited = len(last_step["visited"]) if last_step else 0
        
        return {
            "algorithm": algorithm_name,
            "execution_time": execution_time,
            "peak_memory_kb": peak_memory / 1024,
            "path_length": last_step["stepNumber"] + 1 if last_step and is_goal_reached else float('inf'),
            "nodes_visited": nodes_visited,
            "is_goal_reached": is_goal_reached,
            "timed_out": False,
            "total_steps": len(steps),
            "manhattan_distance": abs(start[0] - end[0]) + abs(start[1] - end[1])
        }
    
    def test_multi_agent_algorithm(self, algorithm_name: str, starts: List[Tuple[int, int]], 
                                  goals: List[Tuple[int, int]], grid_size: Dict[str, int], 
                                  walls: List[List[bool]], timeout_seconds: int = 60) -> Dict[str, Any]:
        """Test a multi-agent algorithm and return performance metrics."""
        algorithm_func = self.multi_agent_algorithms[algorithm_name]
        
        # Start memory tracking
        tracemalloc.start()
        
        # Measure execution time
        start_time = time.time()
        
        # Run with timeout
        steps, timed_out = run_with_timeout(
            algorithm_func, 
            args=(starts, goals, grid_size, walls),
            timeout_seconds=timeout_seconds
        )
        
        execution_time = time.time() - start_time
        
        # Get memory usage
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if timed_out or steps is None:
            return {
                "algorithm": algorithm_name,
                "execution_time": timeout_seconds,
                "peak_memory_kb": peak_memory / 1024,
                "total_path_length": float('inf'),
                "is_goal_reached": False,
                "timed_out": True,
                "agent_count": len(starts)
            }
        
        # Extract metrics from the steps
        last_step = steps[-1] if steps else None
        
        is_goal_reached = last_step.get("isGoalReached", False) if last_step else False
        
        # Calculate path lengths for all agents
        path_lengths = []
        for agent_id in range(len(starts)):
            if last_step and "paths" in last_step and agent_id in last_step["paths"]:
                path_lengths.append(len(last_step["paths"][agent_id]))
        
        total_path_length = sum(path_lengths) if path_lengths else float('inf')
        
        # Count visited nodes across all agents
        nodes_visited = 0
        if last_step and "visited" in last_step:
            for agent_id in last_step["visited"]:
                nodes_visited += len(last_step["visited"][agent_id])
        
        # Calculate average Manhattan distance from starts to goals
        manhattan_distances = []
        for start, goal in zip(starts, goals):
            manhattan_distances.append(abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
        
        return {
            "algorithm": algorithm_name,
            "execution_time": execution_time,
            "peak_memory_kb": peak_memory / 1024,
            "total_path_length": total_path_length,
            "individual_path_lengths": path_lengths,
            "nodes_visited": nodes_visited,
            "is_goal_reached": is_goal_reached,
            "timed_out": False,
            "total_steps": len(steps),
            "agent_count": len(starts),
            "avg_manhattan_distance": sum(manhattan_distances) / len(manhattan_distances) if manhattan_distances else 0
        }
    
    def run_single_agent_tests(self, scenario_type: str, grid_size_name: str, 
                              wall_density: Optional[str] = None, num_trials: int = 10):
        """Run tests for all single-agent algorithms on the specified scenario type."""
        grid_size = self.grid_sizes[grid_size_name]
        results = []
        
        for trial in range(num_trials):
            print(f"Running single-agent trial {trial+1}/{num_trials} for {scenario_type} on {grid_size_name} grid...")
            
            # Generate scenario based on type
            if scenario_type == "empty":
                walls = self.generate_empty_grid(grid_size)
            elif scenario_type == "random":
                density = self.wall_densities[wall_density or "medium"]
                walls = self.generate_random_obstacles(grid_size, density)
            elif scenario_type == "maze":
                walls = self.generate_maze(grid_size)
            elif scenario_type == "bottleneck":
                walls = self.generate_bottleneck(grid_size)
            else:
                raise ValueError(f"Unknown scenario type: {scenario_type}")
            
            # Generate random start and end positions
            rows, cols = grid_size["rows"], grid_size["cols"]
            
            valid_positions = []
            for r in range(rows):
                for c in range(cols):
                    if not walls[r][c]:
                        valid_positions.append((r, c))
            
            if len(valid_positions) < 2:
                print(f"  Warning: Not enough valid positions in trial {trial}")
                continue
            
            start = random.choice(valid_positions)
            valid_positions.remove(start)
            end = random.choice(valid_positions)
            
            # Generate altitude map for weighted algorithms
            initial_altitudes = [[random.randint(1, 9) for _ in range(grid_size["cols"])] 
                                for _ in range(grid_size["rows"])]
            
            # Run each algorithm
            for algorithm_name in self.single_agent_algorithms:
                try:
                    alt = initial_altitudes if algorithm_name in ["Dijkstra", "A*"] else None
                    result = self.test_single_agent_algorithm(
                        algorithm_name, start, end, grid_size, walls, alt
                    )
                        
                    result.update({
                        "scenario_type": scenario_type,
                        "grid_size": grid_size_name,
                        "wall_density": wall_density,
                        "trial": trial
                    })
                    results.append(result)
                    print(f"  Tested {algorithm_name}: {'✓' if result['is_goal_reached'] else '✗'} ({result['execution_time']:.3f}s)")
                except Exception as e:
                    print(f"  Error testing {algorithm_name} on {scenario_type} grid: {e}")
        
        return results

    def run_random_walk_tests(self, scenario_type: str, grid_size_name: str, 
                            wall_density: Optional[str] = None, num_trials: int = 10):
        """Run tests for random walk algorithms on the specified scenario type."""
        grid_size = self.grid_sizes[grid_size_name]
        results = []
        
        for trial in range(num_trials):
            print(f"Running random walk trial {trial+1}/{num_trials} for {scenario_type} on {grid_size_name} grid...")
            
            # Generate scenario based on type
            if scenario_type == "empty":
                walls = self.generate_empty_grid(grid_size)
            elif scenario_type == "random":
                density = self.wall_densities[wall_density or "medium"]
                walls = self.generate_random_obstacles(grid_size, density)
            elif scenario_type == "maze":
                walls = self.generate_maze(grid_size)
            elif scenario_type == "bottleneck":
                walls = self.generate_bottleneck(grid_size)
            else:
                raise ValueError(f"Unknown scenario type: {scenario_type}")
            
            # Generate random start and end positions
            rows, cols = grid_size["rows"], grid_size["cols"]
            
            valid_positions = []
            for r in range(rows):
                for c in range(cols):
                    if not walls[r][c]:
                        valid_positions.append((r, c))
            
            if len(valid_positions) < 2:
                print(f"  Warning: Not enough valid positions in trial {trial}")
                continue
            
            start = random.choice(valid_positions)
            valid_positions.remove(start)
            end = random.choice(valid_positions)
            
            # Run each random walk algorithm
            for algorithm_name in self.walk_algorithms:
                try:
                    # Use the dedicated random walk testing method instead
                    result = self.test_random_walk_algorithm(
                        algorithm_name, start, end, grid_size, walls
                    )
                        
                    result.update({
                        "scenario_type": scenario_type,
                        "grid_size": grid_size_name,
                        "wall_density": wall_density,
                        "trial": trial
                    })
                    results.append(result)
                    print(f"  Tested {algorithm_name}: {'✓' if result['is_goal_reached'] else '✗'} ({result['execution_time']:.3f}s)")
                except Exception as e:
                    print(f"  Error testing {algorithm_name} on {scenario_type} grid: {e}")
        
        return results
        
    def run_multi_agent_tests(self, scenario_type: str, grid_size_name: str, 
                             agent_count: int = 3, num_trials: int = 5):
        """Run tests for all multi-agent algorithms on the specified scenario type."""
        grid_size = self.grid_sizes[grid_size_name]
        results = []
        
        for trial in range(num_trials):
            print(f"Running multi-agent trial {trial+1}/{num_trials} for {scenario_type} on {grid_size_name} grid with {agent_count} agents...")
            
            try:
                # Generate multi-agent scenario
                walls, starts, goals = self.generate_multi_agent_scenario(grid_size, agent_count)
                
                # Run each algorithm
                for algorithm_name in self.multi_agent_algorithms:
                    try:
                        result = self.test_multi_agent_algorithm(
                            algorithm_name, starts, goals, grid_size, walls
                        )
                        result.update({
                            "scenario_type": scenario_type,
                            "grid_size": grid_size_name,
                            "agent_count": agent_count,
                            "trial": trial
                        })
                        results.append(result)
                        print(f"  Tested {algorithm_name}: {'✓' if result['is_goal_reached'] else '✗'} ({result['execution_time']:.3f}s)")
                    except Exception as e:
                        print(f"  Error testing {algorithm_name} on {scenario_type} grid: {e}")
            except Exception as e:
                print(f"Error setting up trial {trial} for {scenario_type}: {e}")
        
        return results
    

    def run_comprehensive_tests(self, single_agent_only=False, multi_agent_only=False, random_walk_only=False):
        """Run a comprehensive test suite for both single and multi-agent algorithms."""
        # Initialize with existing results to preserve previous test data
        if not hasattr(self, 'results') or self.results is None:
            self.results = {}
        
        all_results = self.results.copy()

        # Random walk tests
        if (not single_agent_only and not multi_agent_only) or random_walk_only:
            print("Running random walk tests...")
            random_walk_results = []
            
            # Test different scenario types
            for scenario in ["empty", "random", "maze", "bottleneck"]:
                # Test different grid sizes
                for size in ["small", "medium", "large"]:
                    # For random scenarios, test different densities
                    if scenario == "random":
                        for density in ["sparse", "medium", "dense"]:
                            results = self.run_random_walk_tests(scenario, size, density, num_trials=5)
                            random_walk_results.extend(results)
                    else:
                        results = self.run_random_walk_tests(scenario, size, num_trials=5)
                        random_walk_results.extend(results)
            
            # Add new results to existing ones if present
            if "random_walk" in all_results:
                all_results["random_walk"].extend(random_walk_results)
            else:
                all_results["random_walk"] = random_walk_results
            
            # Save random walk results to CSV
            random_walk_df = pd.DataFrame(random_walk_results)
            random_walk_df.to_csv("test_results/random_walk_results.csv", index=False)
            print(f"Saved random walk results to test_results/random_walk_results.csv")
        
        # Single-agent tests
        if not multi_agent_only or single_agent_only:
            print("Running single-agent tests...")
            single_agent_results = []
            
            # Test different scenario types
            for scenario in ["empty", "random", "maze", "bottleneck"]:
                # Test different grid sizes
                for size in ["small", "medium", "large"]:
                    # For random scenarios, test different densities
                    if scenario == "random":
                        for density in ["sparse", "medium", "dense"]:
                            results = self.run_single_agent_tests(scenario, size, density, num_trials=5)
                            single_agent_results.extend(results)
                    else:
                        results = self.run_single_agent_tests(scenario, size, num_trials=5)
                        single_agent_results.extend(results)
            
            # Add new results to existing ones if present
            if "single_agent" in all_results:
                all_results["single_agent"].extend(single_agent_results)
            else:
                all_results["single_agent"] = single_agent_results
            
            # Save single-agent results to CSV
            single_df = pd.DataFrame(single_agent_results)
            single_df.to_csv("test_results/single_agent_results.csv", index=False)
            print(f"Saved single-agent results to test_results/single_agent_results.csv")
        
        # Multi-agent tests
        if (not single_agent_only and not random_walk_only) or multi_agent_only:
            print("Running multi-agent tests...")
            multi_agent_results = []
            
            # Test with different agent counts
            for agent_count in [2, 3, 4]:
                # Test different grid sizes
                for size in ["small", "medium"]:
                    results = self.run_multi_agent_tests("multi_scenario", size, agent_count, num_trials=3)
                    multi_agent_results.extend(results)
            
            # Add new results to existing ones if present
            if "multi_agent" in all_results:
                all_results["multi_agent"].extend(multi_agent_results)
            else:
                all_results["multi_agent"] = multi_agent_results
            
            # Save multi-agent results to CSV
            multi_df = pd.DataFrame(multi_agent_results)
            multi_df.to_csv("test_results/multi_agent_results.csv", index=False)
            print(f"Saved multi-agent results to test_results/multi_agent_results.csv")
        
        # Store all results back in the instance
        self.results = all_results
        return all_results
        
    def visualize_results(self):
        """Generate visualizations from the test results."""
        if not hasattr(self, 'results') or not self.results:
            print("No results to visualize. Run tests first.")
            
            # Check if we can load from CSV files
            import os
            import pandas as pd
            
            if os.path.exists("test_results/random_walk_results.csv"):
                print("Loading random walk results from CSV...")
                random_walk_df = pd.read_csv("test_results/random_walk_results.csv")
                if 'results' not in self.__dict__:
                    self.results = {}
                self.results["random_walk"] = random_walk_df.to_dict('records')
            
            if os.path.exists("test_results/single_agent_results.csv"):
                print("Loading single agent results from CSV...")
                single_agent_df = pd.read_csv("test_results/single_agent_results.csv")
                if 'results' not in self.__dict__:
                    self.results = {}
                self.results["single_agent"] = single_agent_df.to_dict('records')
            
            if os.path.exists("test_results/multi_agent_results.csv"):
                print("Loading multi agent results from CSV...")
                multi_agent_df = pd.read_csv("test_results/multi_agent_results.csv")
                if 'results' not in self.__dict__:
                    self.results = {}
                self.results["multi_agent"] = multi_agent_df.to_dict('records')
            
            if not self.results:
                print("No results found. Run tests first.")
                return
        
        single_agent_results = self.results.get("single_agent", [])
        multi_agent_results = self.results.get("multi_agent", [])
        random_walk_results = self.results.get("random_walk", [])

        print(f"Results summary:")
        print(f"- Random walk results: {len(random_walk_results)} entries")
        print(f"- Single agent results: {len(single_agent_results)} entries")
        print(f"- Multi agent results: {len(multi_agent_results)} entries")

        # Visualize random walk results
        if random_walk_results:
            print("Generating random walk visualizations...")
            
            try:
                self._visualize_random_walk_runtime_comparison(random_walk_results)
                self._visualize_random_walk_memory_usage(random_walk_results)
                self._visualize_random_walk_success_rate(random_walk_results)
                self._visualize_random_walk_by_grid_size(random_walk_results)
                self._visualize_random_walk_path_quality(random_walk_results)
                self._visualize_random_walk_by_scenario(random_walk_results)
            except Exception as e:
                import traceback
                print(f"Error generating random walk visualizations: {str(e)}")
                traceback.print_exc()
        else:
            print("No random walk results to visualize.")
        
        # Visualize single-agent results
        if single_agent_results:
            print("Generating single-agent visualizations...")
            self._visualize_runtime_comparison(single_agent_results, "single")
            self._visualize_memory_usage(single_agent_results, "single")
            self._visualize_path_quality(single_agent_results, "single")
            self._visualize_nodes_explored(single_agent_results, "single")
            self._visualize_scaling_with_grid_size(single_agent_results, "single")
            self._visualize_performance_by_scenario(single_agent_results, "single")
            self._visualize_success_rate_by_algorithm(single_agent_results, "single")
        
        # Visualize multi-agent results
        if multi_agent_results:
            print("Generating multi-agent visualizations...")
            self._visualize_runtime_comparison(multi_agent_results, "multi")
            self._visualize_memory_usage(multi_agent_results, "multi")
            self._visualize_path_quality(multi_agent_results, "multi")
            self._visualize_scaling_with_agents(multi_agent_results)
            self._visualize_success_rate_by_algorithm(multi_agent_results, "multi")
            
        print("Visualization process completed.")

    def _visualize_runtime_comparison(self, results, agent_type):
        """Visualize execution time across algorithms."""
        plt.figure(figsize=(12, 8))
        
        # Group by algorithm
        algorithms = sorted(set(result["algorithm"] for result in results))
        avg_times = []
        std_times = []
        
        for algorithm in algorithms:
            algorithm_results = [r for r in results if r["algorithm"] == algorithm and not r["timed_out"]]
            if algorithm_results:
                times = [r["execution_time"] for r in algorithm_results]
                # Filter out any potential np.inf values
                times = [t for t in times if not np.isinf(t)]
                avg_times.append(safe_mean(times))
                std_times.append(safe_std(times))
            else:
                avg_times.append(0)
                std_times.append(0)
        
        # Sort by average time
        sorted_indices = np.argsort(avg_times)
        algorithms = [algorithms[i] for i in sorted_indices]
        avg_times = [avg_times[i] for i in sorted_indices]
        std_times = [std_times[i] for i in sorted_indices]
        
        # Plot
        plt.bar(algorithms, avg_times, yerr=std_times, capsize=5, color='skyblue')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Average Execution Time (seconds)', fontsize=14)
        plt.title(f'Average Execution Time ({agent_type}-agent algorithms)', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"test_results/{agent_type}_agent_runtime_comparison.png")
        plt.close()
    
    def _visualize_memory_usage(self, results, agent_type):
        """Visualize memory usage across algorithms."""
        plt.figure(figsize=(12, 8))
        
        # Group by algorithm
        algorithms = sorted(set(result["algorithm"] for result in results))
        avg_memory = []
        std_memory = []
        
        for algorithm in algorithms:
            algorithm_results = [r for r in results if r["algorithm"] == algorithm and not r["timed_out"]]
            if algorithm_results:
                mem_values = [r["peak_memory_kb"] / 1024 for r in algorithm_results]  # Convert to MB
                # Filter out any potential np.inf values
                mem_values = [m for m in mem_values if not np.isinf(m)]
                avg_memory.append(safe_mean(mem_values))
                std_memory.append(safe_std(mem_values))
            else:
                avg_memory.append(0)
                std_memory.append(0)
        
        # Sort by average memory
        sorted_indices = np.argsort(avg_memory)
        algorithms = [algorithms[i] for i in sorted_indices]
        avg_memory = [avg_memory[i] for i in sorted_indices]
        std_memory = [std_memory[i] for i in sorted_indices]
        
        # Plot
        plt.bar(algorithms, avg_memory, yerr=std_memory, capsize=5, color='lightgreen')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Average Peak Memory Usage (MB)', fontsize=14)
        plt.title(f'Average Memory Usage ({agent_type}-agent algorithms)', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"test_results/{agent_type}_agent_memory_usage.png")
        plt.close()
    
    def _visualize_path_quality(self, results, agent_type):
        """Visualize path quality across algorithms."""
        plt.figure(figsize=(12, 10))
        
        # Group by algorithm
        algorithms = sorted(set(result["algorithm"] for result in results))
        
        # For single-agent, use path_length; for multi-agent, use total_path_length
        path_key = "path_length" if agent_type == "single" else "total_path_length"
        
        avg_path_lengths = []
        std_path_lengths = []
        path_optimality = []
        
        for algorithm in algorithms:
            algorithm_results = [r for r in results if r["algorithm"] == algorithm and r["is_goal_reached"]]
            
            if algorithm_results:
                path_lengths = [r[path_key] for r in algorithm_results if not np.isinf(r[path_key])]
                avg_path_lengths.append(safe_mean(path_lengths))
                std_path_lengths.append(safe_std(path_lengths))
                
                if agent_type == "single":
                    # Calculate path optimality relative to Manhattan distance
                    optimality_values = []
                    for r in algorithm_results:
                        if not np.isinf(r[path_key]) and r["manhattan_distance"] > 0:
                            optimality_values.append(r[path_key] / r["manhattan_distance"])
                    
                    path_optimality.append(safe_mean(optimality_values) if optimality_values else 0)
                else:
                    path_optimality.append(0)  # Not applicable for multi-agent
            else:
                avg_path_lengths.append(0)
                std_path_lengths.append(0)
                path_optimality.append(0)
        
        # Sort by average path length
        sorted_indices = np.argsort(avg_path_lengths)
        algorithms = [algorithms[i] for i in sorted_indices]
        avg_path_lengths = [avg_path_lengths[i] for i in sorted_indices]
        std_path_lengths = [std_path_lengths[i] for i in sorted_indices]
        path_optimality = [path_optimality[i] for i in sorted_indices]
        
        # Plot avg path lengths
        plt.subplot(2, 1, 1)
        plt.bar(algorithms, avg_path_lengths, yerr=std_path_lengths, capsize=5, color='coral')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Average Path Length', fontsize=14)
        plt.title(f'Average Path Length ({agent_type}-agent algorithms)', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot path optimality for single-agent algorithms
        if agent_type == "single":
            plt.subplot(2, 1, 2)
            plt.bar(algorithms, path_optimality, color='lightblue')
            plt.xlabel('Algorithm', fontsize=14)
            plt.ylabel('Path Length / Manhattan Distance', fontsize=14)
            plt.title('Path Optimality', fontsize=16)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"test_results/{agent_type}_agent_path_quality.png")
        plt.close()
    
    def _visualize_nodes_explored(self, results, agent_type):
        """Visualize the number of nodes explored vs. path length."""
        plt.figure(figsize=(12, 8))
        
        algorithms = sorted(set(result["algorithm"] for result in results))
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        
        path_key = "path_length" if agent_type == "single" else "total_path_length"
        
        for i, algorithm in enumerate(algorithms):
            algorithm_results = [r for r in results if r["algorithm"] == algorithm and r["is_goal_reached"]]
            
            if not algorithm_results:
                continue
            
            # Filter out infinite values
            valid_results = [r for r in algorithm_results 
                             if not np.isinf(r.get(path_key, 0)) and not np.isnan(r.get(path_key, 0))
                             and not np.isinf(r.get("nodes_visited", 0)) and not np.isnan(r.get("nodes_visited", 0))]
            
            if not valid_results:
                continue
                
            x = [r.get(path_key, 0) for r in valid_results]
            y = [r.get("nodes_visited", 0) for r in valid_results]
            
            plt.scatter(x, y, label=algorithm, color=colors[i], alpha=0.7, s=50)
        
        plt.xlabel('Path Length', fontsize=14)
        plt.ylabel('Nodes Explored', fontsize=14)
        plt.title(f'Exploration Efficiency ({agent_type}-agent algorithms)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"test_results/{agent_type}_agent_exploration_efficiency.png")
        plt.close()
        
        # Also create a ratio visualization
        plt.figure(figsize=(12, 8))
        
        avg_efficiency = []
        std_efficiency = []
        
        for algorithm in algorithms:
            algorithm_results = [r for r in results if r["algorithm"] == algorithm and r["is_goal_reached"]]
            
            if algorithm_results:
                # Calculate exploration efficiency (nodes visited per path unit)
                efficiency = []
                for r in algorithm_results:
                    path_length = r[path_key]
                    if not np.isinf(path_length) and path_length > 0 and not np.isnan(path_length):
                        ratio = r["nodes_visited"] / path_length
                        if not np.isinf(ratio) and not np.isnan(ratio):
                            efficiency.append(ratio)
                
                avg_efficiency.append(safe_mean(efficiency))
                std_efficiency.append(safe_std(efficiency))
            else:
                avg_efficiency.append(0)
                std_efficiency.append(0)
        
        # Sort by efficiency (lower is better)
        sorted_indices = np.argsort(avg_efficiency)
        algorithms = [algorithms[i] for i in sorted_indices]
        avg_efficiency = [avg_efficiency[i] for i in sorted_indices]
        std_efficiency = [std_efficiency[i] for i in sorted_indices]
        
        plt.bar(algorithms, avg_efficiency, yerr=std_efficiency, capsize=5, color='lightcoral')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Nodes Visited / Path Length', fontsize=14)
        plt.title(f'Exploration Efficiency ({agent_type}-agent algorithms)', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"test_results/{agent_type}_agent_efficiency_ratio.png")
        plt.close()
    
    def _visualize_scaling_with_grid_size(self, results, agent_type):
        """Visualize how algorithms scale with grid size."""
        plt.figure(figsize=(12, 8))
        
        algorithms = sorted(set(result["algorithm"] for result in results))
        grid_sizes = ["small", "medium", "large"]
        
        for algorithm in algorithms:
            avg_times = []
            
            for size in grid_sizes:
                size_results = [r for r in results if r["algorithm"] == algorithm and r["grid_size"] == size and not r["timed_out"]]
                if size_results:
                    times = [r["execution_time"] for r in size_results]
                    # Filter out any potential np.inf values
                    times = [t for t in times if not np.isinf(t)]
                    avg_times.append(safe_mean(times))
                else:
                    avg_times.append(0)
            
            plt.plot(grid_sizes, avg_times, marker='o', linewidth=2, markersize=8, label=algorithm)
        
        plt.xlabel('Grid Size', fontsize=14)
        plt.ylabel('Average Execution Time (seconds)', fontsize=14)
        plt.title(f'Algorithm Scaling with Grid Size ({agent_type}-agent algorithms)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"test_results/{agent_type}_agent_grid_scaling.png")
        plt.close()
    
    def _visualize_scaling_with_agents(self, results):
        """Visualize how multi-agent algorithms scale with the number of agents."""
        plt.figure(figsize=(12, 8))
        
        algorithms = sorted(set(result["algorithm"] for result in results))
        agent_counts = sorted(set(result["agent_count"] for result in results))
        
        for algorithm in algorithms:
            avg_times = []
            
            for count in agent_counts:
                count_results = [r for r in results if r["algorithm"] == algorithm and r["agent_count"] == count and not r["timed_out"]]
                if count_results:
                    times = [r["execution_time"] for r in count_results]
                    # Filter out any potential np.inf values
                    times = [t for t in times if not np.isinf(t)]
                    avg_times.append(safe_mean(times))
                else:
                    avg_times.append(0)
            
            plt.plot(agent_counts, avg_times, marker='o', linewidth=2, markersize=8, label=algorithm)
        
        plt.xlabel('Number of Agents', fontsize=14)
        plt.ylabel('Average Execution Time (seconds)', fontsize=14)
        plt.title('Algorithm Scaling with Number of Agents', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("test_results/multi_agent_scaling.png")
        plt.close()
    
    def _visualize_performance_by_scenario(self, results, agent_type):
        """Visualize algorithm performance across different scenario types."""
        plt.figure(figsize=(14, 10))
        
        algorithms = sorted(set(result["algorithm"] for result in results))
        scenarios = sorted(set(result["scenario_type"] for result in results))
        
        x = np.arange(len(scenarios))
        width = 0.8 / len(algorithms)
        
        for i, algorithm in enumerate(algorithms):
            avg_times = []
            
            for scenario in scenarios:
                scenario_results = [r for r in results if r["algorithm"] == algorithm and r["scenario_type"] == scenario and not r["timed_out"]]
                if scenario_results:
                    times = [r["execution_time"] for r in scenario_results]
                    # Filter out any potential np.inf values
                    times = [t for t in times if not np.isinf(t)]
                    avg_times.append(safe_mean(times))
                else:
                    avg_times.append(0)
            
            plt.bar(x + i * width - 0.4 + width/2, avg_times, width, label=algorithm)
        
        plt.xlabel('Scenario Type', fontsize=14)
        plt.ylabel('Average Execution Time (seconds)', fontsize=14)
        plt.title(f'Performance Across Scenarios ({agent_type}-agent algorithms)', fontsize=16)
        plt.xticks(x, scenarios, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"test_results/{agent_type}_agent_scenario_performance.png")
        plt.close()
    
    def _visualize_success_rate_by_algorithm(self, results, agent_type):
        """Visualize success rate by algorithm."""
        plt.figure(figsize=(12, 8))
        
        algorithms = sorted(set(result["algorithm"] for result in results))
        success_rates = []
        
        for algorithm in algorithms:
            algorithm_results = [r for r in results if r["algorithm"] == algorithm]
            if algorithm_results:
                success_rate = sum(1 for r in algorithm_results if r["is_goal_reached"]) / len(algorithm_results) * 100
                success_rates.append((algorithm, success_rate))
            else:
                success_rates.append((algorithm, 0))
        
        # Sort by success rate
        success_rates.sort(key=lambda x: x[1], reverse=True)
        
        # Plot
        algorithms = [a[0] for a in success_rates]
        rates = [a[1] for a in success_rates]
        
        plt.bar(algorithms, rates, color='lightseagreen')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.title(f'Success Rate by Algorithm ({agent_type}-agent algorithms)', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"test_results/{agent_type}_agent_success_rate.png")
        plt.close()


    def _visualize_random_walk_runtime_comparison(self, results):
        """Visualize execution time across random walk algorithms."""
        plt.figure(figsize=(12, 8))
        
        # Group by algorithm
        algorithms = sorted(set(result["algorithm"] for result in results))
        avg_times = []
        std_times = []
        
        for algorithm in algorithms:
            algorithm_results = [r for r in results if r["algorithm"] == algorithm and not r["timed_out"]]
            if algorithm_results:
                times = [r["execution_time"] for r in algorithm_results]
                times = [t for t in times if not np.isinf(t)]
                avg_times.append(safe_mean(times))
                std_times.append(safe_std(times))
            else:
                avg_times.append(0)
                std_times.append(0)
        
        # Sort by average time
        sorted_indices = np.argsort(avg_times)
        algorithms = [algorithms[i] for i in sorted_indices]
        avg_times = [avg_times[i] for i in sorted_indices]
        std_times = [std_times[i] for i in sorted_indices]
        
        # Plot
        plt.bar(algorithms, avg_times, yerr=std_times, capsize=5, color='skyblue')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Average Execution Time (seconds)', fontsize=14)
        plt.title('Average Execution Time (Random Walk Algorithms)', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("test_results/random_walk_runtime_comparison.png")
        plt.close()

    def _visualize_random_walk_memory_usage(self, results):
        """Visualize memory usage across random walk algorithms."""
        plt.figure(figsize=(12, 8))
        
        # Group by algorithm
        algorithms = sorted(set(result["algorithm"] for result in results))
        avg_memory = []
        std_memory = []
        
        for algorithm in algorithms:
            algorithm_results = [r for r in results if r["algorithm"] == algorithm and not r["timed_out"]]
            if algorithm_results:
                mem_values = [r["peak_memory_kb"] / 1024 for r in algorithm_results]  # Convert to MB
                mem_values = [m for m in mem_values if not np.isinf(m)]
                avg_memory.append(safe_mean(mem_values))
                std_memory.append(safe_std(mem_values))
            else:
                avg_memory.append(0)
                std_memory.append(0)
        
        # Sort by average memory
        sorted_indices = np.argsort(avg_memory)
        algorithms = [algorithms[i] for i in sorted_indices]
        avg_memory = [avg_memory[i] for i in sorted_indices]
        std_memory = [std_memory[i] for i in sorted_indices]
        
        # Plot
        plt.bar(algorithms, avg_memory, yerr=std_memory, capsize=5, color='lightgreen')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Average Peak Memory Usage (MB)', fontsize=14)
        plt.title('Average Memory Usage (Random Walk Algorithms)', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("test_results/random_walk_memory_usage.png")
        plt.close()

    def _visualize_random_walk_success_rate(self, results):
        """Visualize success rate across random walk algorithms."""
        plt.figure(figsize=(12, 8))
        
        algorithms = sorted(set(result["algorithm"] for result in results))
        success_rates = []
        
        for algorithm in algorithms:
            algorithm_results = [r for r in results if r["algorithm"] == algorithm]
            if algorithm_results:
                success_rate = sum(1 for r in algorithm_results if r["is_goal_reached"]) / len(algorithm_results) * 100
                success_rates.append((algorithm, success_rate))
            else:
                success_rates.append((algorithm, 0))
        
        # Sort by success rate
        success_rates.sort(key=lambda x: x[1], reverse=True)
        
        # Plot
        algorithms = [a[0] for a in success_rates]
        rates = [a[1] for a in success_rates]
        
        plt.bar(algorithms, rates, color='lightseagreen')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.title('Success Rate by Algorithm (Random Walk Algorithms)', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("test_results/random_walk_success_rate.png")
        plt.close()


    def _visualize_random_walk_by_grid_size(self, results):
        """Visualize how random walk algorithms scale with grid size."""
        plt.figure(figsize=(12, 8))
        
        algorithms = sorted(set(result["algorithm"] for result in results))
        grid_sizes = ["small", "medium", "large"]
        
        for algorithm in algorithms:
            avg_times = []
            success_rates = []
            
            for size in grid_sizes:
                size_results = [r for r in results if r["algorithm"] == algorithm and r["grid_size"] == size and not r.get("timed_out", False)]
                if size_results:
                    times = [r["execution_time"] for r in size_results]
                    times = [t for t in times if not np.isinf(t)]
                    avg_times.append(safe_mean(times))
                    
                    success_rate = sum(1 for r in size_results if r["is_goal_reached"]) / len(size_results) * 100
                    success_rates.append(success_rate)
                else:
                    avg_times.append(0)
                    success_rates.append(0)
            
            plt.plot(grid_sizes, avg_times, marker='o', linewidth=2, markersize=8, label=f"{algorithm} (time)")
        
        plt.xlabel('Grid Size', fontsize=14)
        plt.ylabel('Average Execution Time (seconds)', fontsize=14)
        plt.title('Random Walk Algorithm Scaling with Grid Size', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("test_results/random_walk_grid_scaling.png")
        plt.close()
        
        # Create a second plot for success rates by grid size
        plt.figure(figsize=(12, 8))
        
        for algorithm in algorithms:
            success_rates = []
            
            for size in grid_sizes:
                size_results = [r for r in results if r["algorithm"] == algorithm and r["grid_size"] == size]
                if size_results:
                    success_rate = sum(1 for r in size_results if r["is_goal_reached"]) / len(size_results) * 100
                    success_rates.append(success_rate)
                else:
                    success_rates.append(0)
            
            plt.plot(grid_sizes, success_rates, marker='o', linewidth=2, markersize=8, label=algorithm)
        
        plt.xlabel('Grid Size', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.title('Random Walk Success Rate by Grid Size', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("test_results/random_walk_success_by_grid_size.png")
        plt.close()

    def _visualize_random_walk_by_scenario(self, results):
        """Visualize random walk performance across different scenario types."""
        plt.figure(figsize=(14, 10))
        
        algorithms = sorted(set(result["algorithm"] for result in results))
        scenarios = sorted(set(result["scenario_type"] for result in results))
        
        x = np.arange(len(scenarios))
        width = 0.8 / len(algorithms)
        
        for i, algorithm in enumerate(algorithms):
            avg_times = []
            success_rates = []
            
            for scenario in scenarios:
                scenario_results = [r for r in results if r["algorithm"] == algorithm and r["scenario_type"] == scenario and not r.get("timed_out", False)]
                if scenario_results:
                    times = [r["execution_time"] for r in scenario_results]
                    times = [t for t in times if not np.isinf(t)]
                    avg_times.append(safe_mean(times))
                    
                    success_rate = sum(1 for r in scenario_results if r["is_goal_reached"]) / len(scenario_results) * 100
                    success_rates.append(success_rate)
                else:
                    avg_times.append(0)
                    success_rates.append(0)
            
            plt.bar(x + i * width - 0.4 + width/2, avg_times, width, label=algorithm)
        
        plt.xlabel('Scenario Type', fontsize=14)
        plt.ylabel('Average Execution Time (seconds)', fontsize=14)
        plt.title('Random Walk Performance Across Scenarios', fontsize=16)
        plt.xticks(x, scenarios, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("test_results/random_walk_scenario_performance.png")
        plt.close()
        
        # Create a second plot for success rates by scenario
        plt.figure(figsize=(14, 10))
        
        for i, algorithm in enumerate(algorithms):
            success_rates = []
            
            for scenario in scenarios:
                scenario_results = [r for r in results if r["algorithm"] == algorithm and r["scenario_type"] == scenario]
                if scenario_results:
                    success_rate = sum(1 for r in scenario_results if r["is_goal_reached"]) / len(scenario_results) * 100
                    success_rates.append(success_rate)
                else:
                    success_rates.append(0)
            
            plt.bar(x + i * width - 0.4 + width/2, success_rates, width, label=algorithm)
        
        plt.xlabel('Scenario Type', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.title('Random Walk Success Rate by Scenario Type', fontsize=16)
        plt.xticks(x, scenarios, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("test_results/random_walk_success_by_scenario.png")
        plt.close()

    def _visualize_random_walk_path_quality(self, results):
        """Visualize path quality for random walk algorithms."""
        plt.figure(figsize=(12, 8))
        
        # Only use successful runs
        successful_results = [r for r in results if r["is_goal_reached"]]
        
        if not successful_results:
            print("No successful random walk results to visualize path quality.")
            return
        
        algorithms = sorted(set(result["algorithm"] for result in successful_results))
        
        avg_path_lengths = []
        std_path_lengths = []
        path_optimality = []
        
        for algorithm in algorithms:
            algorithm_results = [r for r in successful_results if r["algorithm"] == algorithm]
            
            if algorithm_results:
                path_lengths = [r["path_length"] for r in algorithm_results if not np.isinf(r["path_length"])]
                avg_path_lengths.append(safe_mean(path_lengths))
                std_path_lengths.append(safe_std(path_lengths))
                
                # Calculate path optimality relative to Manhattan distance
                optimality_values = []
                for r in algorithm_results:
                    if not np.isinf(r["path_length"]) and r["manhattan_distance"] > 0:
                        optimality_values.append(r["path_length"] / r["manhattan_distance"])
                
                path_optimality.append(safe_mean(optimality_values) if optimality_values else 0)
            else:
                avg_path_lengths.append(0)
                std_path_lengths.append(0)
                path_optimality.append(0)
        
        # Sort by average path length
        sorted_indices = np.argsort(avg_path_lengths)
        algorithms = [algorithms[i] for i in sorted_indices]
        avg_path_lengths = [avg_path_lengths[i] for i in sorted_indices]
        std_path_lengths = [std_path_lengths[i] for i in sorted_indices]
        path_optimality = [path_optimality[i] for i in sorted_indices]
        
        # Plot avg path lengths
        plt.subplot(2, 1, 1)
        plt.bar(algorithms, avg_path_lengths, yerr=std_path_lengths, capsize=5, color='coral')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Average Path Length', fontsize=14)
        plt.title('Random Walk Average Path Length', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot path optimality
        plt.subplot(2, 1, 2)
        plt.bar(algorithms, path_optimality, color='lightblue')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Path Length / Manhattan Distance', fontsize=14)
        plt.title('Random Walk Path Optimality', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("test_results/random_walk_path_quality.png")
        plt.close()



def main():
    print("Starting Pathfinding Algorithm Tests")
    print(f"Running on platform: {platform.system()}")
    
    # Create the tester
    tester = PathfindingTester()
    
    # Run comprehensive tests or just a subset
    single_agent_only = False  # Set to True to only test single-agent algorithms
    multi_agent_only = False   # Set to True to only test multi-agent algorithms
    random_walk_only = False   # Set to True to only test random walk algorithms
    
    # Run tests
    print("Running tests...")
    results = tester.run_comprehensive_tests(single_agent_only, multi_agent_only, random_walk_only)
    
    # Generate visualizations
    print("Generating visualizations...")
    tester.visualize_results()
    
    print("Testing complete. Results and visualizations saved to the test_results directory.")

if __name__ == "__main__":
    main()