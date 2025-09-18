#!/usr/bin/env python3
# Specialized benchmarks for pathfinding algorithms

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tracemalloc
import random
import signal
from typing import Dict, List, Tuple, Optional, Callable, Any
import argparse

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

class TimeoutException(Exception):
    """Exception raised when algorithm execution times out."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutException("Algorithm execution timed out")

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """Run a function with a timeout limit."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Disable the alarm
        return result, False
    except TimeoutException:
        return None, True
    finally:
        signal.alarm(0)  # Ensure the alarm is disabled

class PathfindingBenchmark:
    """Specialized benchmarks to test pathfinding algorithms under challenging conditions."""
    
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
        
        self.multi_agent_algorithms = {
            "CBS": generate_cbs,
            "ICTS": generate_icts,
            "M*": generate_mstar,
            "PushAndRotate": generate_push_and_rotate
        }

        self.walk_algorithms = {
            "RandomWalk": generate_random_walk,
            "BiasedRandomWalk": generate_biased_random_walk
        }
        
        # Create output directory
        os.makedirs("benchmark_results", exist_ok=True)
    
    def generate_maze_with_narrow_corridors(self, rows: int, cols: int) -> List[List[bool]]:
        """Generate a maze-like grid with narrow corridors."""
        walls = [[True for _ in range(cols)] for _ in range(rows)]
        
        # Create narrow corridors
        for r in range(1, rows, 4):
            for c in range(1, cols):
                walls[r][c] = False
        
        for c in range(1, cols, 6):
            for r in range(1, rows):
                walls[r][c] = False
        
        # Ensure start and end are accessible
        walls[1][1] = False
        walls[rows-2][cols-2] = False
        
        return walls
    
    def generate_bottleneck_maze(self, rows: int, cols: int) -> List[List[bool]]:
        """Generate a maze with a bottleneck in the middle."""
        walls = [[False for _ in range(cols)] for _ in range(rows)]
        
        # Add border walls
        for r in range(rows):
            walls[r][0] = True
            walls[r][cols-1] = True
        for c in range(cols):
            walls[0][c] = True
            walls[rows-1][c] = True
        
        # Create bottleneck in the middle
        bottleneck_col = cols // 2
        bottleneck_row = rows // 2
        bottleneck_size = 2
        
        # Top and bottom walls
        for c in range(1, cols-1):
            if c != bottleneck_col:
                for r in range(1, bottleneck_row - bottleneck_size):
                    walls[r][c] = random.random() < 0.4
                for r in range(bottleneck_row + bottleneck_size, rows-1):
                    walls[r][c] = random.random() < 0.4
        
        # Middle section walls - force bottleneck
        for r in range(1, rows-1):
            if r < bottleneck_row - bottleneck_size or r > bottleneck_row + bottleneck_size:
                walls[r][bottleneck_col] = True
        
        # Ensure start and end are accessible
        walls[1][1] = False
        walls[rows-2][cols-2] = False
        
        return walls
    
    def generate_spiral_maze(self, rows: int, cols: int) -> List[List[bool]]:
        """Generate a spiral-shaped maze."""
        walls = [[True for _ in range(cols)] for _ in range(rows)]
        
        # Create a spiral path
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        direction_idx = 0
        r, c = 1, 1
        
        walls[r][c] = False  # Start point
        max_steps = rows * cols
        step_count = 1
        
        while step_count < max_steps:
            dr, dc = directions[direction_idx]
            next_r, next_c = r + dr, c + dc
            
            if (0 < next_r < rows-1 and 
                0 < next_c < cols-1 and 
                walls[next_r][next_c]):
                walls[next_r][next_c] = False
                r, c = next_r, next_c
                step_count += 1
            else:
                # Change direction
                direction_idx = (direction_idx + 1) % 4
        
        # Ensure there's a path from center to the end
        mid_r, mid_c = rows // 2, cols // 2
        end_r, end_c = rows - 2, cols - 2
        
        # Create a path from center to end
        for i in range(min(mid_r, end_r), max(mid_r, end_r) + 1):
            walls[i][mid_c] = False
        for j in range(min(mid_c, end_c), max(mid_c, end_c) + 1):
            walls[end_r][j] = False
        
        return walls
    
    def generate_varying_altitudes(self, rows: int, cols: int, 
                                   pattern: str = 'random') -> List[List[int]]:
        """Generate altitude map for weighted algorithms."""
        altitudes = [[1 for _ in range(cols)] for _ in range(rows)]
        
        if pattern == 'random':
            # Random altitudes between 1 and 9
            for r in range(rows):
                for c in range(cols):
                    altitudes[r][c] = random.randint(1, 9)
                    
        elif pattern == 'gradient':
            # Linear gradient from left to right
            for r in range(rows):
                for c in range(cols):
                    altitudes[r][c] = 1 + int(8 * c / (cols - 1))
                    
        elif pattern == 'radial':
            # Radial gradient from center
            center_r, center_c = rows // 2, cols // 2
            max_dist = max(rows, cols) // 2
            
            for r in range(rows):
                for c in range(cols):
                    dist = ((r - center_r)**2 + (c - center_c)**2)**0.5
                    altitudes[r][c] = 1 + min(8, int(8 * dist / max_dist))
                    
        elif pattern == 'valley':
            # Create a valley with high costs on sides, low in middle
            for r in range(rows):
                for c in range(cols):
                    rel_c = abs(c - cols // 2) / (cols // 2)
                    altitudes[r][c] = 1 + int(8 * rel_c)
                    
        elif pattern == 'checkerboard':
            # Alternating high and low costs
            for r in range(rows):
                for c in range(cols):
                    if (r + c) % 2 == 0:
                        altitudes[r][c] = 1
                    else:
                        altitudes[r][c] = 9
                        
        return altitudes
    
    def generate_multi_agent_crossover(self, rows: int, cols: int, 
                                      agent_count: int = 4) -> Tuple[List[List[bool]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Generate scenario where agents must cross each other's paths."""
        walls = [[False for _ in range(cols)] for _ in range(rows)]
        
        # Add border walls
        for r in range(rows):
            walls[r][0] = True
            walls[r][cols-1] = True
        for c in range(cols):
            walls[0][c] = True
            walls[rows-1][c] = True
        
        # Create a plus-shaped intersection in the middle
        center_r, center_c = rows // 2, cols // 2
        corridor_width = 2
        
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                if not (
                    center_r - corridor_width <= r <= center_r + corridor_width or
                    center_c - corridor_width <= c <= center_c + corridor_width
                ):
                    walls[r][c] = True
        
        # Position agents at opposite ends of the corridors
        starts = []
        goals = []
        
        # Top to bottom
        starts.append((1, center_c))
        goals.append((rows-2, center_c))
        
        # Bottom to top
        starts.append((rows-2, center_c + 1))
        goals.append((1, center_c + 1))
        
        # Left to right
        starts.append((center_r, 1))
        goals.append((center_r, cols-2))
        
        # Right to left
        starts.append((center_r + 1, cols-2))
        goals.append((center_r + 1, 1))
        
        # Limit to requested number of agents
        starts = starts[:agent_count]
        goals = goals[:agent_count]
        
        return walls, starts, goals
    
    def generate_multi_agent_corridor(self, rows: int, cols: int, 
                                     agent_count: int = 4) -> Tuple[List[List[bool]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Generate scenario with agents in a narrow corridor that must pass each other."""
        walls = [[True for _ in range(cols)] for _ in range(rows)]
        
        # Create a single corridor
        corridor_row = rows // 2
        for c in range(1, cols-1):
            walls[corridor_row][c] = False
            walls[corridor_row+1][c] = False
        
        # Place agents at opposite ends of the corridor
        starts = []
        goals = []
        
        for i in range(agent_count):
            if i % 2 == 0:
                # Left to right
                starts.append((corridor_row + (i // 2) % 2, 1 + i // 4))
                goals.append((corridor_row + (i // 2) % 2, cols - 2 - i // 4))
            else:
                # Right to left
                starts.append((corridor_row + (i // 2) % 2, cols - 2 - i // 4))
                goals.append((corridor_row + (i // 2) % 2, 1 + i // 4))
        
        return walls, starts, goals
    
    def test_single_agent_algorithm(self, algorithm_name: str, start: Tuple[int, int], 
                                   end: Tuple[int, int], grid_size: Dict[str, int], 
                                   walls: List[List[bool]], initial_altitudes: Optional[List[List[int]]] = None,
                                   timeout_seconds: int = 30) -> Dict[str, Any]:
        """Test a single-agent algorithm and return performance metrics."""
        if algorithm_name not in self.single_agent_algorithms:
            return {
                "algorithm": algorithm_name,
                "error": "Algorithm not found"
            }
            
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
                "is_goal_reached": False,
                "timed_out": True
            }
        
        # Extract metrics from the steps
        last_step = steps[-1] if steps else None
        
        path_length = len(last_step["path"]) - 1 if last_step and last_step.get("path") else float('inf')
        nodes_visited = len(last_step["visited"]) if last_step else 0
        is_goal_reached = last_step.get("isGoalReached", False) if last_step else False
        
        return {
            "algorithm": algorithm_name,
            "execution_time": execution_time,
            "peak_memory_kb": peak_memory / 1024,
            "path_length": path_length,
            "nodes_visited": nodes_visited,
            "is_goal_reached": is_goal_reached,
            "timed_out": False,
            "total_steps": len(steps)
        }
    
    def test_multi_agent_algorithm(self, algorithm_name: str, starts: List[Tuple[int, int]], 
                                  goals: List[Tuple[int, int]], grid_size: Dict[str, int], 
                                  walls: List[List[bool]], timeout_seconds: int = 60) -> Dict[str, Any]:
        """Test a multi-agent algorithm and return performance metrics."""
        if algorithm_name not in self.multi_agent_algorithms:
            return {
                "algorithm": algorithm_name,
                "error": "Algorithm not found"
            }
            
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
        
        return {
            "algorithm": algorithm_name,
            "execution_time": execution_time,
            "peak_memory_kb": peak_memory / 1024,
            "total_path_length": total_path_length,
            "individual_path_lengths": path_lengths,
            "is_goal_reached": is_goal_reached,
            "timed_out": False,
            "total_steps": len(steps),
            "agent_count": len(starts)
        }
    
    def benchmark_narrow_corridors(self, algorithms=None, grid_size=(40, 100), num_trials=5):
        """Benchmark algorithms on narrow corridor maze."""
        rows, cols = grid_size
        results = []
        
        print(f"Running narrow corridors benchmark ({rows}x{cols} grid)...")
        
        for trial in range(num_trials):
            print(f"  Trial {trial+1}/{num_trials}")
            
            # Generate maze
            walls = self.generate_maze_with_narrow_corridors(rows, cols)
            grid_size_dict = {"rows": rows, "cols": cols}
            
            # Set start and end positions
            start = (1, 1)
            end = (rows-2, cols-2)
            
            # Run single-agent algorithms
            if algorithms is None:
                algo_list = list(self.single_agent_algorithms.keys())
            else:
                algo_list = [a for a in algorithms if a in self.single_agent_algorithms]
                
            for algo in algo_list:
                print(f"    Testing {algo}...")
                initial_altitudes = None
                if algo in ["Dijkstra", "A*"]:
                    initial_altitudes = self.generate_varying_altitudes(rows, cols, "random")
                    
                result = self.test_single_agent_algorithm(
                    algo, start, end, grid_size_dict, walls, initial_altitudes
                )
                
                result["trial"] = trial
                result["benchmark"] = "narrow_corridors"
                results.append(result)
                
                print(f"      {'✓' if result['is_goal_reached'] else '✗'} ({result['execution_time']:.3f}s)")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv("benchmark_results/narrow_corridors_benchmark.csv", index=False)
        
        # Create visualizations
        self._visualize_benchmark_results(df, "narrow_corridors")
        
        return results
    
    def benchmark_bottleneck(self, algorithms=None, grid_size=(40, 80), num_trials=5):
        """Benchmark algorithms on bottleneck maze."""
        rows, cols = grid_size
        results = []
        
        print(f"Running bottleneck benchmark ({rows}x{cols} grid)...")
        
        for trial in range(num_trials):
            print(f"  Trial {trial+1}/{num_trials}")
            
            # Generate maze
            walls = self.generate_bottleneck_maze(rows, cols)
            grid_size_dict = {"rows": rows, "cols": cols}
            
            # Set start and end positions
            start = (1, 1)
            end = (rows-2, cols-2)
            
            # Run single-agent algorithms
            if algorithms is None:
                algo_list = list(self.single_agent_algorithms.keys())
            else:
                algo_list = [a for a in algorithms if a in self.single_agent_algorithms]
                
            for algo in algo_list:
                print(f"    Testing {algo}...")
                initial_altitudes = None
                if algo in ["Dijkstra", "A*"]:
                    initial_altitudes = self.generate_varying_altitudes(rows, cols, "random")
                    
                result = self.test_single_agent_algorithm(
                    algo, start, end, grid_size_dict, walls, initial_altitudes
                )
                
                result["trial"] = trial
                result["benchmark"] = "bottleneck"
                results.append(result)
                
                print(f"      {'✓' if result['is_goal_reached'] else '✗'} ({result['execution_time']:.3f}s)")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv("benchmark_results/bottleneck_benchmark.csv", index=False)
        
        # Create visualizations
        self._visualize_benchmark_results(df, "bottleneck")
        
        return results
    
    def benchmark_spiral(self, algorithms=None, grid_size=(40, 40), num_trials=5):
        """Benchmark algorithms on spiral maze."""
        rows, cols = grid_size
        results = []
        
        print(f"Running spiral benchmark ({rows}x{cols} grid)...")
        
        for trial in range(num_trials):
            print(f"  Trial {trial+1}/{num_trials}")
            
            # Generate maze
            walls = self.generate_spiral_maze(rows, cols)
            grid_size_dict = {"rows": rows, "cols": cols}
            
            # Set start and end positions
            start = (1, 1)
            end = (rows-2, cols-2)
            
            # Run single-agent algorithms
            if algorithms is None:
                algo_list = list(self.single_agent_algorithms.keys())
            else:
                algo_list = [a for a in algorithms if a in self.single_agent_algorithms]
                
            for algo in algo_list:
                print(f"    Testing {algo}...")
                initial_altitudes = None
                if algo in ["Dijkstra", "A*"]:
                    initial_altitudes = self.generate_varying_altitudes(rows, cols, "random")
                    
                result = self.test_single_agent_algorithm(
                    algo, start, end, grid_size_dict, walls, initial_altitudes
                )
                
                result["trial"] = trial
                result["benchmark"] = "spiral"
                results.append(result)
                
                print(f"      {'✓' if result['is_goal_reached'] else '✗'} ({result['execution_time']:.3f}s)")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv("benchmark_results/spiral_benchmark.csv", index=False)
        
        # Create visualizations
        self._visualize_benchmark_results(df, "spiral")
        
        return results
    
    def benchmark_altitude_patterns(self, algorithms=None, grid_size=(30, 30), num_trials=3):
        """Benchmark weighted algorithms with different altitude patterns."""
        rows, cols = grid_size
        results = []
        patterns = ['random', 'gradient', 'radial', 'valley', 'checkerboard']
        
        print(f"Running altitude patterns benchmark ({rows}x{cols} grid)...")
        
        # Only test weighted algorithms
        weighted_algos = ["Dijkstra", "A*"]
        if algorithms is not None:
            weighted_algos = [a for a in algorithms if a in weighted_algos]
            
        if not weighted_algos:
            print("  No weighted algorithms selected, skipping benchmark.")
            return []
        
        for pattern in patterns:
            print(f"  Testing with {pattern} altitude pattern:")
            
            for trial in range(num_trials):
                print(f"    Trial {trial+1}/{num_trials}")
                
                # Generate empty grid with border walls
                walls = [[False for _ in range(cols)] for _ in range(rows)]
                for r in range(rows):
                    walls[r][0] = walls[r][cols-1] = True
                for c in range(cols):
                    walls[0][c] = walls[rows-1][c] = True
                
                grid_size_dict = {"rows": rows, "cols": cols}
                
                # Set start and end positions
                start = (1, 1)
                end = (rows-2, cols-2)
                
                # Generate altitude map
                altitudes = self.generate_varying_altitudes(rows, cols, pattern)
                
                for algo in weighted_algos:
                    print(f"      Testing {algo}...")
                    
                    result = self.test_single_agent_algorithm(
                        algo, start, end, grid_size_dict, walls, altitudes
                    )
                    
                    result["trial"] = trial
                    result["benchmark"] = "altitude_patterns"
                    result["pattern"] = pattern
                    results.append(result)
                    
                    print(f"        {'✓' if result['is_goal_reached'] else '✗'} ({result['execution_time']:.3f}s)")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv("benchmark_results/altitude_patterns_benchmark.csv", index=False)
        
        # Create visualizations by pattern
        for pattern in patterns:
            pattern_df = df[df['pattern'] == pattern]
            if not pattern_df.empty:
                self._visualize_benchmark_results(pattern_df, f"altitude_{pattern}")
        
        return results
    
    def benchmark_agent_crossover(self, algorithms=None, grid_size=(30, 30), agent_counts=[2, 3, 4], num_trials=3):
        """Benchmark multi-agent algorithms with agents crossing paths."""
        rows, cols = grid_size
        results = []
        
        print(f"Running agent crossover benchmark ({rows}x{cols} grid)...")
        
        # Filter multi-agent algorithms
        if algorithms is None:
            algo_list = list(self.multi_agent_algorithms.keys())
        else:
            algo_list = [a for a in algorithms if a in self.multi_agent_algorithms]
            
        if not algo_list:
            print("  No multi-agent algorithms selected, skipping benchmark.")
            return []
        
        for agent_count in agent_counts:
            print(f"  Testing with {agent_count} agents:")
            
            for trial in range(num_trials):
                print(f"    Trial {trial+1}/{num_trials}")
                
                # Generate crossover scenario
                walls, starts, goals = self.generate_multi_agent_crossover(rows, cols, agent_count)
                grid_size_dict = {"rows": rows, "cols": cols}
                
                for algo in algo_list:
                    print(f"      Testing {algo}...")
                    
                    result = self.test_multi_agent_algorithm(
                        algo, starts, goals, grid_size_dict, walls, timeout_seconds=120
                    )
                    
                    result["trial"] = trial
                    result["benchmark"] = "agent_crossover"
                    result["agent_count"] = agent_count
                    results.append(result)
                    
                    print(f"        {'✓' if result['is_goal_reached'] else '✗'} ({result['execution_time']:.3f}s)")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv("benchmark_results/agent_crossover_benchmark.csv", index=False)
        
        # Create visualizations
        self._visualize_multi_agent_results(df, "agent_crossover")
        
        return results
    
    def benchmark_agent_corridor(self, algorithms=None, grid_size=(20, 50), agent_counts=[2, 3, 4], num_trials=3):
        """Benchmark multi-agent algorithms with agents in a narrow corridor."""
        rows, cols = grid_size
        results = []
        
        print(f"Running agent corridor benchmark ({rows}x{cols} grid)...")
        
        # Filter multi-agent algorithms
        if algorithms is None:
            algo_list = list(self.multi_agent_algorithms.keys())
        else:
            algo_list = [a for a in algorithms if a in self.multi_agent_algorithms]
            
        if not algo_list:
            print("  No multi-agent algorithms selected, skipping benchmark.")
            return []
        
        for agent_count in agent_counts:
            print(f"  Testing with {agent_count} agents:")
            
            for trial in range(num_trials):
                print(f"    Trial {trial+1}/{num_trials}")
                
                # Generate corridor scenario
                walls, starts, goals = self.generate_multi_agent_corridor(rows, cols, agent_count)
                grid_size_dict = {"rows": rows, "cols": cols}
                
                for algo in algo_list:
                    print(f"      Testing {algo}...")
                    
                    result = self.test_multi_agent_algorithm(
                        algo, starts, goals, grid_size_dict, walls, timeout_seconds=120
                    )
                    
                    result["trial"] = trial
                    result["benchmark"] = "agent_corridor"
                    result["agent_count"] = agent_count
                    results.append(result)
                    
                    print(f"        {'✓' if result['is_goal_reached'] else '✗'} ({result['execution_time']:.3f}s)")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv("benchmark_results/agent_corridor_benchmark.csv", index=False)
        
        # Create visualizations
        self._visualize_multi_agent_results(df, "agent_corridor")
        
        return results
    
    def run_all_benchmarks(self, single_agent_algorithms=None, multi_agent_algorithms=None):
        """Run all benchmarks for the specified algorithms."""
        all_algorithms = []
        
        if single_agent_algorithms:
            all_algorithms.extend(single_agent_algorithms)
        
        if multi_agent_algorithms:
            all_algorithms.extend(multi_agent_algorithms)
        
        # Run single-agent benchmarks
        print("Running single-agent benchmarks...")
        self.benchmark_narrow_corridors(all_algorithms)
        self.benchmark_bottleneck(all_algorithms)
        self.benchmark_spiral(all_algorithms)
        self.benchmark_altitude_patterns(all_algorithms)
        
        # Run multi-agent benchmarks
        print("\nRunning multi-agent benchmarks...")
        self.benchmark_agent_crossover(all_algorithms)
        self.benchmark_agent_corridor(all_algorithms)
        
        print("\nAll benchmarks completed. Results saved to 'benchmark_results' directory.")
    
    def _visualize_benchmark_results(self, df, benchmark_name):
        """Create visualizations for single-agent benchmark results."""
        if df.empty:
            return
        
        # Filter successful runs
        df_success = df[df['is_goal_reached']]
        
        # 1. Execution Time Comparison
        plt.figure(figsize=(12, 8))
        
        algorithms = df['algorithm'].unique()
        execution_times = []
        std_times = []
        
        for algo in algorithms:
            algo_df = df_success[df_success['algorithm'] == algo]
            if not algo_df.empty:
                execution_times.append(algo_df['execution_time'].mean())
                std_times.append(algo_df['execution_time'].std())
            else:
                execution_times.append(0)
                std_times.append(0)
        
        # Sort by execution time
        sorted_indices = np.argsort(execution_times)
        algorithms = [algorithms[i] for i in sorted_indices]
        execution_times = [execution_times[i] for i in sorted_indices]
        std_times = [std_times[i] for i in sorted_indices]
        
        plt.bar(algorithms, execution_times, yerr=std_times, capsize=5, color='skyblue')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Average Execution Time (seconds)', fontsize=14)
        plt.title(f'{benchmark_name.replace("_", " ").title()} Benchmark - Execution Time', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"benchmark_results/{benchmark_name}_execution_time.png")
        plt.close()
        
        # 2. Success Rate Comparison
        plt.figure(figsize=(12, 8))
        
        success_rates = []
        for algo in algorithms:
            algo_df = df[df['algorithm'] == algo]
            if not algo_df.empty:
                success_rates.append(100 * algo_df['is_goal_reached'].mean())
            else:
                success_rates.append(0)
        
        plt.bar(algorithms, success_rates, color='lightgreen')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.title(f'{benchmark_name.replace("_", " ").title()} Benchmark - Success Rate', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"benchmark_results/{benchmark_name}_success_rate.png")
        plt.close()
        
        # 3. Path Length Comparison
        if not df_success.empty:
            plt.figure(figsize=(12, 8))
            
            path_lengths = []
            std_lengths = []
            
            for algo in algorithms:
                algo_df = df_success[df_success['algorithm'] == algo]
                if not algo_df.empty:
                    path_lengths.append(algo_df['path_length'].mean())
                    std_lengths.append(algo_df['path_length'].std())
                else:
                    path_lengths.append(0)
                    std_lengths.append(0)
            
            plt.bar(algorithms, path_lengths, yerr=std_lengths, capsize=5, color='coral')
            plt.xlabel('Algorithm', fontsize=14)
            plt.ylabel('Average Path Length', fontsize=14)
            plt.title(f'{benchmark_name.replace("_", " ").title()} Benchmark - Path Length', fontsize=16)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(f"benchmark_results/{benchmark_name}_path_length.png")
            plt.close()
            
        # 4. Exploration Efficiency
        if not df_success.empty:
            plt.figure(figsize=(12, 8))
            
            efficiency = []
            std_efficiency = []
            
            for algo in algorithms:
                algo_df = df_success[df_success['algorithm'] == algo]
                if not algo_df.empty:
                    algo_df['efficiency'] = algo_df['nodes_visited'] / algo_df['path_length']
                    efficiency.append(algo_df['efficiency'].mean())
                    std_efficiency.append(algo_df['efficiency'].std())
                else:
                    efficiency.append(0)
                    std_efficiency.append(0)
            
            plt.bar(algorithms, efficiency, yerr=std_efficiency, capsize=5, color='lightcoral')
            plt.xlabel('Algorithm', fontsize=14)
            plt.ylabel('Nodes Visited / Path Length', fontsize=14)
            plt.title(f'{benchmark_name.replace("_", " ").title()} Benchmark - Exploration Efficiency', fontsize=16)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(f"benchmark_results/{benchmark_name}_exploration_efficiency.png")
            plt.close()
    
    def _visualize_multi_agent_results(self, df, benchmark_name):
        """Create visualizations for multi-agent benchmark results."""
        if df.empty:
            return
        
        # Filter successful runs
        df_success = df[df['is_goal_reached']]
        
        # 1. Execution Time by Agent Count
        plt.figure(figsize=(14, 10))
        
        algorithms = df['algorithm'].unique()
        agent_counts = sorted(df['agent_count'].unique())
        
        for algo in algorithms:
            times = []
            for count in agent_counts:
                algo_count_df = df_success[(df_success['algorithm'] == algo) & (df_success['agent_count'] == count)]
                if not algo_count_df.empty:
                    times.append(algo_count_df['execution_time'].mean())
                else:
                    times.append(0)
            
            plt.plot(agent_counts, times, marker='o', linewidth=2, label=algo)
        
        plt.xlabel('Number of Agents', fontsize=14)
        plt.ylabel('Average Execution Time (seconds)', fontsize=14)
        plt.title(f'{benchmark_name.replace("_", " ").title()} Benchmark - Execution Time by Agent Count', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"benchmark_results/{benchmark_name}_execution_time_by_agent_count.png")
        plt.close()
        
        # 2. Success Rate by Agent Count
        plt.figure(figsize=(14, 10))
        
        for algo in algorithms:
            rates = []
            for count in agent_counts:
                algo_count_df = df[(df['algorithm'] == algo) & (df['agent_count'] == count)]
                if not algo_count_df.empty:
                    rates.append(100 * algo_count_df['is_goal_reached'].mean())
                else:
                    rates.append(0)
            
            plt.plot(agent_counts, rates, marker='o', linewidth=2, label=algo)
        
        plt.xlabel('Number of Agents', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.title(f'{benchmark_name.replace("_", " ").title()} Benchmark - Success Rate by Agent Count', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"benchmark_results/{benchmark_name}_success_rate_by_agent_count.png")
        plt.close()
        
        # 3. Path Quality by Algorithm
        plt.figure(figsize=(12, 8))
        
        algorithms = df_success['algorithm'].unique()
        path_lengths = []
        std_lengths = []
        
        for algo in algorithms:
            algo_df = df_success[df_success['algorithm'] == algo]
            if not algo_df.empty:
                path_lengths.append(algo_df['total_path_length'].mean())
                std_lengths.append(algo_df['total_path_length'].std())
            else:
                path_lengths.append(0)
                std_lengths.append(0)
        
        plt.bar(algorithms, path_lengths, yerr=std_lengths, capsize=5, color='lightblue')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Average Total Path Length', fontsize=14)
        plt.title(f'{benchmark_name.replace("_", " ").title()} Benchmark - Path Quality', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(f"benchmark_results/{benchmark_name}_path_quality.png")
        plt.close()

    def benchmark_random_walk(self, algorithms=None, grid_sizes=None, num_trials=5):
        """Benchmark random walk algorithms on different scenarios."""
        if grid_sizes is None:
            grid_sizes = [
                (20, 20),  # Small
                (30, 30),  # Medium
                (40, 40)   # Large
            ]
        
        results = []
        
        print("Running random walk benchmarks...")
        
        # Filter for random walk algorithms
        if algorithms is None:
            algo_list = list(self.walk_algorithms.keys())
        else:
            algo_list = [a for a in algorithms if a in self.walk_algorithms]
            
        if not algo_list:
            print("  No random walk algorithms selected, skipping benchmark.")
            return []
        
        # Test on different scenario types
        for scenario_type in ["empty", "random", "maze", "bottleneck"]:
            print(f"\nTesting on {scenario_type} grids:")
            
            for rows, cols in grid_sizes:
                grid_size_name = f"{rows}x{cols}"
                print(f"  Grid size: {grid_size_name}")
                grid_size_dict = {"rows": rows, "cols": cols}
                
                for trial in range(num_trials):
                    print(f"    Trial {trial+1}/{num_trials}")
                    
                    # Generate scenario based on type
                    if scenario_type == "empty":
                        walls = [[False for _ in range(cols)] for _ in range(rows)]
                        for r in range(rows):
                            walls[r][0] = walls[r][cols-1] = True
                        for c in range(cols):
                            walls[0][c] = walls[rows-1][c] = True
                    elif scenario_type == "random":
                        walls = [[False for _ in range(cols)] for _ in range(rows)]
                        for r in range(rows):
                            walls[r][0] = walls[r][cols-1] = True
                        for c in range(cols):
                            walls[0][c] = walls[rows-1][c] = True
                        
                        # Add random internal walls with medium density
                        for r in range(1, rows-1):
                            for c in range(1, cols-1):
                                if random.random() < 0.3:  # Medium density
                                    walls[r][c] = True
                    elif scenario_type == "maze":
                        walls = self.generate_maze_with_narrow_corridors(rows, cols)
                    elif scenario_type == "bottleneck":
                        walls = self.generate_bottleneck_maze(rows, cols)
                    
                    # Generate random start and end positions
                    valid_positions = []
                    for r in range(rows):
                        for c in range(cols):
                            if not walls[r][c]:
                                valid_positions.append((r, c))
                    
                    if len(valid_positions) < 2:
                        print(f"      Warning: Not enough valid positions in trial {trial}")
                        continue
                    
                    start = random.choice(valid_positions)
                    valid_positions.remove(start)
                    end = random.choice(valid_positions)
                    
                    # Manhattan distance from start to end
                    manhattan_distance = abs(start[0] - end[0]) + abs(start[1] - end[1])
                    
                    # Run each algorithm
                    for algo in algo_list:
                        print(f"      Testing {algo}...")
                        
                        # Start memory tracking
                        tracemalloc.start()
                        
                        # Measure execution time
                        start_time = time.time()
                        
                        algorithm_func = self.walk_algorithms[algo]
                        
                        # Run with timeout
                        steps, timed_out = run_with_timeout(
                            algorithm_func, 
                            args=(start, end, grid_size_dict, walls),
                            timeout_seconds=30
                        )
                        
                        execution_time = time.time() - start_time
                        
                        # Get memory usage
                        current, peak_memory = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        
                        # Process results
                        if timed_out or steps is None:
                            result = {
                                "algorithm": algo,
                                "execution_time": 30,  # timeout value
                                "peak_memory_kb": peak_memory / 1024,
                                "path_length": float('inf'),
                                "nodes_visited": 0,
                                "is_goal_reached": False,
                                "timed_out": True
                            }
                        else:
                            # Extract metrics
                            last_step = steps[-1] if steps else None
                            is_goal_reached = last_step.get("isGoalReached", False) if last_step else False
                            nodes_visited = len(last_step["visited"]) if last_step else 0
                            
                            result = {
                                "algorithm": algo,
                                "execution_time": execution_time,
                                "peak_memory_kb": peak_memory / 1024,
                                "path_length": last_step["stepNumber"] + 1 if last_step and is_goal_reached else float('inf'),
                                "nodes_visited": nodes_visited,
                                "is_goal_reached": is_goal_reached,
                                "timed_out": False,
                                "total_steps": len(steps)
                            }
                        
                        # Add scenario info
                        result.update({
                            "scenario_type": scenario_type,
                            "grid_size": f"{rows}x{cols}",
                            "trial": trial,
                            "manhattan_distance": manhattan_distance
                        })
                        
                        results.append(result)
                        print(f"        {'✓' if result['is_goal_reached'] else '✗'} ({result['execution_time']:.3f}s)")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv("benchmark_results/random_walk_benchmark.csv", index=False)
        
        # Create visualizations
        self._visualize_random_walk_results(df)
        
        return results

    def _visualize_random_walk_results(self, df):
        """Create visualizations for random walk benchmark results."""
        if df.empty:
            return
        
        # Filter successful runs
        df_success = df[df['is_goal_reached']]
        
        # 1. Execution Time Comparison
        plt.figure(figsize=(12, 8))
        
        algorithms = df['algorithm'].unique()
        execution_times = []
        std_times = []
        
        for algo in algorithms:
            algo_df = df_success[df_success['algorithm'] == algo]
            if not algo_df.empty:
                execution_times.append(algo_df['execution_time'].mean())
                std_times.append(algo_df['execution_time'].std())
            else:
                execution_times.append(0)
                std_times.append(0)
        
        plt.bar(algorithms, execution_times, yerr=std_times, capsize=5, color='skyblue')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Average Execution Time (seconds)', fontsize=14)
        plt.title('Random Walk - Execution Time', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("benchmark_results/random_walk_execution_time.png")
        plt.close()
        
        # 2. Success Rate Comparison
        plt.figure(figsize=(12, 8))
        
        success_rates = []
        for algo in algorithms:
            algo_df = df[df['algorithm'] == algo]
            if not algo_df.empty:
                success_rates.append(100 * algo_df['is_goal_reached'].mean())
            else:
                success_rates.append(0)
        
        plt.bar(algorithms, success_rates, color='lightgreen')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.title('Random Walk - Success Rate', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("benchmark_results/random_walk_success_rate.png")
        plt.close()
        
        # 3. Memory Usage
        plt.figure(figsize=(12, 8))
        
        memory_usage = []
        std_memory = []
        
        for algo in algorithms:
            algo_df = df[df['algorithm'] == algo]
            if not algo_df.empty:
                memory_usage.append(algo_df['peak_memory_kb'].mean() / 1024)  # Convert to MB
                std_memory.append(algo_df['peak_memory_kb'].std() / 1024)
            else:
                memory_usage.append(0)
                std_memory.append(0)
        
        plt.bar(algorithms, memory_usage, yerr=std_memory, capsize=5, color='lightcoral')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Peak Memory Usage (MB)', fontsize=14)
        plt.title('Random Walk - Memory Usage', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig("benchmark_results/random_walk_memory_usage.png")
        plt.close()
        
        # 4. Performance by Scenario
        plt.figure(figsize=(14, 10))
        
        scenarios = sorted(df['scenario_type'].unique())
        
        # Plot execution time by scenario
        plt.subplot(2, 1, 1)
        
        for algo in algorithms:
            times = []
            for scenario in scenarios:
                scenario_df = df_success[(df_success['algorithm'] == algo) & (df_success['scenario_type'] == scenario)]
                if not scenario_df.empty:
                    times.append(scenario_df['execution_time'].mean())
                else:
                    times.append(0)
                    
            plt.plot(scenarios, times, marker='o', linewidth=2, label=algo)
        
        plt.xlabel('Scenario Type', fontsize=14)
        plt.ylabel('Execution Time (s)', fontsize=14)
        plt.title('Random Walk - Performance by Scenario', fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot success rate by scenario
        plt.subplot(2, 1, 2)
        
        for algo in algorithms:
            rates = []
            for scenario in scenarios:
                scenario_df = df[(df['algorithm'] == algo) & (df['scenario_type'] == scenario)]
                if not scenario_df.empty:
                    rates.append(100 * scenario_df['is_goal_reached'].mean())
                else:
                    rates.append(0)
                    
            plt.plot(scenarios, rates, marker='o', linewidth=2, label=algo)
        
        plt.xlabel('Scenario Type', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("benchmark_results/random_walk_scenario_performance.png")
        plt.close()
        
        # 5. Performance by Grid Size
        plt.figure(figsize=(14, 10))
        
        grid_sizes = sorted(df['grid_size'].unique())
        
        # Plot execution time by grid size
        plt.subplot(2, 1, 1)
        
        for algo in algorithms:
            times = []
            for size in grid_sizes:
                size_df = df_success[(df_success['algorithm'] == algo) & (df_success['grid_size'] == size)]
                if not size_df.empty:
                    times.append(size_df['execution_time'].mean())
                else:
                    times.append(0)
                    
            plt.plot(grid_sizes, times, marker='o', linewidth=2, label=algo)
        
        plt.xlabel('Grid Size', fontsize=14)
        plt.ylabel('Execution Time (s)', fontsize=14)
        plt.title('Random Walk - Performance by Grid Size', fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot success rate by grid size
        plt.subplot(2, 1, 2)
        
        for algo in algorithms:
            rates = []
            for size in grid_sizes:
                size_df = df[(df['algorithm'] == algo) & (df['grid_size'] == size)]
                if not size_df.empty:
                    rates.append(100 * size_df['is_goal_reached'].mean())
                else:
                    rates.append(0)
                    
            plt.plot(grid_sizes, rates, marker='o', linewidth=2, label=algo)
        
        plt.xlabel('Grid Size', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("benchmark_results/random_walk_grid_size_performance.png")
        plt.close()
        
        # 6. Path Length Analysis (for successful runs)
        if not df_success.empty:
            plt.figure(figsize=(12, 8))
            
            path_lengths = []
            std_lengths = []
            optimality = []
            
            for algo in algorithms:
                algo_df = df_success[df_success['algorithm'] == algo]
                if not algo_df.empty:
                    # Calculate average path length
                    path_lengths.append(algo_df['path_length'].mean())
                    std_lengths.append(algo_df['path_length'].std())
                    
                    # Calculate path optimality (path length / manhattan distance)
                    algo_df['optimality'] = algo_df['path_length'] / algo_df['manhattan_distance']
                    optimality.append(algo_df['optimality'].mean())
                else:
                    path_lengths.append(0)
                    std_lengths.append(0)
                    optimality.append(0)
            
            plt.bar(algorithms, path_lengths, yerr=std_lengths, capsize=5, color='coral')
            plt.xlabel('Algorithm', fontsize=14)
            plt.ylabel('Average Path Length', fontsize=14)
            plt.title('Random Walk - Path Length', fontsize=16)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig("benchmark_results/random_walk_path_length.png")
            plt.close()
            
            # 7. Path Optimality
            plt.figure(figsize=(12, 8))
            
            plt.bar(algorithms, optimality, color='lightblue')
            plt.xlabel('Algorithm', fontsize=14)
            plt.ylabel('Path Length / Manhattan Distance', fontsize=14)
            plt.title('Random Walk - Path Optimality', fontsize=16)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig("benchmark_results/random_walk_path_optimality.png")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run specialized benchmarks for pathfinding algorithms")
    parser.add_argument("-s", "--single", type=str, nargs="+",
                      help="Single-agent algorithms to test")
    parser.add_argument("-m", "--multi", type=str, nargs="+",
                      help="Multi-agent algorithms to test")
    parser.add_argument("-w", "--walk", type=str, nargs="+",
                      help="Random walk algorithms to test")
    parser.add_argument("-b", "--benchmark", type=str, nargs="+",
                      choices=["narrow", "bottleneck", "spiral", "altitude", "crossover", "corridor", "random_walk", "all"],
                      default=["all"],
                      help="Specific benchmarks to run")
    
    args = parser.parse_args()
    
    # Create benchmark runner
    benchmark = PathfindingBenchmark()
    
    # Map CLI algorithm names to internal names
    single_agent_map = {
        "bfs": "BFS",
        "dfs": "DFS",
        "dijkstra": "Dijkstra",
        "astar": "A*",
        "gbfs": "GBFS",
        "jps": "JPS"
    }
    
    multi_agent_map = {
        "cbs": "CBS",
        "icts": "ICTS",
        "mstar": "M*",
        "par": "PushAndRotate"
    }
    
    random_walk_map = {
        "rw": "RandomWalk",
        "brw": "BiasedRandomWalk"
    }
    
    # Process algorithm lists
    single_agent_algorithms = None
    if args.single:
        single_agent_algorithms = []
        for algo in args.single:
            if algo in single_agent_map:
                single_agent_algorithms.append(single_agent_map[algo])
            else:
                print(f"Warning: Unknown single-agent algorithm '{algo}'. Skipping.")
    
    multi_agent_algorithms = None
    if args.multi:
        multi_agent_algorithms = []
        for algo in args.multi:
            if algo in multi_agent_map:
                multi_agent_algorithms.append(multi_agent_map[algo])
            else:
                print(f"Warning: Unknown multi-agent algorithm '{algo}'. Skipping.")
    
    random_walk_algorithms = None
    if args.walk:
        random_walk_algorithms = []
        for algo in args.walk:
            if algo in random_walk_map:
                random_walk_algorithms.append(random_walk_map[algo])
            else:
                print(f"Warning: Unknown random walk algorithm '{algo}'. Skipping.")
    
    # Run benchmarks
    if "all" in args.benchmark:
        benchmark.run_all_benchmarks(single_agent_algorithms, multi_agent_algorithms)
    else:
        all_algorithms = []
        if single_agent_algorithms:
            all_algorithms.extend(single_agent_algorithms)
        if multi_agent_algorithms:
            all_algorithms.extend(multi_agent_algorithms)
            
        for bench in args.benchmark:
            if bench == "narrow":
                benchmark.benchmark_narrow_corridors(all_algorithms)
            elif bench == "bottleneck":
                benchmark.benchmark_bottleneck(all_algorithms)
            elif bench == "spiral":
                benchmark.benchmark_spiral(all_algorithms)
            elif bench == "altitude":
                benchmark.benchmark_altitude_patterns(single_agent_algorithms)
            elif bench == "crossover":
                benchmark.benchmark_agent_crossover(multi_agent_algorithms)
            elif bench == "corridor":
                benchmark.benchmark_agent_corridor(multi_agent_algorithms)
            elif bench == "random_walk":
                # Use the dedicated random walk benchmark method
                benchmark.benchmark_random_walk(random_walk_algorithms)