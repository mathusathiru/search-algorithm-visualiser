#!/usr/bin/env python3
# Run specific tests on selected algorithms and scenarios

import argparse
import time
from pathfinder_tester import PathfindingTester

def run_single_algorithm_test(algorithm, grid_size, scenario_type, density=None, agent_count=None):
    """Run tests for a specific algorithm and scenario."""
    print(f"Running tests for {algorithm} on {scenario_type} {grid_size} grid...")
    
    tester = PathfindingTester()
    
    # Determine algorithm type
    is_single_agent = algorithm in tester.single_agent_algorithms
    is_multi_agent = algorithm in tester.multi_agent_algorithms
    is_walk_algorithm = algorithm in tester.walk_algorithms
    
    if is_walk_algorithm or is_single_agent:
        if scenario_type == "random" and density is None:
            density = "medium"
        
        # Use walk_tests for walk algorithms, single_agent_tests for other single-agent algorithms
        test_method = (tester.run_random_walk_tests if is_walk_algorithm 
                       else tester.run_single_agent_tests)
        
        results = test_method(
            scenario_type, 
            grid_size, 
            wall_density=density, 
            num_trials=10
        )
        
        # Filter results for the specific algorithm
        results = [r for r in results if r["algorithm"] == algorithm]
    elif is_multi_agent:
        if agent_count is None:
            agent_count = 3
            
        results = tester.run_multi_agent_tests(
            scenario_type, 
            grid_size, 
            agent_count=agent_count, 
            num_trials=5
        )
        
        # Filter results for the specific algorithm
        results = [r for r in results if r["algorithm"] == algorithm]
    else:
        print(f"Error: Algorithm {algorithm} not recognized.")
        return []
    
    # Compute and print statistics
    successful_runs = [r for r in results if r["is_goal_reached"]]
    success_rate = len(successful_runs) / len(results) if results else 0
    
    avg_time = sum(r["execution_time"] for r in results) / len(results) if results else 0
    avg_memory = sum(r["peak_memory_kb"] for r in results) / len(results) if results else 0
    
    if is_walk_algorithm or is_single_agent:
        avg_path_length = sum(r["path_length"] for r in successful_runs) / len(successful_runs) if successful_runs else float('inf')
        avg_nodes_visited = sum(r["nodes_visited"] for r in results) / len(results) if results else 0
        print(f"\nResults for {algorithm} on {scenario_type} {grid_size} grid:")
        print(f"  Success Rate: {success_rate * 100:.1f}%")
        print(f"  Average Execution Time: {avg_time:.4f} seconds")
        print(f"  Average Memory Usage: {avg_memory / 1024:.2f} MB")
        print(f"  Average Path Length: {avg_path_length:.2f}")
        print(f"  Average Nodes Visited: {avg_nodes_visited:.2f}")
    else:
        avg_path_length = sum(r["total_path_length"] for r in successful_runs) / len(successful_runs) if successful_runs else float('inf')
        print(f"\nResults for {algorithm} on {scenario_type} {grid_size} grid with {agent_count} agents:")
        print(f"  Success Rate: {success_rate * 100:.1f}%")
        print(f"  Average Execution Time: {avg_time:.4f} seconds")
        print(f"  Average Memory Usage: {avg_memory / 1024:.2f} MB")
        print(f"  Average Total Path Length: {avg_path_length:.2f}")
    
    return results

def run_algorithm_comparison(algorithms, grid_size, scenario_type, density=None, agent_count=None):
    """Run comparison tests between multiple algorithms."""
    tester = PathfindingTester()
    all_results = []
    
    # Categorize algorithms
    single_agent_algos = [a for a in algorithms if a in tester.single_agent_algorithms]
    multi_agent_algos = [a for a in algorithms if a in tester.multi_agent_algorithms]
    walk_algos = [a for a in algorithms if a in tester.walk_algorithms]
    
    # Run tests for each algorithm category
    for algorithm in single_agent_algos:
        results = run_single_algorithm_test(algorithm, grid_size, scenario_type, density)
        all_results.extend(results)
    
    for algorithm in walk_algos:
        results = run_single_algorithm_test(algorithm, grid_size, scenario_type, density)
        all_results.extend(results)
    
    for algorithm in multi_agent_algos:
        results = run_single_algorithm_test(algorithm, grid_size, scenario_type, density, agent_count)
        all_results.extend(results)
    
    # Save the comparison results
    if all_results:
        import pandas as pd
        output_file = f"test_results/comparison_{scenario_type}_{grid_size}_{int(time.time())}.csv"
        pd.DataFrame(all_results).to_csv(output_file, index=False)
        print(f"Comparison results saved to {output_file}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Run tests for specific pathfinding algorithms")
    parser.add_argument("-a", "--algorithm", type=str, nargs="+", required=True,
                      help="Algorithm(s) to test (e.g., BFS, A*, CBS)")
    
    parser.add_argument("-g", "--grid-size", type=str, default="medium",
                      choices=["small", "medium", "large"],
                      help="Grid size for testing")
    
    parser.add_argument("-s", "--scenario", type=str, default="random",
                      choices=["empty", "random", "maze", "bottleneck", "multi_scenario"],
                      help="Scenario type for testing")
    
    parser.add_argument("-d", "--density", type=str,
                      choices=["sparse", "medium", "dense"],
                      help="Wall density for random scenarios")
    
    parser.add_argument("-c", "--agent-count", type=int, default=3,
                      help="Number of agents for multi-agent scenarios")
    
    args = parser.parse_args()
    
    # Dictionary of valid algorithm names
    tester = PathfindingTester()
    valid_algorithms = {
        # Single-agent algorithms
        "BFS": "BFS",
        "DFS": "DFS",
        "Dijkstra": "Dijkstra",
        "A*": "A*",
        "GBFS": "GBFS",
        "JPS": "JPS",
        
        # Random walk algorithms
        "RandomWalk": "RandomWalk",
        "BiasedRandomWalk": "BiasedRandomWalk",
        "RW": "RandomWalk",
        "BRW": "BiasedRandomWalk",
        
        # Multi-agent algorithms
        "CBS": "CBS",
        "ICTS": "ICTS",
        "M*": "M*",
        "PushAndRotate": "PushAndRotate",
        
        # Aliases
        "astar": "A*",
        "bfs": "BFS",
        "dfs": "DFS",
        "dijkstra": "Dijkstra",
        "gbfs": "GBFS",
        "jps": "JPS",
        "cbs": "CBS",
        "icts": "ICTS",
        "mstar": "M*",
        "par": "PushAndRotate",
        "rw": "RandomWalk",
        "brw": "BiasedRandomWalk"
    }
    
    # Validate and normalize algorithm names
    algorithms = []
    for algo in args.algorithm:
        normalized_algo = algo.strip()
        if normalized_algo.upper() in valid_algorithms:
            algorithms.append(valid_algorithms[normalized_algo.upper()])
        else:
            print(f"Warning: Unknown algorithm '{algo}'. Skipping.")
    
    if not algorithms:
        print("Error: No valid algorithms specified")
        return
    
    # If multiple algorithms, run comparison
    if len(algorithms) > 1:
        run_algorithm_comparison(
            algorithms, 
            args.grid_size, 
            args.scenario, 
            args.density, 
            args.agent_count
        )
    else:
        # Single algorithm test
        run_single_algorithm_test(
            algorithms[0], 
            args.grid_size, 
            args.scenario, 
            args.density, 
            args.agent_count
        )

if __name__ == "__main__":
    main()