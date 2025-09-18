import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
application_dir = os.path.dirname(current_dir)
sys.path.append(application_dir)

import unittest
from backend.algorithms.astar import generate_astar

class TestAStar(unittest.TestCase):
    def setUp(self):
        """Set up test cases with a simple 3x3 grid"""
        self.grid_size = {"rows": 3, "cols": 3}
        self.start = (0, 0)
        self.end = (2, 2)
        self.walls = [[False, False, False],
                     [False, False, False],
                     [False, False, False]]
        self.altitudes = [[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]]

    def print_maze_path(self, positions, walls, altitudes, grid_size, start, end, message=""):
        """Helper function to print maze and path for debugging"""
        print(f"\n{message}")
        print("Maze configuration (S = start, E = end, X = wall, numbers = costs, * = path):")
        visited_positions = set(positions)
        for i in range(grid_size["rows"]):
            row = ""
            for j in range(grid_size["cols"]):
                if (i, j) == start:
                    row += "S"
                elif (i, j) == end:
                    row += "E"
                elif walls[i][j]:
                    row += "X"
                elif (i, j) in visited_positions:
                    row += "*"
                else:
                    row += str(altitudes[i][j])
            print(row)
        print("\nPath steps with costs:")
        total_cost = 0
        for i, pos in enumerate(positions):
            cost = altitudes[pos[0]][pos[1]]
            total_cost += cost
            print(f"Step {i}: {pos} (cost: {cost})")
        print(f"Total path cost: {total_cost}")

    def count_explored_nodes(self, steps):
        """Helper function to count total nodes explored before finding goal"""
        explored = set()
        for step in steps:
            explored.add(step["position"])
            if step["isGoalReached"]:
                break
        return len(explored)

    def test_basic_path(self):
        """Test that A* finds a path in an unobstructed grid with uniform costs"""
        steps = generate_astar(self.start, self.end, self.grid_size, self.walls, self.altitudes)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, self.walls, self.altitudes, self.grid_size, 
                           self.start, self.end, "Basic Path Test")
        
        self.assertTrue(any(step["isGoalReached"] for step in steps))
        final_step = next(step for step in steps if step["isGoalReached"])
        self.assertEqual(final_step["position"], self.end)

    def test_blocked_path(self):
        """Test that A* handles when no path exists"""
        walls = [[False, False, False],
                [False, True, True],
                [False, True, False]]
        steps = generate_astar(self.start, self.end, self.grid_size, walls, self.altitudes)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, self.altitudes, self.grid_size, 
                           self.start, self.end, "Blocked Path Test")
        
        self.assertFalse(any(step["isGoalReached"] for step in steps))

    def test_optimal_path_costs(self):
        """Test that A* finds the lowest cost path when multiple paths exist"""
        altitudes = [
            [1, 8, 1],  
            [1, 9, 8], 
            [1, 1, 1]
        ]
        steps = generate_astar(self.start, self.end, self.grid_size, self.walls, altitudes)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, self.walls, altitudes, self.grid_size, 
                           self.start, self.end, "Optimal Path Cost Test")

        path_cost = sum(altitudes[pos[0]][pos[1]] for pos in positions)
        self.assertLessEqual(path_cost, 5, "Path cost should be optimal")

    def test_heuristic_efficiency(self):
        """Test that A* explores fewer nodes than an exhaustive search would"""
        self.grid_size = {"rows": 8, "cols": 8}
        
        altitudes = [[1] * 8 for _ in range(8)]
        
        for i in range(8):
            for j in range(8):
                if i < 2 or i > 5 or j < 2 or j > 5:
                    altitudes[i][j] = 10
                if (i == 3 or i == 4) and (j >= 2):
                    altitudes[i][j] = 1
        
        walls = [[False] * 8 for _ in range(8)]
        start = (3, 0) 
        end = (4, 7)
        
        steps = generate_astar(start, end, self.grid_size, walls, altitudes)
        positions = [step["position"] for step in steps]
        
        self.print_maze_path(positions, walls, altitudes, self.grid_size,
                           start, end, "Heuristic Efficiency Test")
        
        nodes_explored = self.count_explored_nodes(steps)
        total_cells = self.grid_size["rows"] * self.grid_size["cols"]
        
        print(f"\nExploration Statistics:")
        print(f"Total cells in grid: {total_cells}")
        print(f"Nodes explored by A*: {nodes_explored}")
        print(f"Exploration percentage: {(nodes_explored/total_cells)*100:.1f}%")
        
        self.assertLess(nodes_explored, 24,
                       "A* should mainly explore the low-cost valley")
        
        valley_positions = {(3,j) for j in range(8)} | {(4,j) for j in range(8)}
        path_positions = set(positions)
        valley_cells_used = len(valley_positions & path_positions)
        self.assertGreater(valley_cells_used, 5,
                          "Path should utilize the low-cost valley")

    def test_uniform_cost_comparison(self):
        """Test that with uniform costs, A* finds the shortest path"""
        altitudes = [[1] * 3 for _ in range(3)]
        steps = generate_astar(self.start, self.end, self.grid_size, self.walls, altitudes)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, self.walls, altitudes, self.grid_size,
                           self.start, self.end, "Uniform Cost Test")
        
        manhattan_distance = abs(self.end[0] - self.start[0]) + abs(self.end[1] - self.start[1])
        final_step = next(step for step in steps if step["isGoalReached"])
        path = final_step.get("path", positions)
        self.assertEqual(len(path) - 1, manhattan_distance,
                        "With uniform costs, path should have minimum length")

if __name__ == '__main__':
    unittest.main()