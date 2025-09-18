import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
application_dir = os.path.dirname(current_dir)
sys.path.append(application_dir)

import unittest
from backend.algorithms.dijkstra import generate_dijkstra

class TestDijkstra(unittest.TestCase):
    def setUp(self):
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

    def test_basic_path(self):
        steps = generate_dijkstra(self.start, self.end, self.grid_size, self.walls, self.altitudes)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, self.walls, self.altitudes, self.grid_size, 
                           self.start, self.end, "Basic Path Test")
        
        self.assertTrue(any(step["isGoalReached"] for step in steps))
        final_step = next(step for step in steps if step["isGoalReached"])
        self.assertEqual(final_step["position"], self.end)

    def test_blocked_path(self):
        walls = [[False, False, False],
                [False, True, True],
                [False, True, False]]
        steps = generate_dijkstra(self.start, self.end, self.grid_size, walls, self.altitudes)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, self.altitudes, self.grid_size, 
                           self.start, self.end, "Blocked Path Test")
        
        self.assertFalse(any(step["isGoalReached"] for step in steps))

    def test_optimal_path_costs(self):
        altitudes = [
            [1, 8, 1], 
            [1, 9, 8],
            [1, 1, 1]
        ]
        steps = generate_dijkstra(self.start, self.end, self.grid_size, self.walls, altitudes)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, self.walls, altitudes, self.grid_size, 
                           self.start, self.end, "Optimal Path Cost Test")

        path_cost = sum(altitudes[pos[0]][pos[1]] for pos in positions)
        
        self.assertLessEqual(path_cost, 5, "Path cost should be optimal")

    def test_high_cost_shortcut(self):
        self.grid_size = {"rows": 4, "cols": 3}
        altitudes = [
            [1,  1,  1],
            [10, 3,  10],
            [10, 20, 10],
            [10, 20, 10]
        ]
        walls = [[False] * 3 for _ in range(4)]
        start = (0, 0)
        end = (2, 2)
        
        steps = generate_dijkstra(start, end, self.grid_size, walls, altitudes)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, altitudes, self.grid_size, 
                           start, end, "High Cost Shortcut Test")
        
        shortcut_position = (1, 1)
        path_positions = set(positions)
        self.assertIn(shortcut_position, path_positions,
                     "Path should take the shortcut through the cheaper middle cell")

    def test_uniform_cost_comparison(self):
        altitudes = [[1] * 3 for _ in range(3)]
        steps = generate_dijkstra(self.start, self.end, self.grid_size, self.walls, altitudes)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, self.walls, altitudes, self.grid_size, 
                           self.start, self.end, "Uniform Cost Test")
        
        manhattan_distance = abs(self.end[0] - self.start[0]) + abs(self.end[1] - self.start[1])
        final_step = next(step for step in steps if step["isGoalReached"])
        path = final_step.get("path", positions)
        self.assertEqual(len(path) - 1, manhattan_distance,
                        "With uniform costs, path should have minimum length")

if __name__ == "__main__":
    unittest.main()