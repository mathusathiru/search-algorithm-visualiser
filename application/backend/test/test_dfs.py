import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
application_dir = os.path.dirname(current_dir)
sys.path.append(application_dir)

import unittest
from backend.algorithms.dfs import generate_dfs

class TestDFS(unittest.TestCase):
    def setUp(self):
        "Set up test cases with a simple 3x3 grid"
        self.grid_size = {"rows": 3, "cols": 3}
        self.start = (0, 0)
        self.end = (2, 2)
        self.walls = [[False, False, False],
                     [False, False, False],
                     [False, False, False]]

    def print_maze_path(self, positions, walls, grid_size, start, end, message=""):
        "Helper function to print maze and path for debugging"
        print(f"\n{message}")
        print("Maze configuration (S = start, E = end, X = wall, . = open, * = path):")
        visited_positions = set(positions)
        for i in range(grid_size["rows"]):
            row = ""
            for j in range(grid_size["cols"]):
                if (i, j) == start:
                    row += "S"
                elif (i, j) == end:
                    row += "E"
                elif (i, j) in visited_positions:
                    row += "*"
                else:
                    row += "X" if walls[i][j] else "."
            print(row)
        print("\nPath steps:")
        for i, pos in enumerate(positions):
            print(f"Step {i}: {pos}")

    def test_path_exists(self):
        "Test that DFS finds a path in an unobstructed grid"
        steps = generate_dfs(self.start, self.end, self.grid_size, self.walls)
        self.assertTrue(any(step["isGoalReached"] for step in steps))
        final_step = next(step for step in steps if step["isGoalReached"])
        self.assertEqual(final_step["position"], self.end)

    def test_blocked_path(self):
        "Test that DFS handles when no path exists"
        walls = [[False, False, False],
                 [False, False, False],
                 [False, False, True]]
        steps = generate_dfs(self.start, (2, 2), self.grid_size, walls)
        self.assertFalse(any(step["isGoalReached"] for step in steps))

    def test_zigzag_path(self):
        "Test DFS with a maze requiring a zigzag pattern"
        self.grid_size = {"rows": 3, "cols": 4}
        walls = [
            [False, False, True,  False],
            [True,  False, False, False],
            [False, True,  False, False]
        ]
        start = (0, 0)
        end = (2, 3)
        
        steps = generate_dfs(start, end, self.grid_size, walls)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, self.grid_size, start, end, "Zigzag Path Test")
        
        self.assertTrue(any(step["isGoalReached"] for step in steps))
        path_positions = set(positions)
        rows_visited = {pos[0] for pos in path_positions}
        self.assertEqual(len(rows_visited), 3, "Path should visit all three rows")
        self.assertGreaterEqual(len(positions), 5)

    def test_multiple_dead_ends(self):
        "Test DFS with multiple dead ends"
        self.grid_size = {"rows": 5, "cols": 5}
        walls = [
            [False, False, False, False, False],
            [False, True,  True,  True,  False],
            [False, True,  False, False, False],
            [False, True,  False, True,  False],
            [False, False, False, True,  False]
        ]
        start = (0, 0)
        end = (2, 2)
        
        steps = generate_dfs(start, end, self.grid_size, walls)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, self.grid_size, start, end, "Multiple Dead Ends Test")
        
        considered_nodes = set()
        visited_positions = set()
        for step in steps:
            considered_nodes.update(step["considered"])
            visited_positions.add(step["position"])
        
        dead_end_path = {(4, 0), (4, 1), (4, 2)}
        explored_areas = considered_nodes.union(visited_positions)
        
        print("\nConsidered nodes:", sorted(considered_nodes))
        print("Visited positions:", sorted(visited_positions))
        print("Dead end path:", sorted(dead_end_path))
        print("Explored areas:", sorted(explored_areas))
        
        explored_dead_ends = dead_end_path.intersection(explored_areas)
        self.assertGreater(len(explored_dead_ends), 0, 
                          "DFS should explore the dead-end path")

    def test_spiral_maze(self):
        "Test DFS with a spiral-shaped maze"
        self.grid_size = {"rows": 5, "cols": 5}
        walls = [
            [False, True,  True,  True,  True ],
            [False, False, False, False, True ],
            [True,  True,  True,  False, True ],
            [True,  False, False, False, True ],
            [True,  True,  True,  True,  True ]
        ]
        start = (0, 0)
        end = (3, 1)
        
        steps = generate_dfs(start, end, self.grid_size, walls)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, self.grid_size, start, end, "Spiral Maze Test")
        
        self.assertTrue(any(step["isGoalReached"] for step in steps))
        self.assertGreaterEqual(len(positions), 7)

    def test_narrow_corridor(self):
        "Test DFS with a single narrow corridor"
        self.grid_size = {"rows": 4, "cols": 5}
        walls = [
            [False, True,  False, True,  False],
            [False, True,  False, True,  False],
            [False, False, False, False, False],
            [True,  True,  True,  True,  True ]
        ]
        start = (0, 0)
        end = (0, 4)
        
        steps = generate_dfs(start, end, self.grid_size, walls)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, self.grid_size, start, end, "Narrow Corridor Test")
        
        self.assertTrue(any(step["isGoalReached"] for step in steps))
        path_positions = set(positions)
        corridor_points = {(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)}
        self.assertTrue(any(point in path_positions for point in corridor_points))

    def test_long_detour(self):
        "Test DFS when the only valid path requires a long detour"
        self.grid_size = {"rows": 4, "cols": 3}
        walls = [
            [False, True,  False],
            [False, True,  False],
            [False, True,  False],
            [False, False, False]
        ]
        start = (0, 0)
        end = (0, 2)
        
        steps = generate_dfs(start, end, self.grid_size, walls)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, self.grid_size, start, end, "Long Detour Test")
        
        self.assertTrue(any(step["isGoalReached"] for step in steps))
        manhattan_distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
        self.assertGreater(len(positions), manhattan_distance)

    def test_maze_edge_path(self):
        "Test DFS with path along the maze boundaries"
        self.grid_size = {"rows": 3, "cols": 4}
        walls = [
            [False, True,  True,  True ],
            [False, True,  True,  True ],
            [False, False, False, False]
        ]
        start = (0, 0)
        end = (2, 3)
        
        steps = generate_dfs(start, end, self.grid_size, walls)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, self.grid_size, start, end, "Edge Path Test")
        
        self.assertTrue(any(step["isGoalReached"] for step in steps))
        edge_positions = {(0, 0), (1, 0), (2, 0)}
        path_positions = set(positions)
        self.assertTrue(edge_positions.issubset(path_positions))

if __name__ == "__main__":
    unittest.main()