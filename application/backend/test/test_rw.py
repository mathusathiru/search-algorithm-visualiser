import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
application_dir = os.path.dirname(current_dir)
sys.path.append(application_dir)

import unittest
from application.backend.algorithms.rw import generate_random_walk

class TestRandomWalk(unittest.TestCase):
    def setUp(self):
        "Set up test cases with a simple 3x3 grid"
        self.grid_size = {"rows": 3, "cols": 3}
        self.start = (0, 0)
        self.end = (2, 2)
        self.walls = [[False, False, False],
                     [False, False, False],
                     [False, False, False]]
        self.max_steps = 100

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
                elif walls[i][j]:
                    row += "X"
                elif (i, j) in visited_positions:
                    row += "*"
                else:
                    row += "."
            print(row)
        print("\nPath steps:")
        for i, pos in enumerate(positions):
            print(f"Step {i}: {pos}")

    def test_stays_within_bounds(self):
        "Test that random walk never leaves the grid boundaries"
        steps = generate_random_walk(self.start, self.end, self.grid_size, self.walls, self.max_steps)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, self.walls, self.grid_size, 
                           self.start, self.end, "Stay Within Bounds Test")

        for pos in positions:
            self.assertGreaterEqual(pos[0], 0, "Should not go above the grid")
            self.assertLess(pos[0], self.grid_size["rows"], "Should not go below the grid")
            self.assertGreaterEqual(pos[1], 0, "Should not go left of the grid")
            self.assertLess(pos[1], self.grid_size["cols"], "Should not go right of the grid")

    def test_respects_walls(self):
        "Test that random walk never moves through walls"
        walls = [[False, True, False],
                [False, True, False],
                [False, False, False]]
        steps = generate_random_walk(self.start, self.end, self.grid_size, walls, self.max_steps)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, self.grid_size, 
                           self.start, self.end, "Respect Walls Test")

        for pos in positions:
            self.assertFalse(walls[pos[0]][pos[1]], "Should never enter wall positions")

    def test_starts_correctly(self):
        "Test that random walk always starts at the start position"
        steps = generate_random_walk(self.start, self.end, self.grid_size, self.walls, self.max_steps)
        self.assertEqual(steps[0]["position"], self.start, "Should start at start position")

    def test_recognizes_goal(self):
        "Test that random walk recognizes when it reaches the goal"
        for _ in range(5):
            steps = generate_random_walk(self.start, self.end, self.grid_size, self.walls, self.max_steps)
            positions = [step["position"] for step in steps]
            
            if self.end in positions:
                end_step = next(step for step in steps if step["position"] == self.end)
                self.assertTrue(end_step["isGoalReached"], 
                              "Should mark step as goal reached when at end position")
                end_index = positions.index(self.end)
                self.assertEqual(len(positions), end_index + 1, 
                               "Should stop after reaching goal")

    def test_visit_counts(self):
        "Test that visit counts are properly tracked"
        steps = generate_random_walk(self.start, self.end, self.grid_size, self.walls, self.max_steps)
        
        manual_counts = {}
        for step in steps:
            pos_str = str(step["position"])
            manual_counts[pos_str] = manual_counts.get(pos_str, 0) + 1
        
        final_counts = steps[-1]["visitCounts"]
        self.assertEqual(manual_counts, final_counts, 
                        "Visit counts should be accurately tracked")

    def test_respects_max_steps(self):
        "Test that random walk stops after max_steps if goal not reached"
        max_steps = 10
        steps = generate_random_walk(self.start, self.end, self.grid_size, self.walls, max_steps)
        self.assertLessEqual(len(steps), max_steps, 
                           f"Should not exceed {max_steps} steps")

    def test_movement_is_valid(self):
        "Test that each move is valid (only moves to adjacent cells)"
        steps = generate_random_walk(self.start, self.end, self.grid_size, self.walls, self.max_steps)
        positions = [step["position"] for step in steps]
        
        for i in range(1, len(positions)):
            prev = positions[i-1]
            curr = positions[i]
            diff_row = abs(curr[0] - prev[0])
            diff_col = abs(curr[1] - prev[1])
            self.assertTrue(
                (diff_row == 1 and diff_col == 0) or (diff_row == 0 and diff_col == 1),
                f"Invalid move from {prev} to {curr}"
            )

if __name__ == "__main__":
    unittest.main()