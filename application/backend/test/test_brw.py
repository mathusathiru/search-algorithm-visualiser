import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
application_dir = os.path.dirname(current_dir)
sys.path.append(application_dir)

import unittest
from collections import defaultdict
from application.backend.algorithms.brw import generate_biased_random_walk

class TestBiasedRandomWalk(unittest.TestCase):
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

    def calculate_direction_bias(self, positions, end):
        "Helper function to calculate if movements tend toward the goal"
        moves_toward_goal = 0
        moves_away_from_goal = 0
        
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            
            prev_distance = abs(end[0] - prev_pos[0]) + abs(end[1] - prev_pos[1])
            curr_distance = abs(end[0] - curr_pos[0]) + abs(end[1] - curr_pos[1])
            
            if curr_distance < prev_distance:
                moves_toward_goal += 1
            elif curr_distance > prev_distance:
                moves_away_from_goal += 1
                
        return moves_toward_goal, moves_away_from_goal

    def test_biased_movement(self):
        "Test that movement tends toward the goal more often than away"
        total_toward = 0
        total_away = 0
        num_trials = 10
        
        print("\nBias Analysis across multiple runs:")
        for i in range(num_trials):
            steps = generate_biased_random_walk(self.start, self.end, self.grid_size, 
                                             self.walls, self.max_steps)
            positions = [step["position"] for step in steps]
            toward, away = self.calculate_direction_bias(positions, self.end)
            total_toward += toward
            total_away += away
            print(f"Run {i+1}: {toward} moves toward goal, {away} moves away")
            
        print(f"\nTotal: {total_toward} toward vs {total_away} away from goal")
        self.assertGreater(total_toward, total_away, 
                          "Should move toward goal more often than away")

    def test_reaches_goal_efficiently(self):
        "Test that biased random walk typically reaches goal in fewer steps than pure random"
        successful_paths = []
        num_trials = 10
        
        print("\nPath Length Analysis:")
        for i in range(num_trials):
            steps = generate_biased_random_walk(self.start, self.end, self.grid_size, 
                                             self.walls, self.max_steps)
            positions = [step["position"] for step in steps]
            if any(step["isGoalReached"] for step in steps):
                successful_paths.append(len(positions))
                print(f"Run {i+1}: Reached goal in {len(positions)} steps")
            else:
                print(f"Run {i+1}: Did not reach goal")
        
        if successful_paths:
            avg_path_length = sum(successful_paths) / len(successful_paths)
            print(f"\nAverage successful path length: {avg_path_length:.2f} steps")
            self.assertLess(avg_path_length, self.max_steps / 2, 
                          "Should reach goal in reasonable number of steps")

    def test_handles_obstacles(self):
        "Test that biased movement still works with obstacles"
        walls = [[False, True, False],
                [False, True, False],
                [False, False, False]]
        steps = generate_biased_random_walk(self.start, self.end, self.grid_size, 
                                          walls, self.max_steps)
        positions = [step["position"] for step in steps]
        self.print_maze_path(positions, walls, self.grid_size, 
                           self.start, self.end, "Obstacle Navigation Test")
        
        for pos in positions:
            self.assertFalse(walls[pos[0]][pos[1]], "Should never enter wall positions")

    def test_revisit_patterns(self):
        "Test that positions closer to the goal tend to be revisited more"
        steps = generate_biased_random_walk(self.start, self.end, self.grid_size, 
                                          self.walls, self.max_steps)
        
        visit_counts = defaultdict(int)
        for step in steps:
            pos = step["position"]
            visit_counts[pos] += 1
            
        distance_visits = defaultdict(list)
        for pos, count in visit_counts.items():
            distance = abs(self.end[0] - pos[0]) + abs(self.end[1] - pos[1])
            distance_visits[distance].append(count)
            
        print("\nVisit count analysis by distance from goal:")
        for distance, counts in sorted(distance_visits.items()):
            avg_visits = sum(counts) / len(counts)
            print(f"Distance {distance} from goal: average {avg_visits:.2f} visits")

    def test_movement_is_valid(self):
        "Test that each move is valid (only moves to adjacent cells)"
        steps = generate_biased_random_walk(self.start, self.end, self.grid_size, 
                                          self.walls, self.max_steps)
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