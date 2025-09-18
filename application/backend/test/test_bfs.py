import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
application_dir = os.path.dirname(current_dir)
sys.path.append(application_dir)

import unittest
from backend.algorithms.bfs import generate_bfs

class TestBFS(unittest.TestCase):
    def setUp(self):
        self.grid_size = {"rows": 3, "cols": 3}
        self.start = (0, 0)
        self.end = (2, 2)
        self.walls = [[False, False, False],
                     [False, False, False],
                     [False, False, False]]     

    def test_path_exists(self):
        steps = generate_bfs(self.start, self.end, self.grid_size, self.walls)
        self.assertTrue(any(step["isGoalReached"] for step in steps))
        final_step = next(step for step in steps if step["isGoalReached"])
        self.assertEqual(final_step["position"], self.end)

    def test_blocked_path(self):
        walls = [[False, False, False],
                 [False, False, False],
                 [False, False, True]]
        steps = generate_bfs(self.start, (2, 2), self.grid_size, walls)
        self.assertFalse(any(step["isGoalReached"] for step in steps))

    def test_visit_order(self):
        steps = generate_bfs(self.start, self.end, self.grid_size, self.walls)
        first_step = steps[0]
        self.assertEqual(first_step["position"], self.start)
        self.assertEqual(first_step["visitCounts"][str(self.start)], 1)

    def test_valid_steps_format(self):
        steps = generate_bfs(self.start, self.end, self.grid_size, self.walls)
        first_step = steps[0]
        
        required_keys = {"position", "visited", "visitCounts", "isGoalReached", "stepNumber"}
        self.assertTrue(all(key in first_step for key in required_keys))
        
        self.assertIsInstance(first_step["position"], tuple)
        self.assertEqual(len(first_step["position"]), 2)

    def test_boundary_conditions(self):
        start = (2, 2)
        end = (0, 0)
        steps = generate_bfs(start, end, self.grid_size, self.walls)
        self.assertTrue(any(step["isGoalReached"] for step in steps))
        
        self.assertEqual(steps[0]["position"], start)
        final_step = next(step for step in steps if step["isGoalReached"])
        self.assertEqual(final_step["position"], end)

if __name__ == "__main__":
    unittest.main()