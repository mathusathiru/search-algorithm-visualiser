from pathfinder_tester import PathfindingTester
import os
import pandas as pd

# Create the tester
tester = PathfindingTester()

# Check if CSV files already exist and load them
if os.path.exists("test_results/random_walk_results.csv"):
    print("Found existing random walk results, loading...")
    random_walk_df = pd.read_csv("test_results/random_walk_results.csv")
    tester.results = {"random_walk": random_walk_df.to_dict('records')}

# Run random walk tests first
print("Running random walk tests first...")
tester.run_comprehensive_tests(random_walk_only=True)

# Then run single-agent tests
print("\nRunning single-agent tests next...")
tester.run_comprehensive_tests(single_agent_only=True)

# Generate visualizations for all results
print("\nGenerating visualizations...")
tester.visualize_results()

print("Testing complete. Results and visualizations saved to the test_results directory.")