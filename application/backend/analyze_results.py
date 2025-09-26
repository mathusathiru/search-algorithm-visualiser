#!/usr/bin/env python3
# Perform in-depth analysis on pathfinder test results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


class ResultAnalyzer:
    """Analyze and visualize pathfinding algorithm test results."""

    def __init__(self, single_agent_file=None, multi_agent_file=None):
        """Initialize the analyzer with results files."""
        self.single_agent_data = None
        self.multi_agent_data = None
        
        if single_agent_file and os.path.exists(single_agent_file):
            self.single_agent_data = pd.read_csv(single_agent_file)
            print(f"Loaded single-agent data with {len(self.single_agent_data)} records")
        
        if multi_agent_file and os.path.exists(multi_agent_file):
            self.multi_agent_data = pd.read_csv(multi_agent_file)
            print(f"Loaded multi-agent data with {len(self.multi_agent_data)} records")
        
        # Create output directory
        os.makedirs("analysis_results", exist_ok=True)
    
    def run_full_analysis(self):
        """Run a comprehensive analysis on all loaded data."""
        if self.single_agent_data is not None:
            print("Analyzing single-agent results...")
            self.analyze_single_agent_data()
        
        if self.multi_agent_data is not None:
            print("Analyzing multi-agent results...")
            self.analyze_multi_agent_data()
        
        if self.single_agent_data is not None and self.multi_agent_data is not None:
            print("Performing comparative analysis...")
            self.comparative_analysis()
    
    def analyze_single_agent_data(self):
        """Perform detailed analysis on single-agent results."""
        df = self.single_agent_data
        
        # 1. Statistical summary by algorithm
        print("\nSingle-agent algorithm statistics:")
        algorithm_stats = df.groupby('algorithm').agg({
            'execution_time': ['mean', 'std', 'min', 'max'],
            'peak_memory_kb': ['mean', 'std', 'min', 'max'],
            'path_length': ['mean', 'std', 'min', 'max'],
            'nodes_visited': ['mean', 'std', 'min', 'max'],
            'is_goal_reached': ['mean', 'count']
        })
        
        # Rename columns for better readability
        algorithm_stats.columns = ['_'.join(col).strip() for col in algorithm_stats.columns.values]
        algorithm_stats = algorithm_stats.rename(columns={'is_goal_reached_mean': 'success_rate'})
        
        # Save the summary
        algorithm_stats.to_csv("analysis_results/single_agent_algorithm_stats.csv")
        print(algorithm_stats[['execution_time_mean', 'peak_memory_kb_mean', 'path_length_mean', 'success_rate']])
        
        # 2. Correlation analysis
        df_success = df[df['is_goal_reached']]
        correlation = df_success[['execution_time', 'peak_memory_kb', 'path_length', 'nodes_visited']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Matrix for Successful Runs')
        plt.tight_layout()
        plt.savefig("analysis_results/single_agent_correlation.png")
        plt.close()
        
        # 3. Algorithm efficiency metric
        df_success['efficiency_ratio'] = df_success['nodes_visited'] / df_success['path_length']
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='algorithm', y='efficiency_ratio', data=df_success)
        plt.title('Algorithm Efficiency (Nodes Visited per Path Unit)')
        plt.xlabel('Algorithm')
        plt.ylabel('Nodes Visited / Path Length')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("analysis_results/single_agent_efficiency_boxplot.png")
        plt.close()
        
        # 4. Path optimality analysis
        # Calculate how close paths are to optimal (Manhattan distance)
        df_success['optimality_ratio'] = df_success['path_length'] / df_success['manhattan_distance']
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='algorithm', y='optimality_ratio', data=df_success)
        plt.title('Path Optimality (Path Length / Manhattan Distance)')
        plt.xlabel('Algorithm')
        plt.ylabel('Path Length / Manhattan Distance')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("analysis_results/single_agent_optimality_boxplot.png")
        plt.close()
        
        # 5. Performance across different scenarios
        scenario_perf = df.groupby(['algorithm', 'scenario_type']).agg({
            'execution_time': 'mean',
            'is_goal_reached': 'mean'
        }).reset_index()

        # Plot execution time by scenario
        plt.figure(figsize=(14, 10))
        g = sns.catplot(
            x='algorithm', y='execution_time', hue='scenario_type',
            data=scenario_perf, kind='bar', height=6, aspect=2
        )
        g.set_xticklabels(rotation=45)
        plt.title('Execution Time by Algorithm and Scenario Type')
        plt.tight_layout()
        plt.savefig("analysis_results/single_agent_scenario_performance.png")
        plt.close()

        # 6. Statistical significance testing
        # Perform ANOVA to see if algorithm choice significantly affects execution time
        model = ols('execution_time ~ algorithm', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        with open("analysis_results/single_agent_anova.txt", "w") as f:
            f.write("ANOVA results for execution time by algorithm:\n")
            f.write(str(anova_table))
            
            # Add post-hoc t-tests if ANOVA is significant
            if anova_table['PR(>F)'][0] < 0.05:
                f.write("\n\nPost-hoc t-tests:\n")
                algorithms = df['algorithm'].unique()
                
                for i, algo1 in enumerate(algorithms):
                    for algo2 in algorithms[i + 1:]:
                        group1 = df[df['algorithm'] == algo1]['execution_time']
                        group2 = df[df['algorithm'] == algo2]['execution_time']
                        
                        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                        f.write(f"{algo1} vs {algo2}: t={t_stat:.4f}, p={p_val:.4f}\n")
        
        # 7. Analysis of execution time vs grid size
        if 'grid_size' in df.columns:
            grid_perf = df.groupby(['algorithm', 'grid_size']).agg({
                'execution_time': 'mean'
            }).reset_index()

            # Convert grid_size to categorical with proper order
            grid_perf['grid_size'] = pd.Categorical(
                grid_perf['grid_size'],
                categories=['small', 'medium', 'large'], 
                ordered=True
            )
            
            # Sort by grid_size
            grid_perf = grid_perf.sort_values(['algorithm', 'grid_size'])
            
            # Calculate scaling factors
            grid_perf_pivot = grid_perf.pivot(index='algorithm', columns='grid_size', values='execution_time')
            grid_perf_pivot['small_to_medium'] = grid_perf_pivot['medium'] / grid_perf_pivot['small']
            grid_perf_pivot['medium_to_large'] = grid_perf_pivot['large'] / grid_perf_pivot['medium']
            
            grid_perf_pivot.to_csv("analysis_results/single_agent_grid_scaling.csv")
            
            # Visualize the scaling behavior
            plt.figure(figsize=(14, 8))
            
            for algorithm in grid_perf['algorithm'].unique():
                alg_data = grid_perf[grid_perf['algorithm'] == algorithm]
                plt.plot(alg_data['grid_size'], alg_data['execution_time'], marker='o', label=algorithm)
            
            plt.xlabel('Grid Size')
            plt.ylabel('Execution Time (s)')
            plt.title('Algorithm Scaling with Grid Size')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("analysis_results/single_agent_grid_scaling_plot.png")
            plt.close()
    
    def analyze_multi_agent_data(self):
        """Perform detailed analysis on multi-agent results."""
        if self.multi_agent_data is None:
            return
            
        df = self.multi_agent_data
        
        # 1. Statistical summary by algorithm
        print("\nMulti-agent algorithm statistics:")
        algorithm_stats = df.groupby('algorithm').agg({
            'execution_time': ['mean', 'std', 'min', 'max'],
            'peak_memory_kb': ['mean', 'std', 'min', 'max'],
            'total_path_length': ['mean', 'std', 'min', 'max'],
            'nodes_visited': ['mean', 'std', 'min', 'max'],
            'is_goal_reached': ['mean', 'count']
        })
        
        # Rename columns for better readability
        algorithm_stats.columns = ['_'.join(col).strip() for col in algorithm_stats.columns.values]
        algorithm_stats = algorithm_stats.rename(columns={'is_goal_reached_mean': 'success_rate'})
        
        # Save the summary
        algorithm_stats.to_csv("analysis_results/multi_agent_algorithm_stats.csv")
        print(algorithm_stats[['execution_time_mean', 'peak_memory_kb_mean', 'total_path_length_mean', 'success_rate']])
        
        # 2. Correlation analysis
        df_success = df[df['is_goal_reached']]
        if len(df_success) > 0:
            correlation = df_success[['execution_time', 'peak_memory_kb', 'total_path_length', 'nodes_visited', 'agent_count']].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            plt.title('Correlation Matrix for Successful Multi-agent Runs')
            plt.tight_layout()
            plt.savefig("analysis_results/multi_agent_correlation.png")
            plt.close()
        
        # 3. Analysis of execution time vs agent count
        agent_perf = df.groupby(['algorithm', 'agent_count']).agg({
            'execution_time': 'mean',
            'is_goal_reached': 'mean'
        }).reset_index()
        
        # Plot execution time by agent count
        plt.figure(figsize=(14, 8))
        
        for algorithm in agent_perf['algorithm'].unique():
            alg_data = agent_perf[agent_perf['algorithm'] == algorithm]
            plt.plot(alg_data['agent_count'], alg_data['execution_time'], marker='o', linewidth=2, label=algorithm)
        
        plt.xlabel('Number of Agents')
        plt.ylabel('Execution Time (s)')
        plt.title('Algorithm Scaling with Number of Agents')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("analysis_results/multi_agent_scaling_plot.png")
        plt.close()
        
        # 4. Success rate by agent count
        plt.figure(figsize=(14, 8))
        
        for algorithm in agent_perf['algorithm'].unique():
            alg_data = agent_perf[agent_perf['algorithm'] == algorithm]
            plt.plot(alg_data['agent_count'], alg_data['is_goal_reached'] * 100, marker='o', linewidth=2, label=algorithm)
        
        plt.xlabel('Number of Agents')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate by Number of Agents')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("analysis_results/multi_agent_success_rate_plot.png")
        plt.close()
        
        # 5. Path quality analysis
        if 'avg_manhattan_distance' in df.columns and len(df_success) > 0:
            # Average path quality ratio
            df_success['path_quality'] = df_success['total_path_length'] / (df_success['avg_manhattan_distance'] * df_success['agent_count'])
            
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='algorithm', y='path_quality', data=df_success)
            plt.title('Path Quality (Total Path Length / Sum of Manhattan Distances)')
            plt.xlabel('Algorithm')
            plt.ylabel('Path Quality Ratio')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("analysis_results/multi_agent_path_quality.png")
            plt.close()
    
    def comparative_analysis(self):
        """Compare performance between single and multi-agent algorithms."""
        if self.single_agent_data is None or self.multi_agent_data is None:
            return
        
        # 1. Success rate comparison
        single_success = self.single_agent_data.groupby('algorithm')['is_goal_reached'].mean()
        multi_success = self.multi_agent_data.groupby('algorithm')['is_goal_reached'].mean()
        
        plt.figure(figsize=(12, 8))
        
        x = np.arange(2)
        width = 0.8 / (len(single_success) + len(multi_success))
        
        for i, (alg, rate) in enumerate(single_success.items()):
            plt.bar(x[0] + i * width, rate * 100, width, label=f"Single: {alg}")
        
        for i, (alg, rate) in enumerate(multi_success.items()):
            plt.bar(x[1] + i * width, rate * 100, width, label=f"Multi: {alg}")
        
        plt.xlabel('Algorithm Type')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate Comparison: Single vs Multi-agent Algorithms')
        plt.xticks(x, ['Single-agent', 'Multi-agent'])
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("analysis_results/success_rate_comparison.png")
        plt.close()
        
        # 2. Execution time comparison
        # Normalize by grid size for fair comparison
        s_time = self.single_agent_data.groupby(['algorithm', 'grid_size'])['execution_time'].mean().reset_index()
        m_time = self.multi_agent_data.groupby(['algorithm', 'grid_size'])['execution_time'].mean().reset_index()
        
        # Combine for medium grids only
        s_time_med = s_time[s_time['grid_size'] == 'medium']
        m_time_med = m_time[m_time['grid_size'] == 'medium']
        
        # Sort by execution time
        s_time_med = s_time_med.sort_values('execution_time')
        m_time_med = m_time_med.sort_values('execution_time')
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        ax1 = plt.subplot(211)
        ax1.bar(s_time_med['algorithm'], s_time_med['execution_time'], color='skyblue')
        ax1.set_title('Single-agent Algorithm Execution Time (Medium Grid)')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_xticklabels(s_time_med['algorithm'], rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2 = plt.subplot(212)
        ax2.bar(m_time_med['algorithm'], m_time_med['execution_time'], color='lightcoral')
        ax2.set_title('Multi-agent Algorithm Execution Time (Medium Grid)')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_xticklabels(m_time_med['algorithm'], rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("analysis_results/execution_time_comparison.png")
        plt.close()
        
        # 3. Memory usage comparison
        # Similar approach as execution time
        s_mem = self.single_agent_data.groupby(['algorithm', 'grid_size'])['peak_memory_kb'].mean().reset_index()
        m_mem = self.multi_agent_data.groupby(['algorithm', 'grid_size'])['peak_memory_kb'].mean().reset_index()
        
        s_mem_med = s_mem[s_mem['grid_size'] == 'medium']
        m_mem_med = m_mem[m_mem['grid_size'] == 'medium']
        
        s_mem_med = s_mem_med.sort_values('peak_memory_kb')
        m_mem_med = m_mem_med.sort_values('peak_memory_kb')
        
        # Convert to MB
        s_mem_med['peak_memory_mb'] = s_mem_med['peak_memory_kb'] / 1024
        m_mem_med['peak_memory_mb'] = m_mem_med['peak_memory_kb'] / 1024
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        ax1 = plt.subplot(211)
        ax1.bar(s_mem_med['algorithm'], s_mem_med['peak_memory_mb'], color='lightgreen')
        ax1.set_title('Single-agent Algorithm Memory Usage (Medium Grid)')
        ax1.set_ylabel('Peak Memory (MB)')
        ax1.set_xticklabels(s_mem_med['algorithm'], rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2 = plt.subplot(212)
        ax2.bar(m_mem_med['algorithm'], m_mem_med['peak_memory_mb'], color='lightpink')
        ax2.set_title('Multi-agent Algorithm Memory Usage (Medium Grid)')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_xticklabels(m_mem_med['algorithm'], rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("analysis_results/memory_usage_comparison.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze pathfinding algorithm test results")
    parser.add_argument("-s", "--single", type=str, default="test_results/single_agent_results.csv",
                        help="Path to single-agent results CSV file")
    parser.add_argument("-m", "--multi", type=str, default="test_results/multi_agent_results.csv",
                        help="Path to multi-agent results CSV file")
    
    args = parser.parse_args()
    
    # Check if at least one file exists
    if not os.path.exists(args.single) and not os.path.exists(args.multi):
        print("Error: No valid result files found. Run tests first or provide valid file paths.")
        return
    
    # Create analyzer and run analysis
    analyzer = ResultAnalyzer(args.single, args.multi)
    analyzer.run_full_analysis()

    print("Analysis complete. Results saved to the 'analysis_results' directory.")


if __name__ == "__main__":
    main()
