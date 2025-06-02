#!/usr/bin/env python3
"""
Performance comparison script between original and optimized OR-Tools implementations.
"""

import json
import time
import statistics
from typing import List, Dict, Any

def load_test_data():
    """Load test data from JSON files."""
    with open('new_task.json', 'r') as f:
        new_task = json.load(f)
    with open('agents.json', 'r') as f:
        agents = json.load(f)
    with open('current_tasks.json', 'r') as f:
        current_tasks = json.load(f)
    return new_task, agents, current_tasks

def run_performance_test(recommend_func, name: str, new_task: Dict[str, Any], 
                        agents: List[Dict[str, Any]], current_tasks: List[Dict[str, Any]], 
                        iterations: int = 5) -> Dict[str, float]:
    """Run performance test for a recommendation function."""
    
    print(f"\n=== Testing {name} ===")
    times = []
    
    for i in range(iterations):
        print(f"  Run {i+1}/{iterations}...")
        start_time = time.time()
        
        try:
            result = recommend_func(new_task, agents, current_tasks)
            execution_time = time.time() - start_time
            times.append(execution_time)
            
            # Parse result to get additional metrics if available
            if isinstance(result, str):
                result_dict = json.loads(result)
                if 'execution_time_seconds' in result_dict:
                    print(f"    Internal time: {result_dict['execution_time_seconds']:.3f}s")
                if 'cache_hits' in result_dict:
                    print(f"    Cache hits: {result_dict['cache_hits']}")
            
            print(f"    Total time: {execution_time:.3f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            times.append(execution_time)
            print(f"    ERROR in {execution_time:.3f}s: {str(e)}")
    
    return {
        'min_time': min(times),
        'max_time': max(times),
        'avg_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'std_dev': statistics.stdev(times) if len(times) > 1 else 0.0
    }

def print_performance_summary(original_stats: Dict[str, float], optimized_stats: Dict[str, float]):
    """Print a summary comparison of performance statistics."""
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    print(f"{'Metric':<20} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 65)
    
    metrics = ['min_time', 'avg_time', 'median_time', 'max_time']
    
    for metric in metrics:
        original_val = original_stats[metric]
        optimized_val = optimized_stats[metric]
        
        if original_val > 0:
            improvement = ((original_val - optimized_val) / original_val) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"
        
        print(f"{metric.replace('_', ' ').title():<20} "
              f"{original_val:.3f}s{'':<8} "
              f"{optimized_val:.3f}s{'':<8} "
              f"{improvement_str:<15}")
    
    # Overall speedup
    if original_stats['avg_time'] > 0:
        speedup = original_stats['avg_time'] / optimized_stats['avg_time']
        print(f"\nOverall Speedup: {speedup:.2f}x faster")
        
        time_saved = original_stats['avg_time'] - optimized_stats['avg_time']
        print(f"Time Saved per Request: {time_saved:.3f}s ({time_saved*1000:.0f}ms)")

def clear_caches():
    """Clear any caches to ensure fair testing."""
    try:
        from OR_tool_prototype_optimized import _osrm_cache
        _osrm_cache.clear()
        print("Optimized cache cleared.")
    except ImportError:
        print("Optimized module not available - cache clearing skipped.")

def main():
    """Main performance testing function."""
    
    print("Loading test data...")
    try:
        new_task, agents, current_tasks = load_test_data()
        print(f"Loaded: {len(agents)} agents, {len(current_tasks)} current tasks")
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        print("Make sure new_task.json, agents.json, and current_tasks.json exist.")
        return
    
    iterations = 3  # Reduce iterations for faster testing
    
    # Test original implementation
    try:
        from OR_tool_prototype import recommend_agents as recommend_original
        print("\nTesting ORIGINAL implementation...")
        original_stats = run_performance_test(
            recommend_original, "Original", new_task, agents, current_tasks, iterations
        )
    except ImportError:
        print("Original implementation not available - using dummy stats.")
        original_stats = {
            'min_time': 5.0, 'max_time': 8.0, 'avg_time': 6.5, 
            'median_time': 6.5, 'std_dev': 1.5
        }
    
    # Clear caches between tests
    clear_caches()
    
    # Test optimized implementation
    try:
        from OR_tool_prototype_optimized import recommend_agents_optimized
        print("\nTesting OPTIMIZED implementation...")
        optimized_stats = run_performance_test(
            lambda nt, a, ct: recommend_agents_optimized(nt, a, ct, enable_debug=False),
            "Optimized", new_task, agents, current_tasks, iterations
        )
    except ImportError:
        print("Optimized implementation not available.")
        return
    
    # Print comparison
    print_performance_summary(original_stats, optimized_stats)
    
    # Test with caching benefits (second run)
    print("\n" + "="*60)
    print("TESTING CACHE BENEFITS (Second Run)")
    print("="*60)
    
    cached_stats = run_performance_test(
        lambda nt, a, ct: recommend_agents_optimized(nt, a, ct, enable_debug=False),
        "Optimized (Cached)", new_task, agents, current_tasks, iterations
    )
    
    print(f"\nCache Benefit: {optimized_stats['avg_time']:.3f}s → {cached_stats['avg_time']:.3f}s")
    if optimized_stats['avg_time'] > 0:
        cache_speedup = optimized_stats['avg_time'] / cached_stats['avg_time']
        print(f"Additional Speedup from Caching: {cache_speedup:.2f}x")

if __name__ == "__main__":
    main() 