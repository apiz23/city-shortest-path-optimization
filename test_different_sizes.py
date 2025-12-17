"""
Test script to compare algorithm performance on different sized city maps
This generates data for your report's performance analysis section
"""

import heapq
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import random

# ============================================================================
# COPY THE THREE ALGORITHMS HERE (from main code)
# ============================================================================

class CityMap:
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()
        self.edges = []
        
    def add_edge(self, from_node, to_node, weight):
        self.graph[from_node].append((to_node, weight))
        self.graph[to_node].append((from_node, weight))
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edges.append((from_node, to_node, weight))
    
    def get_all_nodes(self):
        return list(self.nodes)

def dijkstra(city_map, start, end):
    start_time = time.time()
    step_count = 0
    
    distances = {node: float('inf') for node in city_map.get_all_nodes()}
    distances[start] = 0
    previous_nodes = {node: None for node in city_map.get_all_nodes()}
    pq = [(0, start)]
    visited = set()
    
    while pq:
        step_count += 1
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        if current_node == end:
            break
        
        for neighbor, weight in city_map.graph[current_node]:
            step_count += 1
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    
    return {
        'algorithm': 'Dijkstra',
        'path': path if path and path[0] == start else [],
        'distance': distances[end] if distances[end] != float('inf') else None,
        'execution_time': execution_time,
        'step_count': step_count
    }

def bellman_ford(city_map, start, end):
    start_time = time.time()
    step_count = 0
    
    nodes = city_map.get_all_nodes()
    distances = {node: float('inf') for node in nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in nodes}
    
    for _ in range(len(nodes) - 1):
        step_count += 1
        for from_node in city_map.graph:
            for to_node, weight in city_map.graph[from_node]:
                step_count += 1
                if distances[from_node] + weight < distances[to_node]:
                    distances[to_node] = distances[from_node] + weight
                    previous_nodes[to_node] = from_node
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    
    return {
        'algorithm': 'Bellman-Ford',
        'path': path if path and path[0] == start else [],
        'distance': distances[end] if distances[end] != float('inf') else None,
        'execution_time': execution_time,
        'step_count': step_count
    }

def floyd_warshall(city_map, start, end):
    start_time = time.time()
    step_count = 0
    
    nodes = city_map.get_all_nodes()
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    dist = [[float('inf')] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for from_node in city_map.graph:
        for to_node, weight in city_map.graph[from_node]:
            i = node_index[from_node]
            j = node_index[to_node]
            dist[i][j] = weight
            next_node[i][j] = to_node
    
    for k in range(n):
        step_count += 1
        for i in range(n):
            step_count += 1
            for j in range(n):
                step_count += 1
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    start_idx = node_index[start]
    end_idx = node_index[end]
    
    path = []
    if next_node[start_idx][end_idx] is not None:
        path = [start]
        current = start
        while current != end:
            current = next_node[node_index[current]][end_idx]
            path.append(current)
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    
    return {
        'algorithm': 'Floyd-Warshall',
        'path': path,
        'distance': dist[start_idx][end_idx] if dist[start_idx][end_idx] != float('inf') else None,
        'execution_time': execution_time,
        'step_count': step_count
    }

# ============================================================================
# MAP GENERATION FUNCTIONS
# ============================================================================

def generate_random_map(num_nodes, edge_density=0.3):
    """
    Generate a random city map
    edge_density: probability of edge between two nodes (0-1)
    """
    city = CityMap()
    nodes = [str(i) for i in range(num_nodes)]
    
    # Ensure connected graph - create spanning tree first
    for i in range(1, num_nodes):
        from_node = nodes[i]
        to_node = nodes[random.randint(0, i-1)]
        weight = random.randint(1, 20)
        city.add_edge(from_node, to_node, weight)
    
    # Add additional edges based on density
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if random.random() < edge_density:
                weight = random.randint(1, 20)
                city.add_edge(nodes[i], nodes[j], weight)
    
    return city

# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_scalability():
    """Test algorithm performance on different map sizes"""
    
    # Test with different sizes
    test_sizes = [10, 15, 20, 25, 30, 40, 50]
    
    results = {
        'Dijkstra': {'sizes': [], 'times': [], 'steps': []},
        'Bellman-Ford': {'sizes': [], 'times': [], 'steps': []},
        'Floyd-Warshall': {'sizes': [], 'times': [], 'steps': []}
    }
    
    print("="*80)
    print("SCALABILITY TEST - Testing algorithms on different map sizes")
    print("="*80)
    
    for size in test_sizes:
        print(f"\nTesting with {size} nodes...")
        
        # Generate map
        city_map = generate_random_map(size, edge_density=0.3)
        nodes = city_map.get_all_nodes()
        start = nodes[0]
        end = nodes[-1]
        
        # Test Dijkstra
        result = dijkstra(city_map, start, end)
        results['Dijkstra']['sizes'].append(size)
        results['Dijkstra']['times'].append(result['execution_time'])
        results['Dijkstra']['steps'].append(result['step_count'])
        print(f"  Dijkstra: {result['execution_time']:.4f} ms")
        
        # Test Bellman-Ford
        result = bellman_ford(city_map, start, end)
        results['Bellman-Ford']['sizes'].append(size)
        results['Bellman-Ford']['times'].append(result['execution_time'])
        results['Bellman-Ford']['steps'].append(result['step_count'])
        print(f"  Bellman-Ford: {result['execution_time']:.4f} ms")
        
        # Test Floyd-Warshall (skip for very large graphs)
        if size <= 50:
            result = floyd_warshall(city_map, start, end)
            results['Floyd-Warshall']['sizes'].append(size)
            results['Floyd-Warshall']['times'].append(result['execution_time'])
            results['Floyd-Warshall']['steps'].append(result['step_count'])
            print(f"  Floyd-Warshall: {result['execution_time']:.4f} ms")
    
    # Create visualization
    plot_scalability_results(results)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE FOR REPORT")
    print("="*80)
    print(f"{'Nodes':<10} {'Dijkstra (ms)':<20} {'Bellman-Ford (ms)':<20} {'Floyd-Warshall (ms)':<20}")
    print("-"*80)
    
    for i, size in enumerate(test_sizes):
        dij_time = results['Dijkstra']['times'][i]
        bf_time = results['Bellman-Ford']['times'][i]
        fw_time = results['Floyd-Warshall']['times'][i] if i < len(results['Floyd-Warshall']['times']) else 'N/A'
        
        if fw_time != 'N/A':
            print(f"{size:<10} {dij_time:<20.4f} {bf_time:<20.4f} {fw_time:<20.4f}")
        else:
            print(f"{size:<10} {dij_time:<20.4f} {bf_time:<20.4f} {fw_time:<20}")
    
    print("="*80)

def plot_scalability_results(results):
    """Create plots showing algorithm scalability"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Execution Time vs Graph Size
    ax1.plot(results['Dijkstra']['sizes'], results['Dijkstra']['times'], 
             'o-', label='Dijkstra', linewidth=2, markersize=8, color='#FF6B6B')
    ax1.plot(results['Bellman-Ford']['sizes'], results['Bellman-Ford']['times'], 
             's-', label='Bellman-Ford', linewidth=2, markersize=8, color='#4ECDC4')
    ax1.plot(results['Floyd-Warshall']['sizes'], results['Floyd-Warshall']['times'], 
             '^-', label='Floyd-Warshall', linewidth=2, markersize=8, color='#45B7D1')
    
    ax1.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time vs Graph Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Step Count vs Graph Size
    ax2.plot(results['Dijkstra']['sizes'], results['Dijkstra']['steps'], 
             'o-', label='Dijkstra', linewidth=2, markersize=8, color='#FF6B6B')
    ax2.plot(results['Bellman-Ford']['sizes'], results['Bellman-Ford']['steps'], 
             's-', label='Bellman-Ford', linewidth=2, markersize=8, color='#4ECDC4')
    ax2.plot(results['Floyd-Warshall']['sizes'], results['Floyd-Warshall']['steps'], 
             '^-', label='Floyd-Warshall', linewidth=2, markersize=8, color='#45B7D1')
    
    ax2.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Operations', fontsize=12, fontweight='bold')
    ax2.set_title('Computational Steps vs Graph Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    print("\nScalability graph saved as 'scalability_analysis.png'")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    test_scalability()