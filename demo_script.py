"""
DEMO SCRIPT FOR VIDEO PRESENTATION
This script provides a clear, step-by-step demonstration perfect for recording
"""

import heapq
import time
from collections import defaultdict

# Simple, clean implementation for demo

class CityMap:
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()
        
    def add_edge(self, from_node, to_node, distance):
        self.graph[from_node].append((to_node, distance))
        self.graph[to_node].append((from_node, distance))
        self.nodes.add(from_node)
        self.nodes.add(to_node)
    
    def display_map(self):
        print("\n" + "="*60)
        print("CITY MAP - Roads and Distances")
        print("="*60)
        for node in sorted(self.graph.keys()):
            connections = [f"{neighbor}({dist}km)" for neighbor, dist in self.graph[node]]
            print(f"{node}: {', '.join(connections)}")
        print("="*60 + "\n")

def dijkstra_demo(city_map, start, end):
    """Dijkstra's Algorithm with detailed step-by-step output"""
    print("\n" + "█"*60)
    print("  ALGORITHM 1: DIJKSTRA'S ALGORITHM")
    print("█"*60)
    print("Description: Greedy algorithm for shortest path")
    print("Time Complexity: O((V + E) log V)")
    print("-"*60)
    
    start_time = time.time()
    
    distances = {node: float('inf') for node in city_map.nodes}
    distances[start] = 0
    previous = {node: None for node in city_map.nodes}
    pq = [(0, start)]
    visited = set()
    step = 0
    
    print(f"\nStarting from: {start}")
    print(f"Destination: {end}\n")
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        step += 1
        print(f"Step {step}: Visiting node {current} (distance from start: {current_dist}km)")
        
        if current == end:
            print(f"\n✓ Reached destination {end}!")
            break
        
        for neighbor, weight in city_map.graph[current]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))
                print(f"   → Found shorter path to {neighbor}: {distance}km")
    
    # Reconstruct path
    path = []
    current = end
    while current:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    end_time = time.time()
    exec_time = (end_time - start_time) * 1000
    
    print("\n" + "-"*60)
    print("RESULTS:")
    print(f"  Shortest Path: {' → '.join(path)}")
    print(f"  Total Distance: {distances[end]:.1f} km")
    print(f"  Execution Time: {exec_time:.4f} milliseconds")
    print(f"  Nodes Visited: {len(visited)}")
    print(f"  Total Steps: {step}")
    print("█"*60 + "\n")
    
    return {
        'path': path,
        'distance': distances[end],
        'time': exec_time,
        'steps': step
    }

def bellman_ford_demo(city_map, start, end):
    """Bellman-Ford Algorithm with detailed output"""
    print("\n" + "█"*60)
    print("  ALGORITHM 2: BELLMAN-FORD ALGORITHM")
    print("█"*60)
    print("Description: Dynamic Programming approach")
    print("Time Complexity: O(V × E)")
    print("Advantage: Can handle negative weights")
    print("-"*60)
    
    start_time = time.time()
    
    nodes = list(city_map.nodes)
    distances = {node: float('inf') for node in nodes}
    distances[start] = 0
    previous = {node: None for node in nodes}
    step = 0
    
    print(f"\nStarting from: {start}")
    print(f"Destination: {end}")
    print(f"Relaxing edges {len(nodes)-1} times...\n")
    
    for i in range(len(nodes) - 1):
        updated = False
        print(f"Iteration {i+1}:")
        for node in city_map.graph:
            for neighbor, weight in city_map.graph[node]:
                step += 1
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    previous[neighbor] = node
                    print(f"  Updated {neighbor}: {distances[neighbor]:.1f}km via {node}")
                    updated = True
        if not updated:
            print("  No updates (converged early)")
            break
        print()
    
    # Reconstruct path
    path = []
    current = end
    while current:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    end_time = time.time()
    exec_time = (end_time - start_time) * 1000
    
    print("-"*60)
    print("RESULTS:")
    print(f"  Shortest Path: {' → '.join(path)}")
    print(f"  Total Distance: {distances[end]:.1f} km")
    print(f"  Execution Time: {exec_time:.4f} milliseconds")
    print(f"  Total Steps: {step}")
    print("█"*60 + "\n")
    
    return {
        'path': path,
        'distance': distances[end],
        'time': exec_time,
        'steps': step
    }

def floyd_warshall_demo(city_map, start, end):
    """Floyd-Warshall Algorithm with detailed output"""
    print("\n" + "█"*60)
    print("  ALGORITHM 3: FLOYD-WARSHALL ALGORITHM")
    print("█"*60)
    print("Description: All-pairs shortest path algorithm")
    print("Time Complexity: O(V³)")
    print("Advantage: Finds shortest paths between ALL pairs")
    print("-"*60)
    
    start_time = time.time()
    
    nodes = list(city_map.nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for node in city_map.graph:
        for neighbor, weight in city_map.graph[node]:
            i, j = node_idx[node], node_idx[neighbor]
            dist[i][j] = weight
            next_node[i][j] = neighbor
    
    print(f"\nComputing shortest paths for all {n}×{n} = {n*n} pairs...")
    print("Processing intermediate nodes...\n")
    
    step = 0
    for k in range(n):
        print(f"Using {nodes[k]} as intermediate node...")
        updates = 0
        for i in range(n):
            for j in range(n):
                step += 1
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
                    updates += 1
        print(f"  {updates} paths updated\n")
    
    # Reconstruct path
    start_idx = node_idx[start]
    end_idx = node_idx[end]
    
    path = []
    if next_node[start_idx][end_idx]:
        path = [start]
        current = start
        while current != end:
            current = next_node[node_idx[current]][end_idx]
            path.append(current)
    
    end_time = time.time()
    exec_time = (end_time - start_time) * 1000
    
    print("-"*60)
    print("RESULTS:")
    print(f"  Shortest Path: {' → '.join(path)}")
    print(f"  Total Distance: {dist[start_idx][end_idx]:.1f} km")
    print(f"  Execution Time: {exec_time:.4f} milliseconds")
    print(f"  Total Steps: {step}")
    print(f"  All-pairs computed: {n*n} paths calculated")
    print("█"*60 + "\n")
    
    return {
        'path': path,
        'distance': dist[start_idx][end_idx],
        'time': exec_time,
        'steps': step
    }

def display_comparison(results):
    """Display final comparison table"""
    print("\n" + "="*70)
    print("  FINAL COMPARISON - ALL THREE ALGORITHMS")
    print("="*70)
    print(f"{'Algorithm':<20} {'Path':<15} {'Distance':<12} {'Time (ms)':<12} {'Steps':<10}")
    print("-"*70)
    
    for algo_name, result in results.items():
        path_str = f"{result['path'][0]}→{result['path'][-1]}" if result['path'] else "N/A"
        print(f"{algo_name:<20} {path_str:<15} {result['distance']:<12.1f} "
              f"{result['time']:<12.4f} {result['steps']:<10}")
    
    print("="*70)
    
    # Determine winner
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    print(f"\n🏆 FASTEST ALGORITHM: {fastest[0]}")
    print(f"   Execution time: {fastest[1]['time']:.4f} ms")
    print(f"   {fastest[1]['time']/max(r['time'] for r in results.values())*100:.1f}% "
          f"of slowest algorithm's time")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("-"*70)
    print("• Dijkstra: Best for single-source shortest path (RECOMMENDED)")
    print("• Bellman-Ford: Use when negative weights are possible")
    print("• Floyd-Warshall: Use when all-pairs distances are needed")
    print("="*70 + "\n")

def main():
    """Main demo function"""
    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  SHORTEST PATH OPTIMIZATION IN CITY MAP".center(68) + "█")
    print("█" + "  BIE 20303 - Algorithm and Complexities Project".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    print("\n")
    
    # Create city map
    print("Setting up city map...")
    city = CityMap()
    
    # Define roads with distances
    roads = [
        ('A', 'B', 5), ('A', 'D', 7),
        ('B', 'C', 3), ('B', 'E', 2),
        ('C', 'F', 4), ('C', 'G', 6),
        ('D', 'E', 6), ('D', 'H', 3),
        ('E', 'F', 1), ('E', 'I', 4),
        ('F', 'J', 5), ('G', 'J', 2),
        ('H', 'I', 5), ('I', 'J', 3)
    ]
    
    for from_node, to_node, distance in roads:
        city.add_edge(from_node, to_node, distance)
    
    city.display_map()
    
    # Define start and end
    start_point = 'A'
    end_point = 'J'
    
    print(f"OBJECTIVE: Find shortest path from {start_point} to {end_point}")
    print("\nPress Enter to start demonstration...")
    input()
    
    # Run all three algorithms
    results = {}
    
    # Algorithm 1: Dijkstra
    results['Dijkstra'] = dijkstra_demo(city, start_point, end_point)
    print("\nPress Enter to continue to next algorithm...")
    input()
    
    # Algorithm 2: Bellman-Ford
    results['Bellman-Ford'] = bellman_ford_demo(city, start_point, end_point)
    print("\nPress Enter to continue to next algorithm...")
    input()
    
    # Algorithm 3: Floyd-Warshall
    results['Floyd-Warshall'] = floyd_warshall_demo(city, start_point, end_point)
    
    # Display final comparison
    print("\nPress Enter to see final comparison...")
    input()
    display_comparison(results)
    
    print("\n✓ Demonstration Complete!")
    print("Thank you for watching!\n")

if __name__ == "__main__":
    main()