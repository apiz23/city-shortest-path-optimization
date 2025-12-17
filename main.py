import heapq
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

# ============================================================================
# CITY MAP DATA STRUCTURE
# ============================================================================

class CityMap:
    """Represents a city map as a weighted graph"""
    
    def __init__(self):
        self.graph = defaultdict(list)  # Adjacency list
        self.nodes = set()
        self.edges = []
        
    def add_edge(self, from_node, to_node, weight):
        """Add a bidirectional edge (road) between two nodes (intersections)"""
        self.graph[from_node].append((to_node, weight))
        self.graph[to_node].append((from_node, weight))
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edges.append((from_node, to_node, weight))
    
    def get_all_nodes(self):
        """Return list of all nodes"""
        return list(self.nodes)
    
    def get_edge_list(self):
        """Return list of all edges"""
        return self.edges
    
    def visualize(self):
        """Visualize the city map"""
        G = nx.Graph()
        for from_node, to_node, weight in self.edges:
            G.add_edge(from_node, to_node, weight=weight)
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=12, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        plt.title("City Map Network")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('city_map.png', dpi=300, bbox_inches='tight')
        print("City map saved as 'city_map.png'")

# ============================================================================
# ALGORITHM 1: DIJKSTRA'S ALGORITHM
# ============================================================================

def dijkstra(city_map, start, end):
    """
    Dijkstra's Algorithm - Greedy approach for shortest path
    Time Complexity: O((V + E) log V)
    Space Complexity: O(V)
    """
    # Initialize tracking variables
    start_time = time.time()
    step_count = 0
    
    # Distance from start to each node
    distances = {node: float('inf') for node in city_map.get_all_nodes()}
    distances[start] = 0
    
    # Track the path
    previous_nodes = {node: None for node in city_map.get_all_nodes()}
    
    # Priority queue: (distance, node)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        step_count += 1
        current_distance, current_node = heapq.heappop(pq)
        
        # Skip if already visited
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # If we reached the destination
        if current_node == end:
            break
        
        # Check all neighbors
        for neighbor, weight in city_map.graph[current_node]:
            step_count += 1
            distance = current_distance + weight
            
            # If we found a shorter path
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return {
        'algorithm': 'Dijkstra',
        'path': path if path[0] == start else [],
        'distance': distances[end] if distances[end] != float('inf') else None,
        'execution_time': execution_time,
        'step_count': step_count,
        'nodes_visited': len(visited)
    }

# ============================================================================
# ALGORITHM 2: BELLMAN-FORD ALGORITHM
# ============================================================================

def bellman_ford(city_map, start, end):
    """
    Bellman-Ford Algorithm - Dynamic Programming approach
    Time Complexity: O(V * E)
    Space Complexity: O(V)
    Can handle negative weights (not applicable here but shows versatility)
    """
    start_time = time.time()
    step_count = 0
    
    nodes = city_map.get_all_nodes()
    distances = {node: float('inf') for node in nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in nodes}
    
    # Relax edges V-1 times
    for _ in range(len(nodes) - 1):
        step_count += 1
        for from_node in city_map.graph:
            for to_node, weight in city_map.graph[from_node]:
                step_count += 1
                if distances[from_node] + weight < distances[to_node]:
                    distances[to_node] = distances[from_node] + weight
                    previous_nodes[to_node] = from_node
    
    # Reconstruct path
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
        'path': path if path[0] == start else [],
        'distance': distances[end] if distances[end] != float('inf') else None,
        'execution_time': execution_time,
        'step_count': step_count,
        'nodes_visited': len(nodes)
    }

# ============================================================================
# ALGORITHM 3: FLOYD-WARSHALL ALGORITHM
# ============================================================================

def floyd_warshall(city_map, start, end):
    """
    Floyd-Warshall Algorithm - Dynamic Programming for all pairs
    Time Complexity: O(V^3)
    Space Complexity: O(V^2)
    Finds shortest paths between all pairs of nodes
    """
    start_time = time.time()
    step_count = 0
    
    nodes = city_map.get_all_nodes()
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    
    # Distance from node to itself is 0
    for i in range(n):
        dist[i][i] = 0
    
    # Add edges to distance matrix
    for from_node in city_map.graph:
        for to_node, weight in city_map.graph[from_node]:
            i = node_index[from_node]
            j = node_index[to_node]
            dist[i][j] = weight
            next_node[i][j] = to_node
    
    # Floyd-Warshall main algorithm
    for k in range(n):
        step_count += 1
        for i in range(n):
            step_count += 1
            for j in range(n):
                step_count += 1
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    # Reconstruct path from start to end
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
        'step_count': step_count,
        'nodes_visited': n
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_small_city_map():
    """Create a small city map with 10 nodes"""
    city = CityMap()
    
    # Add roads (edges) with distances in km
    edges = [
        ('A', 'B', 5), ('A', 'D', 7),
        ('B', 'C', 3), ('B', 'E', 2),
        ('C', 'F', 4), ('C', 'G', 6),
        ('D', 'E', 6), ('D', 'H', 3),
        ('E', 'F', 1), ('E', 'I', 4),
        ('F', 'J', 5), ('G', 'J', 2),
        ('H', 'I', 5), ('I', 'J', 3)
    ]
    
    for from_node, to_node, weight in edges:
        city.add_edge(from_node, to_node, weight)
    
    return city

def create_medium_city_map():
    """Create a medium city map with 15 nodes"""
    city = CityMap()
    
    edges = [
        ('A', 'B', 4), ('A', 'C', 2),
        ('B', 'C', 1), ('B', 'D', 5),
        ('C', 'D', 8), ('C', 'E', 10),
        ('D', 'E', 2), ('D', 'F', 6),
        ('E', 'F', 3), ('E', 'G', 7),
        ('F', 'H', 1), ('G', 'H', 4),
        ('G', 'I', 5), ('H', 'J', 3),
        ('I', 'J', 2), ('I', 'K', 6),
        ('J', 'K', 4), ('J', 'L', 8),
        ('K', 'M', 3), ('L', 'M', 5),
        ('L', 'N', 2), ('M', 'O', 4),
        ('N', 'O', 6)
    ]
    
    for from_node, to_node, weight in edges:
        city.add_edge(from_node, to_node, weight)
    
    return city

def print_result(result):
    """Print algorithm result in a formatted way"""
    print(f"\n{'='*60}")
    print(f"Algorithm: {result['algorithm']}")
    print(f"{'='*60}")
    print(f"Path Found: {' -> '.join(result['path']) if result['path'] else 'No path found'}")
    print(f"Total Distance: {result['distance']:.2f} km" if result['distance'] else "No path found")
    print(f"Execution Time: {result['execution_time']:.4f} ms")
    print(f"Steps/Operations: {result['step_count']}")
    print(f"Nodes Visited: {result['nodes_visited']}")
    print(f"{'='*60}")

def compare_algorithms(results):
    """Create comparison table and visualization"""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Algorithm':<20} {'Time (ms)':<15} {'Steps':<15} {'Distance':<15}")
    print("-"*80)
    
    for result in results:
        print(f"{result['algorithm']:<20} {result['execution_time']:<15.4f} "
              f"{result['step_count']:<15} {result['distance']:<15.2f}")
    
    print("="*80)
    
    # Determine winner
    fastest = min(results, key=lambda x: x['execution_time'])
    print(f"\n🏆 WINNER: {fastest['algorithm']} (Fastest execution time)")
    
    # Create comparison graphs
    create_comparison_graphs(results)

def create_comparison_graphs(results):
    """Create visual comparison of algorithms"""
    algorithms = [r['algorithm'] for r in results]
    times = [r['execution_time'] for r in results]
    steps = [r['step_count'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Execution time comparison
    ax1.bar(algorithms, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    # Step count comparison
    ax2.bar(algorithms, steps, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_ylabel('Number of Steps/Operations')
    ax2.set_title('Computational Steps Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison graph saved as 'algorithm_comparison.png'")

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    print("="*80)
    print("SHORTEST PATH OPTIMIZATION IN CITY MAP")
    print("Comparing: Dijkstra, Bellman-Ford, and Floyd-Warshall Algorithms")
    print("="*80)
    
    # Create city map
    print("\nCreating city map...")
    city_map = create_small_city_map()
    
    # Visualize the map
    print("Visualizing city map...")
    city_map.visualize()
    
    # Define start and end points
    start_node = 'A'
    end_node = 'J'
    
    print(f"\nFinding shortest path from {start_node} to {end_node}...")
    print("\nRunning algorithms...\n")
    
    # Run all three algorithms
    results = []
    
    print("[1/3] Running Dijkstra's Algorithm...")
    result1 = dijkstra(city_map, start_node, end_node)
    print_result(result1)
    results.append(result1)
    
    print("\n[2/3] Running Bellman-Ford Algorithm...")
    result2 = bellman_ford(city_map, start_node, end_node)
    print_result(result2)
    results.append(result2)
    
    print("\n[3/3] Running Floyd-Warshall Algorithm...")
    result3 = floyd_warshall(city_map, start_node, end_node)
    print_result(result3)
    results.append(result3)
    
    # Compare results
    compare_algorithms(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nFiles generated:")
    print("  - city_map.png: Visualization of the city network")
    print("  - algorithm_comparison.png: Performance comparison graphs")
    print("\nYou can now use these results in your report and presentation.")
    print("="*80)

if __name__ == "__main__":
    main()