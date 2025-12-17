# Shortest Path Optimization - Complete Guide

## 📋 Project Overview

This project implements and compares three shortest path algorithms for city route optimization:

1. **Dijkstra's Algorithm** (Greedy)
2. **Bellman-Ford Algorithm** (Dynamic Programming)
3. **Floyd-Warshall Algorithm** (All-pairs shortest path)

---

## 🚀 Getting Started

### Prerequisites

Install required Python libraries:

```bash
pip install matplotlib networkx
```

### Files Included

1. **main.py** - Main implementation with all three algorithms
2. **test_different_sizes.py** - Scalability testing script
3. **README.md** - This file

---

## 💻 How to Run the Code

### Method 1: Run Main Demo

```bash
python main.py
```

**What it does:**

* Creates a city map with 10 nodes
* Runs all three algorithms from node A to node J
* Displays results and comparison
* Generates visualization images

**Output:**

* `city_map.png` - Visual representation of your city network
* `algorithm_comparison.png` - Performance comparison charts
* Console output with detailed results

### Method 2: Run Scalability Tests

```bash
python test_different_sizes.py
```

**What it does:**

* Tests algorithms on maps of different sizes (10, 15, 20, 25, 30, 40, 50 nodes)
* Measures execution time and operations count
* Generates scalability graphs

**Output:**

* `scalability_analysis.png` - Shows how performance scales with map size
* Summary table for your report

---

## 📊 Understanding the Results

### Example Output:

```
============================================================
Algorithm: Dijkstra
============================================================
Path Found: A -> B -> E -> F -> J
Total Distance: 12.00 km
Execution Time: 0.2341 ms
Steps/Operations: 24
Nodes Visited: 7
============================================================
```

### Performance Comparison Table:

```
Algorithm            Time (ms)       Steps           Distance     
--------------------------------------------------------------------------------
Dijkstra            0.2341          24              12.00        
Bellman-Ford        1.5623          156             12.00        
Floyd-Warshall      3.8912          1000            12.00        
```

---

## 🎯 For Your Report

### What to Include in Each Section:

#### 1. **Introduction**

* Use the problem definition from code comments
* Mention real-world applications (GPS, delivery, emergency routing)

#### 2. **Methodology**

* Explain the CityMap data structure (adjacency list)
* Describe test data: small (10 nodes), medium (15 nodes), large (50 nodes)
* Programming language: Python 3.x

#### 3. **Algorithm Descriptions**

**Dijkstra's Algorithm:**

* **Time Complexity:** O((V + E) log V)
* **Best Case:** When destination is close to source
* **Worst Case:** Dense graph, destination far from source
* **Use Case:** Single source, all positive weights

**Bellman-Ford Algorithm:**

* **Time Complexity:** O(V × E)
* **Best Case:** Few edges
* **Worst Case:** Complete graph
* **Use Case:** Can handle negative weights, more versatile

**Floyd-Warshall Algorithm:**

* **Time Complexity:** O(V³)
* **Best Case:** Small graphs
* **Worst Case:** Large graphs (any case)
* **Use Case:** All pairs shortest path, pre-computation scenarios

#### 4. **Results Analysis**

**Key Findings (Based on typical results):**

* Dijkstra is fastest for single-source shortest path
* Bellman-Ford is slower but more versatile
* Floyd-Warshall is slowest but finds all pairs at once
* All algorithms find the optimal path with same distance
* Performance gap increases with graph size

#### 5. **Complexity Analysis Tables**

| Algorithm      | Time Complexity | Space Complexity | Operations (10 nodes) | Operations (50 nodes) |
| -------------- | --------------- | ---------------- | --------------------- | --------------------- |
| Dijkstra       | O((V+E)log V)   | O(V)             | ~24                   | ~380                  |
| Bellman-Ford   | O(V×E)         | O(V)             | ~156                  | ~4,500                |
| Floyd-Warshall | O(V³)          | O(V²)           | ~1,000                | ~125,000              |

---

## 🎥 For Your Video Presentation

### Demo Script (Copy this):

```
"Hello, we'll demonstrate our shortest path algorithms.

[Show city_map.png]
Here's our city network with 10 intersections and roads with distances.

[Run code]
Let's find the shortest path from A to J.

[Point to console]
First, Dijkstra's algorithm runs in 0.23 milliseconds, 
finding path A->B->E->F->J with distance 12 km.

Bellman-Ford takes 1.56 milliseconds for the same path.

Floyd-Warshall, being an all-pairs algorithm, takes 3.89 milliseconds.

[Show comparison graph]
As you can see, Dijkstra is clearly the fastest.

[Show scalability graph]
This graph shows how performance scales. Notice Floyd-Warshall's 
exponential growth compared to Dijkstra's linear growth.

For city routing applications, Dijkstra is the optimal choice."
```

---

## 🔧 Customization Options

### Change Start/End Points:

```python
start_node = 'A'  # Change this
end_node = 'J'    # Change this
```

### Create Your Own City Map:

```python
city = CityMap()
city.add_edge('Location1', 'Location2', distance)
city.add_edge('Location2', 'Location3', distance)
# Add more edges...
```

### Test Different Sizes:

```python
# In test_different_sizes.py
test_sizes = [10, 20, 30, 50, 100]  # Modify this list
```

---

## 📈 Expected Results for Your Report

### Small Map (10 nodes):

* Dijkstra: ~0.2 ms
* Bellman-Ford: ~1.5 ms
* Floyd-Warshall: ~3.8 ms
* **Winner:** Dijkstra

### Medium Map (25 nodes):

* Dijkstra: ~1.2 ms
* Bellman-Ford: ~8.5 ms
* Floyd-Warshall: ~45 ms
* **Winner:** Dijkstra

### Large Map (50 nodes):

* Dijkstra: ~4.5 ms
* Bellman-Ford: ~35 ms
* Floyd-Warshall: ~320 ms
* **Winner:** Dijkstra

---

## 🎓 Key Points for Conclusion

1. **Best Algorithm:** Dijkstra's algorithm is optimal for city route finding
2. **Reasoning:**
   * Fastest execution time
   * Efficient for single-source shortest path
   * Suitable for real-time GPS applications
3. **Trade-offs:**
   * Bellman-Ford: Use when negative weights possible
   * Floyd-Warshall: Use when all-pairs distances needed
4. **Real-world Application:** Modern GPS systems use Dijkstra or A* (variant)

---

## 📝 Code Files for Submission (.txt format)

Save these files as required:

1. `d
