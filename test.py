import matplotlib.pyplot as plt
import networkx as nx

def generate_city_map():
    # Create graph
    G = nx.Graph()

    # Define roads (edges) with distances
    roads = [
        ('A', 'B', 5), ('A', 'D', 7),
        ('B', 'C', 3), ('B', 'E', 2),
        ('C', 'F', 4), ('C', 'G', 6),
        ('D', 'E', 6), ('D', 'H', 3),
        ('E', 'F', 1), ('E', 'I', 4),
        ('F', 'J', 5), ('G', 'J', 2),
        ('H', 'I', 5), ('I', 'J', 3)
    ]

    # Add edges to graph
    for u, v, w in roads:
        G.add_edge(u, v, weight=w)

    # Fixed node positions
    pos = {
        'A': (0, 2),
        'B': (1, 2),
        'C': (2, 2),
        'G': (3, 2),
        'D': (0, 1),
        'E': (1, 1),
        'F': (2, 1),
        'H': (0, 0),
        'I': (1, 0),
        'J': (2, 0)
    }

    plt.figure(figsize=(14, 8))

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=2)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color="lightblue",
        node_size=2000
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=14,
        font_weight="bold"
    )

    # Draw edge labels (distances)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)

    plt.title("City Map Network (Without Shortest Path)", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    # Save image
    plt.savefig("city_map_plain.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✓ City map generated: city_map_plain.png")

if __name__ == "__main__":
    generate_city_map()
