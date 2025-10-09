class Node:
    """Represents a raag node in the graph."""

    def __init__(self, raag_name, note_sequence):
        """
        Initialize a Node.

        Args:
            raag_name: Name of the raag
            note_sequence: String representation of notes (e.g., "SRGPD")
        """
        self.raag_name = raag_name
        self.note_sequence = note_sequence
        self.neighbors = []

    def add_neighbor(self, neighbor_node):
        """Add a neighboring node."""
        if neighbor_node not in self.neighbors:
            self.neighbors.append(neighbor_node)

    def __repr__(self):
        return f"Node({self.raag_name}, {self.note_sequence})"

    def __str__(self):
        neighbor_names = [n.raag_name for n in self.neighbors]
        return f"{self.raag_name} ({self.note_sequence}): {len(self.neighbors)} neighbors"


class Graph:
    """Represents the raag adjacency graph."""

    def __init__(self):
        self.nodes = []
        self.node_map = {}  # Maps raag_name -> Node for quick lookup

    def add_node(self, raag_name, note_sequence):
        """
        Add a node to the graph.

        Args:
            raag_name: Name of the raag
            note_sequence: String representation of notes

        Returns:
            The created Node
        """
        node = Node(raag_name, note_sequence)
        self.nodes.append(node)
        self.node_map[raag_name] = node
        return node

    def is_one_edit_distance(self, seq1, seq2):
        """
        Check if two sequences are one edit distance apart.

        Two sequences are one edit distance apart if:
        - They have the same length
        - They differ in exactly one position (substitution)

        Args:
            seq1: First note sequence
            seq2: Second note sequence

        Returns:
            True if sequences are one edit distance apart, False otherwise
        """
        if len(seq1) != len(seq2):
            return False

        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return differences == 1

    def build_edges(self):
        """
        Build edges by connecting nodes that are one edit distance apart.
        """
        n = len(self.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                node1 = self.nodes[i]
                node2 = self.nodes[j]

                if self.is_one_edit_distance(node1.note_sequence, node2.note_sequence):
                    node1.add_neighbor(node2)
                    node2.add_neighbor(node1)

    def get_node(self, raag_name):
        """Get a node by raag name."""
        return self.node_map.get(raag_name)

    def num_nodes(self):
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def __repr__(self):
        return f"Graph with {self.num_nodes()} nodes"
