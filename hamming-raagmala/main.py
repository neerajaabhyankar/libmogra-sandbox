import random
import pandas as pd
import csv
from parse_csv import parse_raag_csv, write_cycles_to_csv
from graph import Graph
from search import find_all_cycles, find_all_spanning_cycles, print_cycles


def construct_graph_from_df(df):
    """
    Construct a graph from the parsed raag dataframe.

    Args:
        df: DataFrame with note columns and raag names

    Returns:
        Graph object with all nodes and edges
    """
    graph = Graph()

    # Create nodes for each raag
    for idx, row in df.iterrows():
        # Build note sequence from columns that are not NaN
        note_names = ['S', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
        note_sequence = ''.join([note for note, present in zip(note_names, row[:12]) if pd.notna(present)])
        raag_name = row['raag']

        graph.add_node(raag_name, note_sequence)

    # Build edges between nodes
    graph.build_edges()

    return graph


def print_graph_stats(graph):
    """
    Print statistics about the graph.

    Args:
        graph: Graph object
    """
    print(f"Number of nodes: {graph.num_nodes()}")
    print()

    # Get all nodes with neighbors
    nodes_with_neighbors = [node for node in graph.nodes if len(node.neighbors) > 0]

    if len(nodes_with_neighbors) >= 3:
        # Randomly choose 3 nodes with neighbors
        selected_nodes = random.sample(nodes_with_neighbors, 3)

        for i, node in enumerate(selected_nodes, 1):
            print(f"Random node {i}: {node.raag_name}")
            print(f"Note sequence: {node.note_sequence}")
            print(f"Number of neighbors: {len(node.neighbors)}")
            print(f"Neighbors:")
            for neighbor in node.neighbors:
                print(f"  - {neighbor.raag_name} ({neighbor.note_sequence})")
            print()
    else:
        print(f"Found only {len(nodes_with_neighbors)} node(s) with neighbors.")


if __name__ == "__main__":
    # Parse the CSV file
    print("Parsing CSV file...")
    df = parse_raag_csv("all_the_odavs.csv")
    print(f"Parsed {len(df)} raags")
    print()

    # Construct the graph
    print("Constructing graph...")
    graph = construct_graph_from_df(df)
    print("Graph construction complete!")
    print()

    # Print statistics
    # print_graph_stats(graph)

    # Find and print all cycles
    print("=" * 60)
    print("CYCLE DETECTION")
    print("=" * 60)
    print()
    print("Searching for cycles in the graph...")
    cycles = find_all_cycles(graph, min_cycle_length=3)
    print()
    print_cycles(cycles, max_to_print=10)  # Print first 10 cycles

    # Find and print spanning cycles
    print()
    print("=" * 60)
    print("SPANNING CYCLES (covering all 12 swars)")
    print("=" * 60)
    print()
    print("Searching for spanning cycles...")
    spanning_cycles = find_all_spanning_cycles(cycles)
    print()
    print_cycles(spanning_cycles, max_to_print=10)  # Print first 10 spanning cycles

    # Write all cycles to CSV
    print()
    print("=" * 60)
    print("WRITING CYCLES TO CSV")
    print("=" * 60)
    print()
    write_cycles_to_csv(cycles, output_file="cycles.csv", sort_by_length="desc")
    write_cycles_to_csv(spanning_cycles, output_file="spanning_cycles.csv", sort_by_length="asc")
