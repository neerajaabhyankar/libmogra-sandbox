import pandas as pd
import csv

def parse_raag_csv(file_path):
    df = pd.read_csv(file_path, header=None)

    # Extract first 12 columns (note presence/absence) and column 14 (raag name)
    # Note: pandas uses 0-based indexing, so column 14 is index 13
    note_columns = df.iloc[:, :12]  # First 12 columns
    raag_column = df.iloc[:, 13]    # Column 14 (index 13)

    # Combine the selected columns
    result_df = pd.concat([note_columns, raag_column], axis=1)

    # Set column names
    note_names = ['S', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
    column_names = note_names + ['raag']
    result_df.columns = column_names

    return result_df


def write_cycles_to_csv(cycles, output_file="cycles.csv", sort_by_length="asc"):
    """
    Write all cycles to a CSV file.

    Args:
        cycles: List of cycles, where each cycle is a list of Node objects
        output_file: Name of the output CSV file
    """
    if sort_by_length == "asc":
        cycles = sorted(cycles, key=lambda c: len(c))
    elif sort_by_length == "desc":
        cycles = sorted(cycles, key=lambda c: len(c))[::-1]
    else:
        print("unknown sorting")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(['Cycle_ID', 'Raag_Name', 'Notes'])

        # Write each cycle
        for cycle_idx, cycle in enumerate(cycles, 1):
            for node in cycle:
                # Convert note sequence to comma-separated notes
                notes = ','.join(list(node.note_sequence))
                writer.writerow([cycle_idx, node.raag_name, notes])

    print(f"Wrote {len(cycles)} cycles to {output_file}")


if __name__ == "__main__":
    df = parse_raag_csv("all_the_odavs.csv")
    print("Dataframe shape:", df.shape)
    print("\nDataframe:")
    print(df)
    print("\nFirst few rows:")
    print(df.head())