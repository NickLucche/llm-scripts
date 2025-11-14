import re
import sys

def extract_graphs(input_file, output_dir="graphs"):
    import os

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Regular expressions
    graph_start_re = re.compile(r"## BEGIN_GRAPH")
    graph_end_re = re.compile(r"## END_GRAPH")
    graph_hash_re = re.compile(r"Graph Hash: ([a-f0-9]+)")
    model_loading_re = re.compile(r"load_weights|weight_loader|model_loader")

    # Variables for tracking extraction
    current_graph = []
    current_hash = None
    in_graph = False
    skip_graph = False

    # Process input file
    with open(input_file, "r") as f:
        for line in f:
            if graph_start_re.search(line):
                in_graph = True
                current_graph = [line]  # Start new graph
                current_hash = None
                skip_graph = False  # Reset skip flag
            
            elif in_graph:
                current_graph.append(line)

                # Detect graph hash
                hash_match = graph_hash_re.search(line)
                if hash_match:
                    current_hash = hash_match.group(1)
                
                # Detect model-loading patterns
                if model_loading_re.search(line):
                    skip_graph = True

                # End of graph, process storage
                if graph_end_re.search(line):
                    in_graph = False
                    if current_hash and not skip_graph:
                        output_file = os.path.join(output_dir, f"graph_{current_hash}.txt")
                        with open(output_file, "w") as out_f:
                            out_f.writelines(current_graph)
                        print(f"Saved graph: {output_file}")

    print("Graph extraction completed.")

# Run on your file
extract_graphs(sys.argv[1])
