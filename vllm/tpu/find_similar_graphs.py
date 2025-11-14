import os
import re
import itertools
from collections import Counter
from difflib import SequenceMatcher
import sys

def extract_graph_ops(file_path):
    """Extracts unique operation types from an XLA IR graph file."""
    with open(file_path, 'r') as f:
        content = f.read()

    match = re.search(r"## BEGIN_GRAPH\n(.*?)\n## END_GRAPH", content, re.DOTALL)
    if not match:
        return None

    graph_body = match.group(1)
    
    # Extract operation types from lines like "%3 = s64[] aten::add(...)"
    ops = set(re.findall(r"\b([a-zA-Z0-9_:]+)\(", graph_body))

    return ops

def jaccard_similarity(set1, set2):
    """Computes Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def find_similar_graphs(directory, threshold=0.5):
    """Finds and groups similar XLA graphs."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("graph_") and f.endswith(".txt")]
    graph_ops = {f: extract_graph_ops(f) for f in files}

    similarities = []
    for (file1, ops1), (file2, ops2) in itertools.combinations(graph_ops.items(), 2):
        if ops1 and ops2:
            similarity = jaccard_similarity(ops1, ops2)
            if similarity >= threshold:
                similarities.append((file1, file2, similarity))

    # Sort by similarity
    similarities.sort(key=lambda x: -x[2])

    print("\n=== Similar Graphs Found ===")
    for f1, f2, score in similarities:
        print(f"{os.path.basename(f1)} <-> {os.path.basename(f2)} | Similarity: {score:.2f}")

# Run on a directory containing graph files
find_similar_graphs(sys.argv[1])  # Change "./" to your directory path
