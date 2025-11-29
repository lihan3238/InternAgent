import json
import os
import textwrap
from collections import defaultdict
from graphviz import Digraph
from pypdf import PdfWriter # Use pypdf instead of PyPDF2

# NOTE: add trajectory to the JSON file containing ideas
JSON_FILE = ''

name = JSON_FILE.split("/")[-1].split(".")[0]

OUTPUT_PDF = f'tree_{name}.pdf'
TEMP_FILE_PREFIX = '_temp_hyp_viz_'
NODE_WIDTH = 80 # Approx characters per line in node label

def wrap_text(text, width):
    """Wraps text to a specified width for node labels."""
    return '\n'.join(textwrap.wrap(text, width=width, replace_whitespace=False))

def build_graph_recursive(graph, hyp_id, hyp_map, child_map):
    """Recursively adds nodes and edges to the graphviz object."""
    if hyp_id not in hyp_map:
        print(f"Warning: Hypothesis ID {hyp_id} not found in map.")
        return

    hyp = hyp_map[hyp_id]
    node_label = f"ID: {hyp['id']}\nIter: {hyp.get('iteration', 'N/A')}\n\n{wrap_text(hyp['text'], NODE_WIDTH)}"

    # Add the node
    graph.node(hyp_id, label=node_label, shape='box', style='rounded,filled', fillcolor='lightblue' if hyp.get('iteration', -1) == 0 else 'lightgrey')

    # Add edges to children and recurse
    if hyp_id in child_map:
        for child_id in child_map[hyp_id]:
            if child_id in hyp_map:
                # Add child node (and its subtree)
                build_graph_recursive(graph, child_id, hyp_map, child_map)
                # Add edge from current node to child
                graph.edge(hyp_id, child_id)
            else:
                 print(f"Warning: Child ID {child_id} referenced by {hyp_id} not found in map.")

def main():
    # --- 1. Load and Prepare Data ---
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file '{JSON_FILE}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{JSON_FILE}'.")
        return

    hypotheses = data.get('hypotheses', [])
    if not hypotheses:
        print("No hypotheses found in the JSON file.")
        return

    # Create maps for easy lookup
    hyp_map = {hyp['id']: hyp for hyp in hypotheses}
    child_map = defaultdict(list)
    roots = []

    # Identify roots (iteration 0) and build child map
    for hyp in hypotheses:
        parent_id = hyp.get('parent_id')
        if parent_id:
             # Check if parent exists before adding to child_map
            if parent_id in hyp_map:
                child_map[parent_id].append(hyp['id'])
            else:
                print(f"Warning: Parent ID {parent_id} for hypothesis {hyp['id']} not found. Skipping child mapping.")
        # Consider iteration 0 as roots even if they have a parent_id (though unlikely based on structure)
        if hyp.get('iteration') == 0:
             # Check if this root already identified by null parent_id, avoid duplicates
            if not parent_id and hyp['id'] not in roots:
                roots.append(hyp['id'])
            elif parent_id and hyp['id'] not in roots:
                 # If iteration 0 has a parent_id, still treat as a root for visualization start
                 print(f"Info: Treating Iteration 0 hypothesis {hyp['id']} with parent {parent_id} as a visualization root.")
                 roots.append(hyp['id'])
        elif not parent_id and hyp['id'] not in roots: # Catch roots that might not have iteration field or are not 0
             print(f"Info: Treating hypothesis {hyp['id']} with null parent_id as a root.")
             roots.append(hyp['id'])


    if not roots:
        print("Error: Could not identify any root hypotheses (iteration 0 or null parent_id).")
        # Fallback: maybe treat *all* hypotheses as roots if none are explicitly iteration 0?
        # Or just exit. Let's exit for now.
        return

    print(f"Identified {len(roots)} root hypotheses.")

    temp_pdf_files = []
    # --- 2. Generate Individual Graph PDFs ---
    for i, root_id in enumerate(roots):
        print(f"Processing root hypothesis {i+1}/{len(roots)}: {root_id}")
        if root_id not in hyp_map:
            print(f"Skipping root {root_id} as it's not in hyp_map.")
            continue

        graph_attr = {'rankdir': 'TB', 'splines': 'ortho'} # Top-to-Bottom layout, orthogonal edges
        node_attr = {'fontsize': '10'}
        edge_attr = {'arrowhead': 'vee'}

        g = Digraph(
            name=f'hypothesis_tree_{root_id}',
            comment=f'Inheritance tree for hypothesis {root_id}',
            graph_attr=graph_attr,
            node_attr=node_attr,
            edge_attr=edge_attr
        )

        # Build the graph structure starting from the root
        build_graph_recursive(g, root_id, hyp_map, child_map)

        # Render to a temporary PDF
        temp_filename_base = f"{TEMP_FILE_PREFIX}{root_id}"
        try:
            # format='pdf' tells render to output PDF directly
            # cleanup=True removes the intermediate .gv source file
            g.render(temp_filename_base, format='pdf', cleanup=True, view=False)
            temp_pdf_files.append(f"{temp_filename_base}.pdf")
            print(f"  Generated temporary PDF: {temp_filename_base}.pdf")
        except Exception as e:
            print(f"Error rendering graph for root {root_id}: {e}")
            print("  Ensure Graphviz executables are in your system PATH.")

    # --- 3. Merge PDFs ---
    if not temp_pdf_files:
        print("No temporary PDF files were generated. Cannot create final PDF.")
        return

    merger = PdfWriter()
    print(f"\nMerging {len(temp_pdf_files)} temporary PDFs into '{OUTPUT_PDF}'...")
    for pdf_file in temp_pdf_files:
        try:
            if os.path.exists(pdf_file):
                merger.append(pdf_file)
            else:
                print(f"  Warning: Temporary file {pdf_file} not found for merging.")
        except Exception as e:
            print(f"Error merging file {pdf_file}: {e}")


    try:
        with open(OUTPUT_PDF, 'wb') as f_out:
            merger.write(f_out)
        print(f"Successfully created '{OUTPUT_PDF}'.")
    except Exception as e:
        print(f"Error writing final PDF '{OUTPUT_PDF}': {e}")
    finally:
        merger.close() # Close the merger object

    # --- 4. Cleanup Temporary Files ---
    print("Cleaning up temporary files...")
    for pdf_file in temp_pdf_files:
        if os.path.exists(pdf_file):
            try:
                os.remove(pdf_file)
                print(f"  Removed: {pdf_file}")
            except OSError as e:
                print(f"  Error removing {pdf_file}: {e}")

    print("Visualization process complete.")

if __name__ == "__main__":
    main()
