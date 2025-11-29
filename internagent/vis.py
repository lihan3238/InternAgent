import json
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import textwrap
from matplotlib.patches import Rectangle, FancyBboxPatch
from collections import defaultdict
import matplotlib.colors as mcolors
import os

def visualize_hypotheses(json_file_path, output_pdf_path=None, font_size=11, max_evidence_items=20):
    """
    Visualize hypotheses and their relationships from a JSON file.
    
    Parameters:
    -----------
    json_file_path : str
        Path to the JSON file containing idea data
    output_pdf_path : str, optional
        Path for the output PDF file. If None, will use the input filename with '_visualization.pdf'
    font_size : int, optional
        Font size for idea content (default: 11)
    max_evidence_items : int, optional
        Maximum number of evidence items to display (default: 20)
        
    Returns:
    --------
    str
        Path to the generated PDF file
    """
    
    # Set default output path if not provided
    if output_pdf_path is None:
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_pdf_path = f"{base_name}_visualization.pdf"
    
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Create a PDF file to save all visualizations
    pdf = PdfPages(output_pdf_path)
    
    # Function to get idea details
    def get_idea_details(idea_id):
        for idea in data['ideas']:
            if idea['id'] == idea_id:
                return idea
        return None

    # Function to get parent idea
    def get_parent(idea_id):
        for idea in data['ideas']:
            if idea['id'] == idea_id:
                return idea.get('parent_id')
        return None
    
    # Function to get all ancestors of an idea
    def get_ancestors(idea_id):
        ancestors = []
        parent_id = get_parent(idea_id)
        while parent_id:
            ancestors.append(parent_id)
            parent_id = get_parent(parent_id)
        return ancestors
    
    # Function to wrap text
    def wrap_text(text, width=40):
        return '\n'.join(textwrap.wrap(text, width=width))
    
    # Improved color palette for different idea levels (colorblind-friendly)
    level_colors = [
        '#E69F00',  # Orange - Current idea
        '#56B4E9',  # Blue - Level 1 parent
        '#009E73',  # Green - Level 2 parent
        '#F0E442',  # Yellow - Level 3 parent
        '#0072B2',  # Dark blue - Level 4 parent
        '#D55E00',  # Red-brown - Level 5 parent
        '#CC79A7',  # Pink - Level 6 parent
        '#999999',  # Grey - Level 7+ parent
    ]
    
    # Extract top ideas
    top_ideas = data.get('top_ideas', [])
    if not top_ideas:
        # If top_ideas not explicitly defined, find ideas with iteration > 0
        for idea in data['ideas']:
            if idea.get('iteration', 0) > 0:
                top_ideas.append(idea['id'])

    print(f"Top Ideas: {top_ideas}")
    
    # Create a graph to represent idea inheritance
    G = nx.DiGraph()

    # Add all ideas to the graph
    for idea in data['ideas']:
        idea_id = idea['id']
        parent_id = idea.get('parent_id')

        # Add node with attributes
        scores = idea.get('scores', {})
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        # Check if this is a top idea
        is_top = idea_id in top_ideas

        # Get evidence titles
        evidence_titles = [e['title'] for e in idea.get('evidence', [])]

        G.add_node(idea_id,
                   text=idea['text'],
                   scores=scores,
                   avg_score=avg_score,
                   is_top=is_top,
                   evidence=evidence_titles)

        # Add edge from parent to child
        if parent_id:
            G.add_edge(parent_id, idea_id)
    
    # Calculate text height based on content
    def calculate_text_height(text, width=60):  # Further reduced width for better estimation
        wrapped = wrap_text(text, width)
        return len(wrapped.split('\n'))
    
    # For each top idea, create a comprehensive visualization
    for idea_id in top_ideas:
        # Estimate number of ancestors to determine figure size
        ancestors = get_ancestors(idea_id)
        num_ideas = len(ancestors) + 1

        # Dynamically adjust figure size based on number of ideas
        # More ideas need taller figure
        base_height = 20
        height_per_idea = 2.5
        fig_height = max(base_height, 15 + (num_ideas * height_per_idea))
        
        # Create a figure with increased height
        fig = plt.figure(figsize=(20, fig_height))
        
        # Create GridSpec with more separation between sections
        # Significantly increase the height ratio for the top section and add more space between sections
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 2], 
                             height_ratios=[1, 2.5, 1.5],  # Increased middle section for idea content
                             left=0.05, right=0.95, 
                             bottom=0.05, top=0.92, 
                             wspace=0.1, hspace=0.2)  # Increased vertical space
    
        # Get all ancestors of this idea
        relevant_nodes = [idea_id] + ancestors
    
        # Reverse the list to have the oldest ancestor first
        relevant_nodes_ordered = list(reversed(relevant_nodes))
    
        # Create a mapping of node to level (for coloring)
        node_level_map = {node: idx for idx, node in enumerate(relevant_nodes_ordered)}
    
        # Create a mapping of node to color
        node_color_map = {node: level_colors[min(idx, len(level_colors)-1)] 
                          for node, idx in node_level_map.items()}
    
        # Create subgraph
        subgraph = G.subgraph(relevant_nodes)
    
        # 1. Inheritance relationship (top section) - now in its own row
        ax1 = fig.add_subplot(gs[0, :])  # Span both columns in the top row
    
        # Layout for the subgraph - use a hierarchical layout with more compact spacing
        try:
            pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot', args='-Grankdir=TB -Gnodesep=0.3 -Granksep=0.4')
        except ImportError:
            # Fallback if graphviz is not available
            pos = nx.spring_layout(subgraph)
            print("Warning: Graphviz not available. Using spring layout instead.")
    
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', arrows=True, 
                              arrowstyle='-|>', arrowsize=15, width=1.5, ax=ax1)
    
        # Draw rectangle nodes with idea IDs - with different colors per level
        for node in subgraph.nodes():
            x, y = pos[node]
            
            # Get color based on level
            node_color = node_color_map[node]
            
            # Calculate rectangle width based on ID length (with some padding)
            node_font_size = font_size  # Use the provided font size
            char_width = node_font_size * 0.6  # Approximate width per character
            text_width = len(node) * char_width
            rect_width = max(text_width + 30, 130)  # Add padding with minimum width
            rect_height = 35  # Fixed height
            
            # Create rectangle with rounded corners
            rect = FancyBboxPatch((x - rect_width/2, y - rect_height/2), 
                                 rect_width, rect_height, 
                                 boxstyle="round,pad=0.3",
                                 facecolor=node_color,
                                 edgecolor='black',
                                 alpha=0.8)
            ax1.add_patch(rect)
            
            # Add node ID text
            ax1.text(x, y, node, ha='center', va='center', fontsize=node_font_size, 
                    fontweight='bold', color='black')
    
        # Adjust the axis limits to ensure all nodes are visible with padding
        x_values = [pos[node][0] for node in subgraph.nodes()]
        y_values = [pos[node][1] for node in subgraph.nodes()]
        
        if x_values and y_values:  # Check if lists are not empty
            x_min, x_max = min(x_values) - 150, max(x_values) + 150
            y_min, y_max = min(y_values) - 80, max(y_values) + 80
            
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
        
        ax1.set_title(f'Inheritance Relationship for Idea {idea_id}', fontsize=14, pad=10)
        ax1.axis('off')
    
        # 2. Idea content (middle section) - now in its own row
        ax2 = fig.add_subplot(gs[1, :])  # Span both columns in the middle row
        ax2.axis('off')
    
        # Pre-calculate text heights to better position boxes
        text_heights = {}
        total_height_needed = 0
        
        for node in relevant_nodes_ordered:
            idea = get_idea_details(node)
            if idea:
                text = idea['text']
                scores = idea.get('scores', {})
                
                # Calculate height based on content - use smaller width for wrapping
                text_height = calculate_text_height(text, width=60)
                
                # Base height + title + content lines + scores + padding
                total_height = 3 + text_height + (2 if scores else 1)
                text_heights[node] = total_height
                total_height_needed += total_height
        
        # Calculate scaling factor based on total content and available space
        # Adjust based on number of ideas
        max_total_height = 40 * (1 + (num_ideas / 10))  # Scale with number of ideas
        scale_factor = min(0.022, 0.9 / (total_height_needed / max_total_height))
        
        # Create a grid layout for idea content boxes
        num_cols = min(2, num_ideas)  # Use 2 columns if we have enough ideas
        num_rows = (num_ideas + num_cols - 1) // num_cols  # Ceiling division
        
        # Calculate grid positions
        grid_positions = []
        for i, node in enumerate(relevant_nodes_ordered):
            row = i // num_cols
            col = i % num_cols
            grid_positions.append((row, col))
        
        # Calculate grid cell size
        cell_width = 1.0 / num_cols
        cell_height = 1.0 / num_rows
        
        # Process ideas in the same order as the inheritance graph
        for i, node in enumerate(relevant_nodes_ordered):
            idea = get_idea_details(node)
            if idea:
                # Use smaller width for wrapping to prevent text overflow
                idea_text = wrap_text(idea['text'], width=60)
                scores = idea.get('scores', {})
                score_text = ', '.join([f"{k}: {v:.1f}" for k, v in scores.items()]) if scores else "No scores"
                
                # Determine if this is the current idea
                is_current = node == idea_id
                
                # Get the appropriate color from the map
                box_color = node_color_map[node]
                
                # Create content
                title = "CURRENT HYPOTHESIS: " if is_current else "PARENT HYPOTHESIS: "
                content = f"{title}{node}\n\nCONTENT: {idea_text}\n\nSCORES: {score_text}"
                
                # Create a text box with matching color and rounded corners
                props = dict(boxstyle='round,pad=0.8', facecolor=box_color, alpha=0.3)
                
                # Get grid position
                row, col = grid_positions[i]
                
                # Calculate position in the subplot
                x_pos = col * cell_width + 0.01
                y_pos = 1.0 - (row * cell_height) - 0.05
                
                # Calculate box width and height
                box_width = cell_width * 0.98
                box_height = cell_height * 0.9
                
                # Add text box with appropriate font size
                # Adjust font size based on content length and available space
                content_length = len(idea_text)
                adjusted_font_size = max(8, min(font_size - 1, 12 - (content_length // 500)))
                
                ax2.text(x_pos, y_pos, content, transform=ax2.transAxes,
                        fontsize=adjusted_font_size,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=props, wrap=True)
    
        ax2.set_title('Idea Content and Scores (Current and Parents)', fontsize=14, pad=10)
    
        # 3. Evidence used (bottom section) - now in its own row
        ax3 = fig.add_subplot(gs[2, :])  # Span both columns in the bottom row
        ax3.axis('off')
    
        # Collect evidence from this idea and all its ancestors
        evidence_by_idea = defaultdict(list)
        all_evidence = []
    
        for node_id in relevant_nodes:
            idea = get_idea_details(node_id)
            if idea and 'evidence' in idea:
                for evidence in idea['evidence']:
                    # Get the idea color for this evidence
                    evidence_color = node_color_map[node_id]

                    evidence_item = {
                        'idea_id': node_id,
                        'title': evidence['title'],
                        'authors': evidence.get('authors', ''),
                        'year': evidence.get('year', ''),
                        'relevance_score': evidence.get('relevance_score', 0),
                        'color': evidence_color
                    }
                    all_evidence.append(evidence_item)
                    evidence_by_idea[node_id].append(evidence_item)

        # Sort all evidence by relevance score
        all_evidence.sort(key=lambda x: x['relevance_score'], reverse=True)
    
        # Create table data with reference numbers and idea IDs
        table_data = []
        cell_colors = []
        
        for i, evidence in enumerate(all_evidence):
            if i >= max_evidence_items:  # Use the parameter to limit items
                break
            
            # Truncate long titles more aggressively
            truncated_title = evidence['title']
            if len(truncated_title) > 55:  # Further reduced character limit
                truncated_title = truncated_title[:52] + '...'
            
            # Create row data
            row = [
                str(i+1),  # Reference number
                truncated_title,
                evidence['idea_id'],  # Add idea ID
                str(evidence['year']),
                f"{evidence['relevance_score']:.2f}"
            ]
            table_data.append(row)
            
            # Create row colors (use light version of the idea color)
            row_color = [evidence['color']] * 5
            cell_colors.append(row_color)
    
        # Create table with optimized column widths
        if table_data:
            # Create the table with custom column widths - more compact
            table = plt.table(
                cellText=table_data,
                colLabels=['#', 'Title', 'Idea', 'Year', 'Relevance'],
                cellLoc='left',
                loc='center',
                colWidths=[0.03, 0.65, 0.17, 0.05, 0.1]  # Adjusted widths
            )
            
            # Apply cell colors with reduced alpha for better readability
            for (i, j), cell in table.get_celld().items():
                if i > 0:  # Skip header row
                    cell.set_facecolor(mcolors.to_rgba(cell_colors[i-1][j], alpha=0.2))
            
            # Format the table for better readability - more compact
            table.auto_set_font_size(False)
            table.set_fontsize(7)  # Smaller font for better fit
            table.scale(1, 1.1)  # More compact rows
            
            # Set cell heights and adjust text properties to prevent overlap
            for (i, j), cell in table.get_celld().items():
                cell.set_height(0.04)  # More compact
                
                # Make title column text smaller and ensure wrapping
                if j == 1:  # Title column
                    cell.get_text().set_fontsize(6)  # Even smaller for titles
                    cell.get_text().set_wrap(True)
                    cell.get_text().set_va('center')
                
                # Make idea IDs smaller too
                if j == 2:  # Idea column
                    cell.get_text().set_fontsize(6)
                    cell.get_text().set_wrap(True)
    
        ax3.set_title('Evidence Used by This Idea and Its Ancestors', fontsize=14, pad=10)
    
        # Set the main title for the entire figure - positioned lower to avoid overlap
        plt.suptitle(f'Comprehensive View of Top Idea: {idea_id}', fontsize=16, y=0.98)
    
        # Save the figure with tight layout
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # Save and close the PDF file
    pdf.close()
    print(f"Visualization saved to {output_pdf_path}")
    
    return output_pdf_path
