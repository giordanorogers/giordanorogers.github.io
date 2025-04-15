#!/usr/bin/env python3
"""
Convert Quarto notebooks to HTML for the blog.
This script takes a .ipynb file and converts it to HTML using the blog's template.
"""

import os
import sys
import json
import re
import nbformat
from nbconvert import HTMLExporter
from pathlib import Path

def convert_notebook_to_html(notebook_path, output_dir):
    """
    Convert a Jupyter notebook to HTML using the blog's template.
    
    Args:
        notebook_path: Path to the .ipynb file
        output_dir: Directory to save the HTML file
    """
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Extract title from the first markdown cell
    title = "Untitled"
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            # Look for a heading in the markdown
            match = re.search(r'^#\s+(.+)$', cell.source, re.MULTILINE)
            if match:
                title = match.group(1)
                break
    
    # Convert notebook to HTML
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'basic'
    body, resources = html_exporter.from_notebook_node(nb)
    
    # Load the template
    template_path = Path('blogs/quarto_template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Replace placeholders in the template
    html_content = template.replace('{{ TITLE }}', title)
    html_content = html_content.replace('{{ QUARTO_CONTENT }}', body)
    
    # Create output filename
    notebook_name = Path(notebook_path).stem
    output_path = Path(output_dir) / f"{notebook_name}.html"
    
    # Write the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Converted {notebook_path} to {output_path}")
    return output_path

def update_index(notebook_path, output_path):
    """
    Update the index.html file to include the new notebook.
    
    Args:
        notebook_path: Path to the .ipynb file
        output_path: Path to the generated HTML file
    """
    # Load the notebook to get metadata
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Extract title and date from the notebook
    title = "Untitled"
    date = "1st January 2023"  # Default date
    
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            # Look for a heading in the markdown
            match = re.search(r'^#\s+(.+)$', cell.source, re.MULTILINE)
            if match:
                title = match.group(1)
                break
    
    # Try to find a date in the notebook metadata
    if 'metadata' in nb and 'date' in nb.metadata:
        date = nb.metadata.date
    
    # Read the index.html file
    with open('index.html', 'r', encoding='utf-8') as f:
        index_content = f.read()
    
    # Create the new article HTML
    relative_path = os.path.relpath(output_path, '.')
    new_article = f"""
                <article data-news-id="{hash(relative_path) % 1000}">
                    <h2><a href="{relative_path}">{title}</a></h2>
                    <address></address>
                    <small><a href="0#down" class="downarrow disabled">&#9660;</a></small>
                    <topcomment>
                        <article class="comment" style="margin-left:0px" data-comment-id="{hash(relative_path) % 1000}-" id="{hash(relative_path) % 1000}-">
                            <span class="info">
                                {date}
                            </span>
                            <pre>A Jupyter notebook exploring {title.lower()}.</pre>
                        </article>
                        <div class="readmore">
                            <i><a href="{relative_path}">read the full post at http://giordanorogers.github.io/{relative_path}</a></i>
                        </div>
                    </topcomment>
                </article>
    """
    
    # Insert the new article after the first article
    article_pattern = r'<article data-news-id="\d+">'
    index_content = re.sub(article_pattern, new_article + r'\g<0>', index_content, count=1)
    
    # Write the updated index.html file
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"Updated index.html to include {title}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_notebooks.py <notebook_path>")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    output_dir = 'blogs'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the notebook to HTML
    output_path = convert_notebook_to_html(notebook_path, output_dir)
    
    # Update the index.html file
    update_index(notebook_path, output_path)

if __name__ == "__main__":
    main() 