import os
import markdown

# Read the markdown file
with open('blogs/1.md', 'r', encoding='utf-8') as md_file:
    markdown_content = md_file.read()

# Convert markdown to html
html_content = markdown.markdown(markdown_content)

# Read the html template
with open('blogs/template.html', 'r', encoding='utf-8') as html_file:
    html_template = html_file.read()

# Inject the converted html into the template
final_html = html_template.replace('{{ MARKDOWN_CONTENT }}', html_content)

# Write the final html to a new file
with open('blogs/1.html', 'w', encoding='utf-8') as final_html_file:
    final_html_file.write(final_html)

print("HTML file generated successfully.")