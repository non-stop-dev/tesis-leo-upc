import argparse
import subprocess
import os
import sys

def convert_html_to_md(source_html: str, target_md: str):
    """
    Converts HTML to Markdown using Pandoc with specific flags for GFX and MathJax.
    """
    if not os.path.exists(source_html):
        print(f"Error: Source file '{source_html}' does not exist.")
        sys.exit(1)

    # Use a more robust markdown format if gfm escapes too much
    # but initially sticking to user's requested 'gfm'
    command = [
        "pandoc",
        source_html,
        "--from=html",
        "--to=gfm",
        "--mathjax",
        "-o", target_md
    ]

    try:
        print(f"Converting '{source_html}' to '{target_md}'...")
        subprocess.run(command, check=True)
        print("Conversion successful.")
        
        # Post-processing to ensure LaTeX blocks are correctly formatted as requested by user
        # User wants \[ ... \] blocks and \( ... \) inline
        # Pandoc sometimes escapes underscores or other characters in gfm.
        cleanup_math(target_md)
        
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'pandoc' command not found. Please install pandoc.")
        sys.exit(1)

def cleanup_math(file_path: str):
    """
    Performs minor cleanup on the generated markdown to ensure math blocks 
    are clean and follows the user's requested formatting.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pandoc gfm escapes underscores in math blocks like \mathbf{x}\_i
    # We want to unescape them in math contexts if possible, or use a better pandoc format.
    # However, a simple replacement for common escapes in math can help.
    
    # Note: This is a simple heuristic. For a more robust solution, 
    # using 'markdown' instead of 'gfm' in pandoc is usually better.
    
    # Replacing escaped underscores that are likely math:
    import re
    
    # Strip common sphinx/mathjax HTML noise that Pandoc preserves
    # Handle multi-line spans and divs
    content = re.sub(r'<span class="math notranslate nohighlight">\s*(.*?)\s*</span>', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'<div class="math notranslate nohighlight">\s*(.*?)\s*</div>', r'\[\n\1\n\]', content, flags=re.DOTALL)

    # Unescape characters that Pandoc escapes in GFM but are part of LaTeX
    # We want to be careful not to corrupt the markdown but help the LaTeX
    content = content.replace('\\_', '_')
    content = content.replace('\\{', '{').replace('\\}', '}')
    content = content.replace('\\$', '$')
    
    # Pandoc often double-escapes backslashes in GFM math: \\\\ -> \
    # We want to preserve \mathbf, \sum, etc.
    # First, handle the common 4-to-1 or 2-to-1 mapping
    content = content.replace('\\\\', '\\')
    content = content.replace('\\\\', '\\') # Do it twice to handle quadruple
    
    # Fix the common case where a trailing backslash might have been added
    content = re.sub(r'\\(\s*\])', r'\1', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HTML to GFM Markdown using Pandoc.")
    parser.add_argument("source", help="Path to the source HTML file")
    parser.add_argument("target", help="Path to the target MD file")

    args = parser.parse_args()
    
    convert_html_to_md(args.source, args.target)
