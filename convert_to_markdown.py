#!/usr/bin/env python3
"""
Convert all .docx files in the specified directory to markdown format using pandoc.
"""

import subprocess
import os
from pathlib import Path

# Directory containing the Word documents
SOURCE_DIR = Path("/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Entrega 3 (final)/Tesis por seccion")

# Output directory for markdown files (same as source)
OUTPUT_DIR = SOURCE_DIR

def convert_docx_to_markdown(docx_file: Path, output_dir: Path) -> None:
    """
    Convert a single .docx file to markdown using pandoc.

    Args:
        docx_file: Path to the .docx file
        output_dir: Directory where the .md file will be saved
    """
    # Create output filename (replace .docx with .md)
    md_filename = docx_file.stem + ".md"
    md_filepath = output_dir / md_filename

    try:
        # Run pandoc to convert docx to markdown
        subprocess.run(
            [
                "pandoc",
                str(docx_file),
                "-f", "docx",
                "-t", "markdown",
                "-o", str(md_filepath),
                "--wrap=none",  # Don't wrap lines
                "--extract-media", str(output_dir / "media")  # Extract images
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Converted: {docx_file.name} -> {md_filename}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting {docx_file.name}: {e.stderr}")
    except Exception as e:
        print(f"✗ Unexpected error converting {docx_file.name}: {str(e)}")

def main():
    """Main function to convert all .docx files in the source directory."""
    # Check if source directory exists
    if not SOURCE_DIR.exists():
        print(f"Error: Directory not found: {SOURCE_DIR}")
        return

    # Find all .docx files (excluding temporary files starting with ~$)
    docx_files = [f for f in SOURCE_DIR.glob("*.docx") if not f.name.startswith("~$")]

    if not docx_files:
        print(f"No .docx files found in {SOURCE_DIR}")
        return

    print(f"Found {len(docx_files)} .docx file(s) to convert\n")

    # Convert each file
    for docx_file in sorted(docx_files):
        convert_docx_to_markdown(docx_file, OUTPUT_DIR)

    print(f"\nConversion complete! Markdown files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
