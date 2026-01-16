#!/usr/bin/env python3
"""
PDF to Markdown Converter for GNN Bibliography

Converts PDF papers to markdown format for agent skill consumption.
Uses pdftotext for extraction.
"""

import subprocess
import os
from pathlib import Path

# Source directory with PDFs
PDF_DIR = Path("/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/GNN/gnn-theory-knowledge")

# Output directory for converted files
OUTPUT_DIR = PDF_DIR / "converted-papers"

# PDFs to convert with their target skill (if applicable)
PDFS_TO_CONVERT = [
    {
        "filename": "Inductive Representation Learning on Large Graphs.pdf",
        "output_name": "hamilton_2017_graphsage.md",
        "skill_target": "large-scale-graph-sampling/resources",
    },
    {
        "filename": "A Gentle Introduction to Graph Neural Networks.pdf",
        "output_name": "gentle_intro_gnn.md",
        "skill_target": None,
    },
    {
        "filename": "Understanding Convolutions on Graphs.pdf",
        "output_name": "understanding_graph_convolutions.md",
        "skill_target": None,
    },
    {
        "filename": "Creating Message Passing Networks — pytorch_geometric documentation.pdf",
        "output_name": "pyg_message_passing.md",
        "skill_target": "custom-gnn-architectures/resources",
    },
    {
        "filename": "Graph neural networks- A review of methods and applications.pdf",
        "output_name": "gnn_review_zhou_2020.md",
        "skill_target": None,
    },
    {
        "filename": "Foundations and modelling of dynamic networks using Dynamic Graph Neural Networks- A survey.pdf",
        "output_name": "dynamic_gnn_survey.md",
        "skill_target": None,
    },
]


def convert_pdf_to_text(pdf_path: Path) -> str:
    """Convert PDF to plain text using pdftotext."""
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")
    return result.stdout


def text_to_markdown(text: str, title: str) -> str:
    """Convert plain text to basic markdown structure."""
    lines = text.split("\n")
    
    # Clean up excessive whitespace
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)
    
    # Add markdown header
    md_content = f"# {title}\n\n"
    md_content += "> **Note**: This is an auto-converted document from PDF. Some formatting may be imperfect.\n\n"
    md_content += "---\n\n"
    md_content += "\n".join(cleaned_lines)
    
    return md_content


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"Converting PDFs from: {PDF_DIR}")
    print(f"Output to: {OUTPUT_DIR}")
    print("-" * 60)
    
    for pdf_info in PDFS_TO_CONVERT:
        pdf_path = PDF_DIR / pdf_info["filename"]
        output_name = pdf_info["output_name"]
        skill_target = pdf_info["skill_target"]
        
        if not pdf_path.exists():
            print(f"⚠️  Not found: {pdf_info['filename']}")
            continue
        
        print(f"📄 Converting: {pdf_info['filename']}")
        
        try:
            # Convert PDF to text
            text = convert_pdf_to_text(pdf_path)
            
            # Convert to markdown
            title = output_name.replace("_", " ").replace(".md", "").title()
            md_content = text_to_markdown(text, title)
            
            # Save to output directory
            output_path = OUTPUT_DIR / output_name
            output_path.write_text(md_content)
            print(f"   ✅ Saved: {output_path}")
            
            # If skill target specified, also copy there
            if skill_target:
                skill_path = Path("/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/.agent/skills") / skill_target / output_name
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                skill_path.write_text(md_content)
                print(f"   📁 Also copied to skill: {skill_target}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("-" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
