#!/usr/bin/env python3
"""
Academic Paper Merger - Merges multiple Markdown sections into a single formatted document
Optimized for economic/econometric papers with LaTeX math support

Features:
- Merges multiple markdown files in order
- Preserves LaTeX mathematical equations
- Handles tables and figures with proper formatting
- Academic citation support
- Configurable output format (Markdown with LaTeX)
"""

import os
import re
from pathlib import Path
from datetime import datetime


class AcademicPaperMerger:
    """Merges markdown documents with academic formatting for economics papers."""
    
    def __init__(self, output_dir=None):
        """Initialize the merger with optional custom output directory."""
        self.output_dir = output_dir or os.getcwd()
        self.sections = []
        self.media_dir = None
        
    def add_section(self, filepath, title=None):
        """Add a markdown section to be merged.
        
        Args:
            filepath: Path to the markdown file
            title: Optional custom title (otherwise extracted from file)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from content if not provided
        if title is None:
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
            else:
                title = os.path.splitext(os.path.basename(filepath))[0]
        
        self.sections.append({
            'title': title,
            'content': content,
            'filepath': filepath
        })
        
    def set_media_directory(self, media_dir):
        """Set the directory containing media files (images, etc.)."""
        self.media_dir = media_dir
        
    def _process_math_equations(self, content):
        """Process and standardize LaTeX math equations.
        
        Ensures proper formatting for:
        - Inline math: $...$
        - Display math: $$...$$
        - Equation environments
        """
        # Preserve display equations ($$...$$)
        content = re.sub(r'\$\$([^$]+?)\$\$', r'\n$$\1$$\n', content)
        
        # Preserve inline math ($...$)
        # Already in correct format, no change needed
        
        # Handle LaTeX equation environments
        content = re.sub(
            r'\\begin\{equation\}(.*?)\\end\{equation\}',
            r'\n$$\1$$\n',
            content,
            flags=re.DOTALL
        )
        
        return content
    
    def _process_tables(self, content):
        """Process markdown tables for better academic formatting.
        
        Ensures:
        - Proper alignment
        - Caption formatting
        - Source notes
        """
        # Find tables and add spacing
        lines = content.split('\n')
        processed = []
        in_table = False
        
        for i, line in enumerate(lines):
            # Detect table start
            if re.match(r'^\s*\|.*\|.*\|', line) and not in_table:
                in_table = True
                # Look for caption above
                if i > 0 and not lines[i-1].strip():
                    processed.append('')  # Ensure space before table
            
            processed.append(line)
            
            # Detect table end
            if in_table and (i == len(lines)-1 or not re.match(r'^\s*\|', lines[i+1])):
                in_table = False
                processed.append('')  # Ensure space after table
        
        return '\n'.join(processed)
    
    def _process_images(self, content):
        """Process image references and ensure proper paths."""
        if not self.media_dir:
            return content
        
        # Find markdown image syntax: ![alt](path)
        def replace_image(match):
            alt_text = match.group(1)
            image_path = match.group(2)
            
            # If relative path, make it relative to media directory
            if not image_path.startswith(('http://', 'https://', '/')):
                image_path = f"media/{os.path.basename(image_path)}"
            
            return f'![{alt_text}]({image_path})'
        
        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, content)
        return content
    
    def _process_citations(self, content):
        """Standardize citation format for academic papers.
        
        Converts various citation formats to consistent style:
        - (Author, Year) format
        - (Author et al., Year) for multiple authors
        """
        # This is a basic implementation - can be enhanced with bibTeX support
        return content
    
    def _add_document_header(self):
        """Generate document header with metadata and formatting (no hardcoded content)."""
        header = r"""---
output: pdf_document
fontsize: 12pt
papersize: a4
geometry:
  - a4paper
  - margin=1in
linestretch: 1.5
documentclass: article
lang: es
header-includes:
  - \renewcommand{\listtablename}{Índice de tablas}
  - \usepackage{unicode-math}
  - \usepackage{fontspec}
  - \setmainfont{Latin Modern Roman}
  - \usepackage{url}
  - \usepackage[hyphens]{url}
  - \usepackage[breaklinks=true]{hyperref}
  - \usepackage{xurl}
  - \urlstyle{same}
  - \PassOptionsToPackage{hyphens}{url}
  - \def\UrlBreaks{\do\/\do-\do.\do=\do?\do\&}
---

"""
        return header
    
    def _clean_section_content(self, content, keep_h1=False):
        """Clean individual section content before merging.
        
        - Removes document-level headers (# 1. TITLE or # TITLE)
        - PRESERVES manual section numbers (1.1, 1.2.1, etc.)
        - Keeps original heading hierarchy (## stays ##, ### stays ###)
        - Removes redundant spacing
        
        Args:
            keep_h1: If True, keeps the top-level # heading (for Referencias)
        """
        if not keep_h1:
            # Remove top-level title (e.g., "# 1. INTRODUCCIÓN")
            content = re.sub(r'^#\s+\d+\.\s+[^\n]+\n+', '', content, count=1)
        
        # DO NOT remove manual section numbers - we want to keep them!
        # Keep headings as-is: "## 1.1 Title", "### 1.2.1 Title", etc.
        
        # Ensure consistent spacing
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content
    
    def merge(self, output_filename='merged_paper.md', include_abstract=True):
        """Merge all sections into a single academic paper.
        
        Args:
            output_filename: Name of the output file
            include_abstract: Whether to include abstract section
            
        Returns:
            Path to the merged document
        """
        if not self.sections:
            raise ValueError("No sections added. Use add_section() first.")
        
        # Build the document
        document_parts = []
        
        # Add header with metadata
        document_parts.append(self._add_document_header())
        
        # Define H1 section titles to add to TOC
        section_h1_titles = {
            1: None,  # Presentation - no H1 needed (has LaTeX title page)
            2: "# 1. INTRODUCCIÓN",
            3: "# 2. APROXIMACIÓN METODOLÓGICA", 
            4: "# 3. CONCLUSIONES",
            5: None  # Referencias has its own H1 in the file
        }
        
        # Define sections that should start on a new page
        new_page_sections = {4, 5}  # Conclusions and References
        
        # Process and add each section
        for i, section in enumerate(self.sections, 1):
            print(f"Processing section {i}: {section['title']}")
            
            # After first section (Presentation), add TOC and lists
            if i == 1:
                # Clean and add presentation content
                content = self._clean_section_content(section['content'])
                content = self._process_math_equations(content)
                content = self._process_tables(content)
                content = self._process_images(content)
                content = self._process_citations(content)
                document_parts.append(content)
                document_parts.append("\n\\newpage\n\n")
                
                # Add table of contents
                document_parts.append("\\tableofcontents\n\\newpage\n\n")
                
                # Add list of tables (ÍNDICE DE TABLAS)
                document_parts.append("\\listoftables\n\\newpage\n\n")
                
                # Add list of figures (ÍNDICE DE FIGURAS)
                document_parts.append("\\listoffigures\n\\newpage\n\n")
                continue
            
            # Add page break before sections that need it (Conclusions, References)
            if i in new_page_sections:
                document_parts.append("\\clearpage\n\n")
            
            # Add H1 section title for TOC (if defined)
            if i in section_h1_titles and section_h1_titles[i]:
                document_parts.append(f"{section_h1_titles[i]}\n\n")
            
            # Clean content (removes top-level # heading, except for Referencias which keeps it)
            keep_h1 = (i == 5)  # Keep H1 for Referencias (section 5)
            content = self._clean_section_content(section['content'], keep_h1=keep_h1)
            
            # Process various elements
            content = self._process_math_equations(content)
            content = self._process_tables(content)
            content = self._process_images(content)
            content = self._process_citations(content)
            
            # Add section content (now starts at ## level)
            document_parts.append(content)
            
            # Add page break between sections (except last or sections that start with newpage)
            if i < len(self.sections) and (i + 1) not in new_page_sections:
                document_parts.append("\n\\clearpage\n\n")
        
        # Join all parts
        merged_content = ''.join(document_parts)
        
        # Write output
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        
        print(f"\n✓ Successfully merged {len(self.sections)} sections")
        print(f"✓ Output written to: {output_path}")
        print(f"✓ Total characters: {len(merged_content):,}")
        
        return output_path
    
    def generate_stats(self):
        """Generate statistics about the merged document."""
        total_words = 0
        total_equations = 0
        total_tables = 0
        total_images = 0
        
        for section in self.sections:
            content = section['content']
            
            # Count words (rough estimate)
            words = len(re.findall(r'\b\w+\b', content))
            total_words += words
            
            # Count equations
            display_eq = len(re.findall(r'\$\$.*?\$\$', content, re.DOTALL))
            inline_eq = len(re.findall(r'\$[^$]+?\$', content))
            total_equations += display_eq + inline_eq
            
            # Count tables
            tables = len(re.findall(r'\n\|.*\|.*\|\n', content))
            total_tables += tables
            
            # Count images
            images = len(re.findall(r'!\[.*?\]\(.*?\)', content))
            total_images += images
        
        return {
            'sections': len(self.sections),
            'words': total_words,
            'equations': total_equations,
            'tables': total_tables,
            'images': total_images
        }


def main():
    """Main execution function."""
    
    # Define base path
    base_path = Path('/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Entrega 5 Paper/Paper')
    
    # Define input files in order
    input_files = [
        base_path / '0. PRESENTACIÓN.md',
        base_path / '1. INTRODUCCIÓN.md',
        base_path / '2. APROXIMACIÓN METODOLÓGICA.md',
        base_path / '3. CONCLUSIONES.md',
        base_path / '4. REFERENCIAS.md'
    ]
    
    # Create merger instance
    merger = AcademicPaperMerger(output_dir=str(base_path))
    
    # Set media directory
    merger.set_media_directory(str(base_path / 'media'))
    
    # Add all sections
    print("Adding sections...")
    for i, filepath in enumerate(input_files, 1):
        if filepath.exists():
            merger.add_section(str(filepath))
            print(f"  ✓ Added section {i}: {filepath.name}")
        else:
            print(f"  ✗ Warning: File not found: {filepath}")
    
    # Generate statistics
    print("\nDocument Statistics:")
    stats = merger.generate_stats()
    print(f"  Sections:  {stats['sections']}")
    print(f"  Words:     {stats['words']:,}")
    print(f"  Equations: {stats['equations']}")
    print(f"  Tables:    {stats['tables']}")
    print(f"  Images:    {stats['images']}")
    
    # Merge the document
    print("\nMerging document...")
    output_path = merger.merge(
        output_filename='WORKING_PAPER_COMPLETE.md',
        include_abstract=True
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("\n1. Review the merged document:")
    print(f"   {output_path}")
    print("\n2. Convert to PDF using pandoc:")
    print(f"   pandoc WORKING_PAPER_COMPLETE.md -o paper.pdf \\")
    print(f"          --pdf-engine=xelatex \\")
    print(f"          --number-sections \\")
    print(f"          --toc \\")
    print(f"          --citeproc")
    print("\n3. Alternative: Open in Markdown editor with LaTeX support")
    print("   (e.g., Typora, Obsidian, VS Code with Markdown Preview)")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
