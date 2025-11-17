#!/usr/bin/env python3
"""
PDF Conversion Script for Academic Paper
Converts WORKING_PAPER_COMPLETE.md to PDF using pandoc

Usage:
    python3 convert_to_pdf.py
    
Requirements:
    - pandoc (brew install pandoc)
    - LaTeX (brew install --cask basictex)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'


def print_header():
    """Print formatted header"""
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BLUE}{Colors.BOLD}Academic Paper PDF Converter{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}\n")


def check_file_exists(filepath):
    """Check if input markdown file exists"""
    if not filepath.exists():
        print(f"{Colors.RED}✗ Error: {filepath.name} not found!{Colors.NC}")
        print("  Please run merge_academic_paper.py first.")
        return False
    print(f"{Colors.GREEN}✓{Colors.NC} Found input file: {filepath.name}")
    return True


def check_pandoc():
    """Check if pandoc is installed"""
    if not shutil.which('pandoc'):
        print(f"{Colors.RED}✗ Error: pandoc is not installed!{Colors.NC}\n")
        print("Install pandoc:")
        print("  macOS:   brew install pandoc")
        print("  Ubuntu:  sudo apt-get install pandoc")
        print("  Windows: choco install pandoc")
        return False
    
    # Get pandoc version
    try:
        result = subprocess.run(['pandoc', '--version'], 
                              capture_output=True, 
                              text=True,
                              check=True)
        version = result.stdout.split('\n')[0]
        print(f"{Colors.GREEN}✓{Colors.NC} Pandoc is installed ({version})")
        return True
    except subprocess.CalledProcessError:
        print(f"{Colors.RED}✗ Error: Cannot determine pandoc version{Colors.NC}")
        return False


def check_latex():
    """Check if LaTeX engine is available"""
    has_xelatex = shutil.which('xelatex')
    has_pdflatex = shutil.which('pdflatex')
    
    if not (has_xelatex or has_pdflatex):
        print(f"{Colors.RED}✗ Warning: No LaTeX engine found!{Colors.NC}\n")
        print("Install LaTeX:")
        print("  macOS:   brew install --cask basictex")
        print("  Ubuntu:  sudo apt-get install texlive-xetex")
        print("\nAfter installing, update LaTeX packages:")
        print("  sudo tlmgr update --self")
        print("  sudo tlmgr install collection-fontsrecommended")
        return False
    
    engine = "xelatex" if has_xelatex else "pdflatex"
    print(f"{Colors.GREEN}✓{Colors.NC} LaTeX engine is available ({engine})")
    return True


def check_rsvg():
    """Check if rsvg-convert is available (needed for SVG to PDF conversion)"""
    if not shutil.which('rsvg-convert'):
        print(f"\n{Colors.YELLOW}⚠{Colors.NC}  rsvg-convert not found (needed for SVG support)")
        print(f"{Colors.YELLOW}   Installing librsvg...{Colors.NC}\n")
        
        try:
            result = subprocess.run(
                ['brew', 'install', 'librsvg'],
                capture_output=False,
                text=True,
                timeout=180
            )
            
            if result.returncode == 0:
                print(f"\n{Colors.GREEN}✓{Colors.NC} librsvg installed successfully\n")
                return True
            else:
                print(f"\n{Colors.YELLOW}⚠{Colors.NC}  Could not install librsvg automatically")
                print("   Install manually: brew install librsvg")
                print("   SVG images will not be included\n")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"\n{Colors.YELLOW}⚠{Colors.NC}  Installation timed out")
            print("   Install manually: brew install librsvg\n")
            return False
        except Exception as e:
            print(f"\n{Colors.YELLOW}⚠{Colors.NC}  Installation failed: {e}")
            print("   Install manually: brew install librsvg\n")
            return False
    else:
        print(f"{Colors.GREEN}✓{Colors.NC} rsvg-convert is available (SVG support)")
        return True


def convert_svg_to_pdf(paper_dir):
    """Convert all SVG files to PDF for LaTeX inclusion."""
    media_dir = paper_dir / 'media'
    if not media_dir.exists():
        return True
    
    svg_files = list(media_dir.glob('*.svg'))
    if not svg_files:
        print(f"{Colors.BLUE}No SVG files to convert{Colors.NC}")
        return True
    
    print(f"\n{Colors.BLUE}Converting {len(svg_files)} SVG file(s) to PDF...{Colors.NC}")
    
    for svg_file in svg_files:
        pdf_file = svg_file.with_suffix('.pdf')
        
        # Skip if PDF exists and is newer
        if pdf_file.exists() and pdf_file.stat().st_mtime > svg_file.stat().st_mtime:
            print(f"  {svg_file.name} → {pdf_file.name} (cached)")
            continue
        
        print(f"  {svg_file.name} → {pdf_file.name}")
        
        try:
            subprocess.run(
                ['rsvg-convert', '-f', 'pdf', '-o', str(pdf_file), str(svg_file)],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}✗ Failed to convert {svg_file.name}{Colors.NC}")
            if e.stderr:
                print(f"  Error: {e.stderr}")
            return False
    
    print(f"{Colors.GREEN}✓{Colors.NC} SVG files converted to PDF\n")
    return True


def convert_to_pdf(input_file, output_file):
    """Convert markdown to PDF using pandoc"""
    print(f"\n{Colors.BLUE}Converting to PDF...{Colors.NC}")
    print(f"  Input:  {input_file.name}")
    print(f"  Output: {output_file.name}\n")
    
    # Convert SVG files to PDF (LaTeX can include PDF but not SVG directly)
    if not convert_svg_to_pdf(input_file.parent):
        return False
    
    # Get path to latex-header.tex
    latex_header = input_file.parent / 'latex-header.tex'
    
    # Pandoc command with options
    # NOTE: No --number-sections flag since we use manual numbering (1.1, 1.2, etc.) in headings
    # NOTE: No --toc flag since we use \tableofcontents in markdown to control placement
    # NOTE: Using xelatex for better Unicode/font support
    pandoc_cmd = [
        'pandoc',
        str(input_file),
        '-o', str(output_file),
        '--pdf-engine=xelatex',
        '-V', 'fontsize=12pt',
        '-V', 'papersize=a4',
        '-V', 'geometry:margin=1in',
        '-V', 'linestretch=1.5',
        '-V', 'documentclass=article',
        '-V', 'lang=es-PE',
        '-H', str(latex_header),  # Include LaTeX packages
        '--highlight-style=tango'
    ]
    
    try:
        # Run pandoc conversion
        result = subprocess.run(
            pandoc_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Success!
        print(f"\n{Colors.GREEN}{'=' * 60}{Colors.NC}")
        print(f"{Colors.GREEN}✓ Success! PDF created successfully{Colors.NC}")
        print(f"{Colors.GREEN}{'=' * 60}{Colors.NC}\n")
        
        print(f"Output file: {output_file.name}")
        print(f"Location: {output_file.absolute()}\n")
        
        # Get file size
        file_size = output_file.stat().st_size
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        print(f"File size: {size_str}\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.RED}{'=' * 60}{Colors.NC}")
        print(f"{Colors.RED}✗ Error: PDF conversion failed{Colors.NC}")
        print(f"{Colors.RED}{'=' * 60}{Colors.NC}\n")
        
        if e.stderr:
            print("Error details:")
            print(e.stderr)
        
        print("\nCommon issues:")
        print("  1. Missing LaTeX packages")
        print("     Solution: sudo tlmgr install <package-name>")
        print("\n  2. Unicode/encoding errors")
        print("     Solution: Ensure input file is UTF-8 encoded")
        print("\n  3. Math rendering issues")
        print("     Solution: Check LaTeX equation syntax\n")
        
        return False


def offer_to_open(pdf_file):
    """Ask user if they want to open the PDF"""
    try:
        response = input(f"Open PDF now? (y/n) ").strip().lower()
        if response in ('y', 'yes'):
            # Try different methods to open PDF
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(pdf_file)], check=True)
            elif sys.platform == 'linux':
                subprocess.run(['xdg-open', str(pdf_file)], check=True)
            elif sys.platform == 'win32':
                os.startfile(str(pdf_file))
            else:
                print(f"Please open {pdf_file} manually")
    except KeyboardInterrupt:
        print("\n")
    except Exception as e:
        print(f"Could not open PDF: {e}")
        print(f"Please open {pdf_file} manually")


def main():
    """Main execution function"""
    print_header()
    
    # Navigate to Paper directory
    script_dir = Path(__file__).parent
    paper_dir = script_dir / 'Paper'
    
    if not paper_dir.exists():
        print(f"{Colors.RED}✗ Error: Paper directory not found!{Colors.NC}")
        print(f"  Expected: {paper_dir}")
        sys.exit(1)
    
    os.chdir(paper_dir)
    
    # Define input and output files
    input_file = Path('WORKING_PAPER_COMPLETE.md')
    output_file = Path(f'Working_Paper_{datetime.now().strftime("%Y%m%d")}.pdf')
    
    # Check prerequisites
    if not check_file_exists(input_file):
        sys.exit(1)
    
    if not check_pandoc():
        sys.exit(1)
    
    if not check_latex():
        sys.exit(1)
    
    # Check and install rsvg-convert if needed (for SVG to PDF conversion)
    check_rsvg()
    
    # Convert to PDF
    success = convert_to_pdf(input_file, output_file)
    
    if success:
        offer_to_open(output_file)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Conversion cancelled by user.{Colors.NC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.NC}")
        sys.exit(1)
