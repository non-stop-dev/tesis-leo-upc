# Academic Paper Merger Tool - Documentation

## Overview

This tool merges multiple Markdown files into a single, properly formatted academic paper suitable for economic/econometric research. It preserves LaTeX mathematical equations, handles tables and figures professionally, and outputs a document ready for PDF conversion.

## Features

### ✅ LaTeX Math Support
- **Inline equations**: `$E = mc^2$` → $E = mc^2$
- **Display equations**: `$$\frac{a}{b}$$` → 
  $$\frac{a}{b}$$
- **Complex formulas**: Preserves all LaTeX mathematical notation
- **Equation environments**: Converts `\begin{equation}...\end{equation}` to display math

### ✅ Table Formatting
- Proper spacing around tables
- Academic-style alignment
- Support for caption and source notes
- Compatible with `booktabs` package

### ✅ Image Handling
- Relative path resolution
- Media directory support
- Proper figure referencing
- Alt text preservation

### ✅ Document Structure
- YAML front matter for metadata
- Automatic table of contents
- Section numbering
- Page breaks between sections

## Quick Start

### Step 1: Merge Markdown Files

```bash
cd '/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Entrega 5 Paper'
python3 merge_academic_paper.py
```

### Step 2: Convert to PDF

```bash
python3 convert_to_pdf.py
```

That's it! You'll get a PDF file named `Working_Paper_YYYYMMDD.pdf` in the `Paper/` directory.

## Usage

### Basic Usage - Merge Script

```bash
cd '/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Entrega 5 Paper'
python3 merge_academic_paper.py
```

### Basic Usage - PDF Conversion

```bash
python3 convert_to_pdf.py
```

The script will:
- Check for required tools (pandoc, LaTeX)
- Find WORKING_PAPER_COMPLETE.md
- Convert to PDF with proper formatting
- Offer to open the PDF automatically

### Custom Configuration

Edit the `main()` function in `merge_academic_paper.py`:

```python
# Change input files
input_files = [
    base_path / 'your_intro.md',
    base_path / 'your_methods.md',
    # ... add more
]

# Change output filename
output_path = merger.merge(
    output_filename='YOUR_PAPER_NAME.md',
    include_abstract=True  # Set to False to exclude abstract
)
```

## Converting to PDF

### Method 1: Using Pandoc (Recommended)

Install pandoc first (if not installed):
```bash
# macOS
brew install pandoc
brew install --cask basictex  # Or full MacTeX

# Update LaTeX packages
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended
```

Convert to PDF:
```bash
cd '/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Entrega 5 Paper/Paper'

pandoc WORKING_PAPER_COMPLETE.md -o paper.pdf \
       --pdf-engine=xelatex \
       --number-sections \
       --toc \
       --toc-depth=3 \
       -V fontsize=12pt \
       -V geometry:margin=1in \
       -V linestretch=1.5
```

### Method 2: Using Markdown Editors

**Typora** (Commercial):
1. Open `WORKING_PAPER_COMPLETE.md`
2. File → Export → PDF

**VS Code** (Free):
1. Install "Markdown Preview Enhanced" extension
2. Open `WORKING_PAPER_COMPLETE.md`
3. Right-click → Markdown Preview Enhanced → Export (PDF)

**Obsidian** (Free):
1. Install "Pandoc Plugin"
2. Open document and use export command

### Method 3: Online Converters

Upload to:
- **Overleaf**: Copy content to new LaTeX project
- **Markdown to PDF**: Various online converters (ensure privacy for sensitive research)

## Document Statistics

After running the merge, you'll see:
```
Document Statistics:
  Sections:  5
  Words:     10,692
  Equations: 39
  Tables:    0 (in markdown format)
  Images:    4
```

## LaTeX Formatting Reference

### Mathematical Equations

#### Inline Math
```markdown
The efficiency parameter $\theta$ determines survival probability.
```

#### Display Math
```markdown
$$
P(Y=1|X) = \frac{\exp(X\beta)}{1 + \exp(X\beta)}
$$
```

#### Complex Formulas
```markdown
$$
F_n = \frac{1}{\sqrt{5}} \left(\frac{1 + \sqrt{5}}{2}\right)^n
     - \frac{1}{\sqrt{5}} \left(\frac{1 - \sqrt{5}}{2}\right)^n
$$
```

### Tables

#### Standard Markdown Table
```markdown
| Variable | Coefficient | Std. Error | p-value |
|:---------|------------:|-----------:|--------:|
| RUC      | 2.34        | 0.08       | <0.001  |
| Digital  | 0.23        | 0.03       | <0.001  |
```

#### With Caption
```markdown
**Table 1: Logistic Regression Results**

| Variable | Coefficient | Std. Error | p-value |
|:---------|------------:|-----------:|--------:|
| RUC      | 2.34        | 0.08       | <0.001  |

*Source: V Censo 2022 (INEI), authors' calculations*
```

### Figures

```markdown
![Figure 1: Regional Distribution of MYPEs](media/regional_map.png)

*Source: INEI (2022)*
```

### Citations

```markdown
According to Jovanovic (1982), firms discover their efficiency through market performance.

Several studies document this effect (Yamada, 2009; Chacaltana, 2016).
```

## Advanced Customization

### Modify Document Header

Edit `_add_document_header()` method:

```python
def _add_document_header(self):
    header = f"""---
title: "Your Custom Title"
author: "Your Name"
date: "{datetime.now().strftime('%B %d, %Y')}"
output: pdf_document
fontsize: 11pt              # Change font size
geometry: margin=1.25in     # Change margins
linestretch: 2.0            # Change line spacing
bibliography: references.bib # Add bibliography
csl: apa.csl                # Citation style
---
"""
    return header
```

### Add Custom Abstract

Edit `_add_abstract()` method with your abstract text.

### Process Citations from BibTeX

To integrate with a `.bib` file:

1. Add bibliography to YAML header:
   ```yaml
   bibliography: references.bib
   csl: chicago-author-date.csl
   ```

2. Cite in text: `[@jovanovic1982]` or `@jovanovic1982`

3. Pandoc will generate references automatically

## Troubleshooting

### "File not found" error
- Check that file paths in `input_files` list are correct
- Use absolute paths or ensure you're in the correct directory

### Math not rendering in PDF
- Ensure you're using `xelatex` or `pdflatex` as PDF engine
- Check that LaTeX distribution includes `amsmath` and `amssymb` packages

### Images not showing
- Verify `media` directory path is correct
- Check that image files exist in the media folder
- Use relative paths: `media/image.png`

### Table formatting issues
- Ensure proper markdown table syntax (pipes and hyphens)
- Use `:---` for left align, `:---:` for center, `---:` for right

### Spanish characters (á, é, í, ó, ú, ñ)
- Ensure input files are UTF-8 encoded
- Use `xelatex` engine for proper Unicode support:
  ```bash
  pandoc ... --pdf-engine=xelatex
  ```

## File Structure

```
Entrega 5 Paper/
├── merge_academic_paper.py       # Main merge script
├── README_MERGE_TOOL.md          # This documentation
├── Paper/
│   ├── 1. INTRODUCCIÓN.md        # Source sections
│   ├── 2. MODELO TEÓRICO.md
│   ├── 3. DATOS Y CONTEXTO.md
│   ├── 4. ESTRATEGIA EMPÍRICA.md
│   ├── 5. CONCLUSIONES.md
│   ├── WORKING_PAPER_COMPLETE.md # Merged output
│   └── media/                    # Images and figures
│       ├── figure1.png
│       └── table_results.png
```

## Tips for Academic Writing

### Structure
- **Introduction**: 5 pages - motivation, literature, research question
- **Methodology**: 7 pages - data, variables, model specification, problems & solutions
- **Conclusions**: 3 pages - findings, limitations, policy recommendations

### Formatting Best Practices
1. **Equations**: Number important equations for reference
2. **Tables**: Always include caption and source
3. **Figures**: High resolution (300 DPI minimum)
4. **Citations**: Use consistent format (APA 7th recommended)

### LaTeX Math Tips
- Use `\text{}` for text within equations: `$\beta_\text{RUC}$`
- Use `\times` for multiplication: `$2 \times 3$`
- Use `\cdot` for dot product: `$X \cdot \beta$`
- Use `\log` not `log`: `$\log(odds)$`

## Support

For issues or questions:
1. Check this documentation
2. Review the Context7 documentation links provided
3. Consult LaTeX documentation: https://www.latex-project.org/
4. Pandoc manual: https://pandoc.org/MANUAL.html

## Version History

- **v1.0** (2025-01-01): Initial release
  - Basic merge functionality
  - LaTeX math support
  - Table and image handling
  - Academic formatting

## License

This tool is provided as-is for academic use.
