#!/usr/bin/env python3
"""
PyTorch Geometric Documentation Scraper - Curated Pages Only

Downloads only the specific documentation pages requested:
- Install, Get Started, Tutorials, Advanced Concepts
- Core module references, Cheatsheets

Usage:
    python pyg_docs_scraper.py [--output-dir PATH]
"""

import argparse
import re
import time
import urllib.parse
from typing import Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://pytorch-geometric.readthedocs.io/en/latest/"

# Exact pages to download (curated list)
PAGES_TO_DOWNLOAD = [
    # Index
    "",
    
    # Install PyG
    "install/installation.html",
    
    # Get Started
    "get_started/introduction.html",
    "get_started/colabs.html",
    
    # Tutorials - Main pages + actual content subpages
    "tutorial/gnn_design.html",
    "tutorial/create_gnn.html",        # Creating Message Passing Networks
    "tutorial/heterogeneous.html",     # Heterogeneous Graph Learning
    "tutorial/dataset.html",
    "tutorial/load_csv.html",          # Loading Graphs from CSV
    "tutorial/neighbor_loader.html",   # Neighbor Sampling
    "tutorial/application.html",
    "tutorial/explain.html",           # GNN Explainability
    "tutorial/point_cloud.html",       # Point Cloud Processing
    "tutorial/shallow_node_embeddings.html",  # Shallow Node Embeddings
    "tutorial/graph_transformer.html",  # Graph Transformer
    "tutorial/link_pred.html",         # Link Prediction
    "tutorial/distributed.html",
    
    # Distributed Training (subpages with custom naming)
    # These will be named 07a, 07b, 07c via CUSTOM_FILENAMES
    "tutorial/multi_gpu_vanilla.html",
    "tutorial/multi_node_multi_gpu_vanilla.html",
    "tutorial/distributed_pyg.html",
    
    # Advanced Concepts
    "advanced/batching.html",
    "advanced/sparse_tensor.html",
    "advanced/hgam.html",
    "advanced/compile.html",
    "advanced/jit.html",
    "advanced/remote.html",
    "advanced/graphgym.html",
    "advanced/cpu_affinity.html",
    
    # Package Reference (main overview pages only)
    "modules/root.html",
    "modules/nn.html",
    "modules/data.html",
    "modules/loader.html",
    "modules/sampler.html",
    "modules/datasets.html",
    "modules/llm.html",
    "modules/transforms.html",
    "modules/utils.html",
    "modules/explain.html",
    "modules/metrics.html",
    "modules/distributed.html",
    "modules/contrib.html",
    "modules/graphgym.html",
    "modules/profile.html",
    
    # Cheatsheets
    "cheatsheet/gnn_cheatsheet.html",
    "cheatsheet/data_cheatsheet.html",
]

# Custom filename overrides for specific pages
CUSTOM_FILENAMES = {
    # Design of GNN subpages
    "tutorial/create_gnn.html": "04a_creating_message_passing_networks",
    "tutorial/heterogeneous.html": "04b_heterogeneous_graph_learning",
    # Use Cases and Applications subpages
    "tutorial/neighbor_loader.html": "06c_scaling_gnns_neighbor_sampling",
    "tutorial/explain.html": "06a_gnn_explainability",
    "tutorial/point_cloud.html": "06b_point_cloud_processing",
    "tutorial/shallow_node_embeddings.html": "06d_shallow_node_embeddings",
    "tutorial/graph_transformer.html": "06e_graph_transformer",
    # Distributed Training subpages
    "tutorial/multi_gpu_vanilla.html": "07a_multi_gpu_vanilla",
    "tutorial/multi_node_multi_gpu_vanilla.html": "07b_multi_node_multi_gpu_slurm",
    "tutorial/distributed_pyg.html": "07c_distributed_training_pyg",
}


def get_soup(url: str, retries: int = 3, delay: float = 1.0) -> Optional[BeautifulSoup]:
    """Fetches a URL and returns a BeautifulSoup object."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None


def extract_content(soup: BeautifulSoup, url: str, page_path: str = "") -> tuple[str, str]:
    """
    Extracts the main content from a documentation page and converts to Markdown.
    Returns (filename, markdown_content).
    """
    # Get the page title
    title = ""
    title_tag = soup.find("h1")
    if title_tag:
        title = title_tag.get_text(strip=True)
        title = re.sub(r"[¶#]", "", title).strip()
    
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True).split("—")[0].strip()
    
    # Find main content area
    content_div = (
        soup.find("div", class_="document") or
        soup.find("div", role="main") or
        soup.find("main") or
        soup.find("article") or
        soup.find("div", class_="body")
    )
    
    if not content_div:
        content_div = soup.body
    
    if not content_div:
        return "", ""
    
    # Convert HTML to Markdown
    markdown = html_to_markdown(content_div, title)
    
    # Check for custom filename override
    if page_path in CUSTOM_FILENAMES:
        filename = CUSTOM_FILENAMES[page_path]
    else:
        # Generate filename from URL
        path = urllib.parse.urlparse(url).path
        path = path.replace("/en/latest/", "").replace("/en/stable/", "")
        path = path.rstrip("/").rstrip(".html")
        
        if not path:
            filename = "00_index"
        else:
            filename = path.replace("/", "_")
        
        filename = re.sub(r"[^\w\-_]", "_", filename)
        filename = re.sub(r"_+", "_", filename).strip("_")
    
    return filename, markdown


def html_to_markdown(element: BeautifulSoup, title: str = "") -> str:
    """
    Converts an HTML element to Markdown with proper LaTeX formatting.
    """
    lines = []
    
    if title:
        lines.append(f"# {title}\n")
    
    def process_element(el, depth: int = 0) -> str:
        """Recursively processes HTML elements to Markdown."""
        if el is None:
            return ""
        
        if isinstance(el, str):
            return el
        
        tag_name = el.name if hasattr(el, "name") else None
        
        if tag_name is None:
            return str(el)
        
        # Skip navigation, footers, and other non-content elements
        skip_classes = [
            "headerlink", "nav", "navigation", "footer", "sphinxsidebar",
            "related", "breadcrumb", "toctree-wrapper", "admonition-title"
        ]
        el_classes = el.get("class", [])
        if any(sc in " ".join(el_classes) for sc in skip_classes):
            if "admonition-title" in el_classes:
                return ""
        
        children_text = "".join(process_element(child, depth) for child in el.children)
        
        # Handle different HTML tags
        if tag_name in ["script", "style", "nav", "footer", "header"]:
            return ""
        
        # Headings
        if tag_name == "h1":
            return ""  # Already added at the top
        if tag_name == "h2":
            text = el.get_text(strip=True).replace("¶", "").replace("#", "")
            return f"\n## {text}\n\n"
        if tag_name == "h3":
            text = el.get_text(strip=True).replace("¶", "").replace("#", "")
            return f"\n### {text}\n\n"
        if tag_name == "h4":
            text = el.get_text(strip=True).replace("¶", "").replace("#", "")
            return f"\n#### {text}\n\n"
        if tag_name == "h5":
            text = el.get_text(strip=True).replace("¶", "").replace("#", "")
            return f"\n##### {text}\n\n"
        if tag_name == "h6":
            text = el.get_text(strip=True).replace("¶", "").replace("#", "")
            return f"\n###### {text}\n\n"
        
        # Paragraphs
        if tag_name == "p":
            return f"\n{children_text.strip()}\n"
        
        # Line break
        if tag_name == "br":
            return "\n"
        
        # Links
        if tag_name == "a":
            href = el.get("href", "")
            text = el.get_text(strip=True)
            if href.startswith("#") or href.startswith("mailto:"):
                return text
            return f"[{text}]({href})"
        
        # Bold/Strong
        if tag_name in ["strong", "b"]:
            return f"**{children_text}**"
        
        # Italic/Emphasis
        if tag_name in ["em", "i"]:
            return f"*{children_text}*"
        
        # Inline code
        if tag_name == "code":
            parent_classes = el.parent.get("class", []) if el.parent else []
            if "math" in parent_classes or "math" in el_classes:
                return children_text
            text = el.get_text()
            if "`" in text:
                return f"``{text}``"
            return f"`{text}`"
        
        # Preformatted code blocks
        if tag_name == "pre":
            code_el = el.find("code")
            if code_el:
                code_text = code_el.get_text()
                lang = ""
                code_classes = code_el.get("class", [])
                for cls in code_classes:
                    if cls.startswith("language-"):
                        lang = cls.replace("language-", "")
                        break
                    if cls in ["python", "python3", "py", "bash", "shell", "json", "yaml"]:
                        lang = cls
                        break
                if not lang and (">>>" in code_text or "import " in code_text):
                    lang = "python"
                return f"\n```{lang}\n{code_text.strip()}\n```\n"
            return f"\n```\n{el.get_text().strip()}\n```\n"
        
        # Lists
        if tag_name == "ul":
            items = []
            for li in el.find_all("li", recursive=False):
                item_text = process_element(li, depth + 1).strip()
                items.append(f"- {item_text}")
            return "\n" + "\n".join(items) + "\n"
        
        if tag_name == "ol":
            items = []
            for i, li in enumerate(el.find_all("li", recursive=False), 1):
                item_text = process_element(li, depth + 1).strip()
                items.append(f"{i}. {item_text}")
            return "\n" + "\n".join(items) + "\n"
        
        if tag_name == "li":
            return children_text
        
        # Definition lists
        if tag_name == "dl":
            parts = []
            current_dt = None
            for child in el.children:
                if hasattr(child, "name"):
                    if child.name == "dt":
                        current_dt = process_element(child, depth).strip()
                    elif child.name == "dd" and current_dt:
                        dd_text = process_element(child, depth).strip()
                        parts.append(f"**{current_dt}**\n: {dd_text}\n")
                        current_dt = None
            return "\n" + "\n".join(parts) + "\n"
        
        # Tables
        if tag_name == "table":
            return convert_table(el)
        
        # Blockquotes
        if tag_name == "blockquote":
            quote_text = children_text.strip()
            quoted = "\n".join(f"> {line}" for line in quote_text.split("\n"))
            return f"\n{quoted}\n"
        
        # Math elements (LaTeX)
        if tag_name == "span" and "math" in el_classes:
            math_text = el.get_text()
            math_text = clean_latex(math_text)
            return f"${math_text}$"
        
        if tag_name == "div" and "math" in el_classes:
            math_text = el.get_text()
            math_text = clean_latex(math_text)
            return f"\n$$\n{math_text}\n$$\n"
        
        # MathJax script tags
        if tag_name == "script" and el.get("type") in ["math/tex", "math/tex; mode=display"]:
            math_text = el.get_text()
            math_text = clean_latex(math_text)
            if "mode=display" in str(el.get("type", "")):
                return f"\n$$\n{math_text}\n$$\n"
            return f"${math_text}$"
        
        # Admonitions
        if tag_name == "div" and any(c in el_classes for c in ["note", "warning", "tip", "important", "caution", "danger", "admonition"]):
            admon_type = "Note"
            if "warning" in el_classes or "caution" in el_classes:
                admon_type = "Warning"
            elif "tip" in el_classes:
                admon_type = "Tip"
            elif "important" in el_classes:
                admon_type = "Important"
            elif "danger" in el_classes:
                admon_type = "Danger"
            
            content = children_text.strip()
            return f"\n> **{admon_type}:** {content}\n"
        
        # Images
        if tag_name == "img":
            src = el.get("src", "")
            alt = el.get("alt", "Image")
            if src:
                if not src.startswith("http"):
                    src = urllib.parse.urljoin(BASE_URL, src)
                return f"![{alt}]({src})"
            return ""
        
        # Figures
        if tag_name == "figure":
            return f"\n{children_text}\n"
        
        if tag_name == "figcaption":
            return f"\n*{children_text.strip()}*\n"
        
        # Horizontal rule
        if tag_name == "hr":
            return "\n---\n"
        
        return children_text
    
    content = process_element(element)
    content = cleanup_markdown(content)
    
    return content


def convert_table(table_el: BeautifulSoup) -> str:
    """Converts an HTML table to Markdown format."""
    rows = []
    header_row = None
    
    thead = table_el.find("thead")
    if thead:
        header_cells = thead.find_all(["th", "td"])
        if header_cells:
            header_row = [cell.get_text(strip=True) for cell in header_cells]
    
    if not header_row:
        first_row = table_el.find("tr")
        if first_row:
            th_cells = first_row.find_all("th")
            if th_cells:
                header_row = [cell.get_text(strip=True) for cell in th_cells]
    
    tbody = table_el.find("tbody") or table_el
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if cells:
            row = [cell.get_text(strip=True) for cell in cells]
            if header_row and row == header_row:
                continue
            rows.append(row)
    
    if not header_row and rows:
        header_row = rows.pop(0)
    
    if not header_row:
        return ""
    
    lines = []
    lines.append("| " + " | ".join(header_row) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")
    
    for row in rows:
        while len(row) < len(header_row):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header_row)]) + " |")
    
    return "\n" + "\n".join(lines) + "\n"


def clean_latex(math_text: str) -> str:
    """Cleans up LaTeX math expressions."""
    math_text = math_text.strip()
    
    if math_text.startswith("\\[") and math_text.endswith("\\]"):
        math_text = math_text[2:-2].strip()
    if math_text.startswith("\\(") and math_text.endswith("\\)"):
        math_text = math_text[2:-2].strip()
    
    return math_text


def cleanup_markdown(content: str) -> str:
    """Performs final cleanup on the generated Markdown."""
    # Remove excessive blank lines
    content = re.sub(r"\n{4,}", "\n\n\n", content)
    
    # Remove trailing whitespace
    content = "\n".join(line.rstrip() for line in content.split("\n"))
    
    # Fix double-escaped LaTeX
    content = content.replace("\\\\(", "\\(").replace("\\\\)", "\\)")
    content = content.replace("\\\\[", "\\[").replace("\\\\]", "\\]")
    
    # Convert \[ \] to $$ $$
    content = re.sub(r"\\\[\s*", "\n$$\n", content)
    content = re.sub(r"\s*\\\]", "\n$$\n", content)
    
    # Convert \( \) to $ $
    content = re.sub(r"\\\(", "$", content)
    content = re.sub(r"\\\)", "$", content)
    
    content = content.replace("¶", "")
    content = content.lstrip("\n")
    
    return content


def save_markdown(filename: str, content: str, output_dir: Path) -> bool:
    """Saves Markdown content to a file."""
    if not content.strip():
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.md"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download PyTorch Geometric curated documentation pages."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "pytorch-docs",
        help="Output directory for Markdown files"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between requests in seconds"
    )
    
    args = parser.parse_args()
    
    print(f"PyTorch Geometric Documentation Scraper")
    print(f"========================================")
    print(f"Base URL: {BASE_URL}")
    print(f"Output directory: {args.output_dir}")
    print(f"Pages to download: {len(PAGES_TO_DOWNLOAD)}")
    print()
    
    # Process each page
    successful = 0
    failed = 0
    
    for i, page_path in enumerate(PAGES_TO_DOWNLOAD, 1):
        url = BASE_URL + page_path
        print(f"[{i}/{len(PAGES_TO_DOWNLOAD)}] {page_path or 'index'}")
        
        soup = get_soup(url)
        if not soup:
            print(f"  ❌ Failed to fetch")
            failed += 1
            continue
        
        filename, markdown = extract_content(soup, url, page_path)
        
        if not filename or not markdown.strip():
            print(f"  ⚠️  No content")
            failed += 1
            continue
        
        if save_markdown(filename, markdown, args.output_dir):
            print(f"  ✅ {filename}.md")
            successful += 1
        else:
            print(f"  ⚠️  Empty")
            failed += 1
        
        time.sleep(args.delay)
    
    print(f"\n========================================")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output: {args.output_dir}")


if __name__ == "__main__":
    main()
