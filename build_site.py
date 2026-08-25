#!/usr/bin/env python3
"""
Build script for the PyTorch Reference Guide static documentation site.

Reads all module README.md files, converts markdown to HTML, and generates
a single-page documentation site at docs/index.html.
"""

import os
import re
import html
from pathlib import Path

REPO_URL = "https://github.com/subinz1/pytorch-reference-guide/blob/main"
REPO_ROOT = Path(__file__).parent

CATEGORIES = {
    "Foundations": ["01", "02", "03", "04", "05", "06", "07"],
    "Compilation & Performance": ["08", "16", "17", "20", "21", "38"],
    "Attention & Transformers": ["09", "22", "34"],
    "Distributed Training": ["10", "27"],
    "Export & Deployment": ["11", "37"],
    "Architecture & Design": ["12", "13", "19", "23", "24", "35"],
    "Testing & Debugging": ["14", "28", "30"],
    "Utilities & Tooling": ["15", "18", "26", "29", "31", "32", "33"],
    "Projects": ["39", "40", "41", "42", "43", "44"],
}


def discover_modules():
    """Find all module directories and return sorted list of (dir_name, path)."""
    modules = []
    for entry in sorted(REPO_ROOT.iterdir()):
        if entry.is_dir() and entry.name[:2].isdigit() and entry.name != ".git":
            readme = entry / "README.md"
            if readme.exists():
                modules.append((entry.name, entry))
    return modules


def extract_title(content):
    """Extract the first H1 heading from markdown content."""
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def extract_description(content):
    """Extract first paragraph after the title."""
    lines = content.splitlines()
    in_content = False
    desc_lines = []
    for line in lines:
        if line.startswith("# ") and not in_content:
            in_content = True
            continue
        if in_content:
            if line.strip() == "":
                if desc_lines:
                    break
                continue
            if line.startswith("#"):
                break
            desc_lines.append(line.strip())
    return " ".join(desc_lines)


def extract_files_table(content):
    """Extract the files table from README if present."""
    files = []
    in_table = False
    for line in content.splitlines():
        if "| File" in line and "Description" in line:
            in_table = True
            continue
        if in_table:
            if line.startswith("|---"):
                continue
            if not line.startswith("|"):
                break
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 2:
                filename = re.sub(r'`([^`]+)`', r'\1', parts[0])
                desc = parts[1]
                files.append((filename, desc))
    return files


def md_to_html(content):
    """Convert markdown to HTML with basic formatting."""
    lines = content.splitlines()
    html_parts = []
    in_code_block = False
    code_lang = ""
    code_lines = []
    in_list = False
    in_table = False
    table_rows = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.startswith("```"):
            if in_code_block:
                code_content = html.escape("\n".join(code_lines))
                lang_class = f' class="language-{code_lang}"' if code_lang else ""
                html_parts.append(
                    f'<pre><code{lang_class}>{code_content}</code></pre>'
                )
                code_lines = []
                in_code_block = False
            else:
                if in_list:
                    html_parts.append("</ul>")
                    in_list = False
                in_code_block = True
                code_lang = line[3:].strip()
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Tables
        if line.startswith("|") and "|" in line[1:]:
            if not in_table:
                if in_list:
                    html_parts.append("</ul>")
                    in_list = False
                in_table = True
                table_rows = []
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(re.match(r'^[-:]+$', c) for c in cells):
                i += 1
                continue
            table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            html_parts.append(render_table(table_rows))
            in_table = False
            table_rows = []

        # Headings
        if line.startswith("######"):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            text = inline_format(line[6:].strip())
            html_parts.append(f"<h6>{text}</h6>")
            i += 1
            continue
        if line.startswith("#####"):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            text = inline_format(line[5:].strip())
            html_parts.append(f"<h5>{text}</h5>")
            i += 1
            continue
        if line.startswith("####"):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            text = inline_format(line[4:].strip())
            html_parts.append(f"<h4>{text}</h4>")
            i += 1
            continue
        if line.startswith("###"):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            text = inline_format(line[3:].strip())
            html_parts.append(f"<h3>{text}</h3>")
            i += 1
            continue
        if line.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            text = inline_format(line[3:].strip())
            anchor = slugify(line[3:].strip())
            html_parts.append(f'<h2 id="{anchor}">{text}</h2>')
            i += 1
            continue
        if line.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            text = inline_format(line[2:].strip())
            html_parts.append(f"<h1>{text}</h1>")
            i += 1
            continue

        # Blockquotes
        if line.startswith("> "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            quote_lines = []
            while i < len(lines) and lines[i].startswith("> "):
                quote_lines.append(inline_format(lines[i][2:]))
                i += 1
            html_parts.append(
                f'<blockquote>{"<br>".join(quote_lines)}</blockquote>'
            )
            continue

        # Horizontal rules
        if re.match(r'^[-*_]{3,}\s*$', line):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append("<hr>")
            i += 1
            continue

        # List items
        if re.match(r'^[\s]*[-*+]\s', line):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            text = re.sub(r'^[\s]*[-*+]\s', '', line)
            html_parts.append(f"<li>{inline_format(text)}</li>")
            i += 1
            continue

        # Numbered lists
        if re.match(r'^[\s]*\d+\.\s', line):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            text = re.sub(r'^[\s]*\d+\.\s', '', line)
            html_parts.append(f"<li>{inline_format(text)}</li>")
            i += 1
            continue

        # Close list if no longer in one
        if in_list and line.strip() == "":
            html_parts.append("</ul>")
            in_list = False
            i += 1
            continue

        # Empty line
        if line.strip() == "":
            i += 1
            continue

        # Skip HTML/div blocks and navigation links
        if line.strip().startswith("<") or "[🏠" in line or "←" in line or "→" in line:
            i += 1
            continue

        # Paragraph
        if in_list:
            html_parts.append("</ul>")
            in_list = False
        para_lines = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#") and not lines[i].startswith("```") and not lines[i].startswith("|") and not re.match(r'^[-*+]\s', lines[i]) and not re.match(r'^\d+\.\s', lines[i]):
            para_lines.append(lines[i])
            i += 1
        if para_lines:
            text = inline_format(" ".join(para_lines))
            html_parts.append(f"<p>{text}</p>")
        continue

    if in_list:
        html_parts.append("</ul>")
    if in_table:
        html_parts.append(render_table(table_rows))
    if in_code_block:
        code_content = html.escape("\n".join(code_lines))
        html_parts.append(f'<pre><code>{code_content}</code></pre>')

    return "\n".join(html_parts)


def render_table(rows):
    """Render table rows as HTML."""
    if not rows:
        return ""
    parts = ['<div class="table-wrap"><table>']
    parts.append("<thead><tr>")
    for cell in rows[0]:
        parts.append(f"<th>{inline_format(cell)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows[1:]:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{inline_format(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def inline_format(text):
    """Apply inline markdown formatting."""
    text = html.escape(text)
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Links
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    # Images
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1">', text)
    return text


def slugify(text):
    """Convert text to URL-friendly slug."""
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[\s]+', '-', text).strip('-')


def get_category(dir_name):
    """Get the category for a module directory."""
    prefix = dir_name[:2]
    for cat, prefixes in CATEGORIES.items():
        if prefix in prefixes:
            return cat
    return "Other"


def build_sidebar(modules):
    """Build sidebar navigation HTML."""
    categorized = {}
    for dir_name, path in modules:
        cat = get_category(dir_name)
        if cat not in categorized:
            categorized[cat] = []
        readme = (path / "README.md").read_text(encoding="utf-8")
        title = extract_title(readme)
        short_title = title.split("—")[-1].strip() if "—" in title else title.split(":")[-1].strip()
        if not short_title:
            short_title = dir_name.replace("_", " ").title()
        categorized[cat].append((dir_name, short_title))

    parts = []
    for cat_name in CATEGORIES:
        if cat_name not in categorized:
            continue
        parts.append(f'<div class="nav-category">')
        parts.append(f'<div class="nav-category-title">{cat_name}</div>')
        for dir_name, short_title in categorized[cat_name]:
            parts.append(
                f'<a class="nav-item" href="#module-{dir_name}" title="{short_title}">'
                f'<span class="nav-num">{dir_name[:2]}</span>{short_title}</a>'
            )
        parts.append("</div>")

    return "\n".join(parts)


def build_module_section(dir_name, path):
    """Build the HTML section for a single module."""
    readme_content = (path / "README.md").read_text(encoding="utf-8")
    title = extract_title(readme_content)
    files = extract_files_table(readme_content)

    # Strip navigation header (div blocks at top)
    lines = readme_content.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            start = i
            break
    clean_content = "\n".join(lines[start:])

    content_html = md_to_html(clean_content)

    # Build file links
    file_links = ""
    if files:
        file_links = '<div class="module-files"><h4>Source Files</h4><ul>'
        for fname, desc in files:
            if fname == "README.md":
                continue
            url = f"{REPO_URL}/{dir_name}/{fname}"
            file_links += f'<li><a href="{url}" target="_blank"><code>{fname}</code></a> — {desc}</li>'
        file_links += "</ul></div>"

    repo_link = f"{REPO_URL}/{dir_name}"

    return f'''
<section class="module-section" id="module-{dir_name}">
  <div class="module-header">
    <a href="{repo_link}" target="_blank" class="module-repo-link" title="View on GitHub">View Source</a>
  </div>
  <div class="module-content">
    {content_html}
  </div>
  {file_links}
</section>
'''


def build_html(modules):
    """Assemble the full HTML page."""
    sidebar = build_sidebar(modules)
    sections = []
    for dir_name, path in modules:
        sections.append(build_module_section(dir_name, path))

    return HTML_TEMPLATE.format(
        sidebar=sidebar,
        sections="\n".join(sections),
        module_count=len(modules),
    )


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyTorch: The Complete Reference Guide</title>
    <meta name="description" content="A structured, self-contained PyTorch course — 46 modules from foundations to production serving.">
    <link rel="icon" href="https://pytorch.org/favicon.ico">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.css">
    <style>
{css}
    </style>
</head>
<body>
    <nav class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="PyTorch" class="logo">
            <h2>Reference Guide</h2>
        </div>
        <div class="sidebar-search">
            <input type="text" id="search-input" placeholder="Search modules..." autocomplete="off">
        </div>
        <div class="sidebar-nav" id="sidebar-nav">
            {sidebar}
        </div>
    </nav>

    <button class="menu-toggle" id="menu-toggle" aria-label="Toggle navigation">
        <span></span><span></span><span></span>
    </button>

    <main class="main-content" id="main-content">
        <header class="hero">
            <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="PyTorch" class="hero-logo">
            <h1>PyTorch: The Complete Reference Guide</h1>
            <p class="hero-subtitle">From Absolute Beginner to Advanced Practitioner</p>
            <div class="hero-badges">
                <span class="badge badge-red">PyTorch 2.14+</span>
                <span class="badge badge-blue">{module_count} Modules</span>
                <span class="badge badge-green">127+ Code Examples</span>
                <span class="badge badge-purple">44 Notebooks</span>
                <span class="badge badge-orange">111,000+ Lines</span>
            </div>
            <p class="hero-desc">
                A structured, self-contained course organized into <strong>{module_count} modules</strong> and
                <strong>44 interactive notebooks</strong>. Each module contains detailed explanations, theory,
                formulas, runnable Python scripts, and a Jupyter playbook.
            </p>
            <div class="hero-links">
                <a href="https://github.com/subinz1/pytorch-reference-guide" class="btn btn-primary" target="_blank">GitHub Repository</a>
                <a href="#module-01_foundations" class="btn btn-secondary">Start Learning</a>
            </div>
        </header>

        {sections}

        <footer class="site-footer">
            <p>Built with PyTorch v2.14+ — Updated August 2026</p>
            <p><a href="https://github.com/subinz1/pytorch-reference-guide" target="_blank">subinz1/pytorch-reference-guide</a></p>
        </footer>
    </main>

    <button class="scroll-top" id="scroll-top" aria-label="Scroll to top">↑</button>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-yaml.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-cpp.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-toml.min.js"></script>
    <script>
{js}
    </script>
</body>
</html>
"""


CSS = """\
:root {
    --bg: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --border: #30363d;
    --link: #58a6ff;
    --link-hover: #79c0ff;
    --accent: #f78166;
    --code-bg: #1c2128;
    --sidebar-width: 280px;
    --header-height: 0px;
    --red: #f85149;
    --green: #3fb950;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --orange: #d29922;
}
[data-theme="light"] {
    --bg: #ffffff;
    --bg-secondary: #f6f8fa;
    --bg-tertiary: #eaeef2;
    --text: #1f2328;
    --text-muted: #656d76;
    --border: #d0d7de;
    --link: #0969da;
    --link-hover: #0550ae;
    --accent: #cf222e;
    --code-bg: #f6f8fa;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
    font-size: 16px;
}
a { color: var(--link); text-decoration: none; }
a:hover { color: var(--link-hover); text-decoration: underline; }
code {
    background: var(--code-bg);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.88em;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
}
pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    overflow-x: auto;
    margin: 16px 0;
    font-size: 0.85rem;
    line-height: 1.5;
}
pre code { background: none; padding: 0; font-size: inherit; }

/* Sidebar */
.sidebar {
    position: fixed;
    top: 0;
    left: 0;
    width: var(--sidebar-width);
    height: 100vh;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    z-index: 100;
    transition: transform 0.3s ease;
}
.sidebar-header {
    padding: 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}
.sidebar-header .logo { width: 32px; height: 32px; }
.sidebar-header h2 { font-size: 1rem; font-weight: 600; }
.sidebar-search { padding: 12px 16px; border-bottom: 1px solid var(--border); }
.sidebar-search input {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 0.85rem;
    outline: none;
}
.sidebar-search input:focus { border-color: var(--link); }
.sidebar-nav { padding: 8px 0; }
.nav-category { margin-bottom: 4px; }
.nav-category-title {
    padding: 8px 16px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
}
.nav-item {
    display: flex;
    align-items: center;
    padding: 6px 16px;
    font-size: 0.82rem;
    color: var(--text-muted);
    text-decoration: none;
    border-left: 3px solid transparent;
    transition: all 0.15s;
}
.nav-item:hover {
    color: var(--text);
    background: var(--bg-tertiary);
    text-decoration: none;
}
.nav-item.active {
    color: var(--link);
    border-left-color: var(--link);
    background: var(--bg-tertiary);
}
.nav-num {
    display: inline-block;
    width: 22px;
    font-weight: 600;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-right: 6px;
}

/* Main content */
.main-content {
    margin-left: var(--sidebar-width);
    min-height: 100vh;
}
.hero {
    text-align: center;
    padding: 80px 40px 60px;
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg) 100%);
    border-bottom: 1px solid var(--border);
}
.hero-logo { width: 80px; margin-bottom: 20px; }
.hero h1 { font-size: 2.4rem; margin-bottom: 8px; }
.hero-subtitle { color: var(--text-muted); font-size: 1.1rem; margin-bottom: 20px; }
.hero-badges { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; }
.badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-red { background: rgba(248,81,73,0.15); color: var(--red); }
.badge-blue { background: rgba(88,166,255,0.15); color: var(--blue); }
.badge-green { background: rgba(63,185,80,0.15); color: var(--green); }
.badge-purple { background: rgba(188,140,255,0.15); color: var(--purple); }
.badge-orange { background: rgba(210,153,34,0.15); color: var(--orange); }
.hero-desc { max-width: 600px; margin: 0 auto 24px; color: var(--text-muted); font-size: 0.95rem; }
.hero-links { display: flex; gap: 12px; justify-content: center; }
.btn {
    padding: 10px 24px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    text-decoration: none;
    transition: all 0.2s;
}
.btn-primary { background: var(--link); color: #fff; }
.btn-primary:hover { background: var(--link-hover); color: #fff; text-decoration: none; }
.btn-secondary { background: var(--bg-tertiary); color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--border); text-decoration: none; }

/* Module sections */
.module-section {
    padding: 48px 40px;
    border-bottom: 1px solid var(--border);
    max-width: 900px;
    margin: 0 auto;
}
.module-header {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 8px;
}
.module-repo-link {
    font-size: 0.8rem;
    padding: 4px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-muted);
}
.module-repo-link:hover { color: var(--link); border-color: var(--link); text-decoration: none; }
.module-content h1 { font-size: 1.8rem; margin: 0 0 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
.module-content h2 { font-size: 1.4rem; margin: 32px 0 12px; }
.module-content h3 { font-size: 1.15rem; margin: 24px 0 8px; }
.module-content h4 { font-size: 1rem; margin: 16px 0 8px; }
.module-content p { margin: 12px 0; }
.module-content ul, .module-content ol { padding-left: 24px; margin: 12px 0; }
.module-content li { margin: 4px 0; }
.module-content blockquote {
    border-left: 3px solid var(--link);
    padding: 12px 16px;
    margin: 16px 0;
    background: var(--bg-secondary);
    border-radius: 0 6px 6px 0;
    color: var(--text-muted);
}
.module-content hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
.module-content img { max-width: 100%; border-radius: 8px; }
.module-files {
    margin-top: 24px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
}
.module-files h4 { margin: 0 0 8px; font-size: 0.9rem; }
.module-files ul { list-style: none; padding: 0; }
.module-files li { padding: 4px 0; font-size: 0.85rem; }
.table-wrap { overflow-x: auto; margin: 16px 0; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
th, td { padding: 8px 12px; border: 1px solid var(--border); text-align: left; }
th { background: var(--bg-secondary); font-weight: 600; }

/* Footer */
.site-footer {
    padding: 40px;
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
}

/* Scroll to top */
.scroll-top {
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    color: var(--text);
    font-size: 1.2rem;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.3s;
    z-index: 50;
}
.scroll-top.visible { opacity: 1; }

/* Mobile menu toggle */
.menu-toggle {
    display: none;
    position: fixed;
    top: 12px;
    left: 12px;
    z-index: 200;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
    cursor: pointer;
    flex-direction: column;
    gap: 4px;
}
.menu-toggle span { display: block; width: 20px; height: 2px; background: var(--text); }

/* Responsive */
@media (max-width: 768px) {
    .sidebar { transform: translateX(-100%); }
    .sidebar.open { transform: translateX(0); }
    .main-content { margin-left: 0; }
    .menu-toggle { display: flex; }
    .module-section { padding: 32px 20px; }
    .hero { padding: 60px 20px 40px; }
    .hero h1 { font-size: 1.6rem; }
}
"""

JS = """\
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const menuToggle = document.getElementById('menu-toggle');
    const searchInput = document.getElementById('search-input');
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.module-section');
    const scrollTop = document.getElementById('scroll-top');
    const mainContent = document.getElementById('main-content');

    // Mobile menu toggle
    menuToggle.addEventListener('click', function() {
        sidebar.classList.toggle('open');
    });

    // Close sidebar on link click (mobile)
    navItems.forEach(function(item) {
        item.addEventListener('click', function() {
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('open');
            }
        });
    });

    // Search
    searchInput.addEventListener('input', function() {
        const query = this.value.toLowerCase();
        navItems.forEach(function(item) {
            const text = item.textContent.toLowerCase();
            const category = item.closest('.nav-category');
            if (text.includes(query) || query === '') {
                item.style.display = '';
            } else {
                item.style.display = 'none';
            }
        });
        // Show/hide categories
        document.querySelectorAll('.nav-category').forEach(function(cat) {
            const visibleItems = cat.querySelectorAll('.nav-item:not([style*=\"display: none\"])');
            cat.style.display = visibleItems.length > 0 || query === '' ? '' : 'none';
        });
    });

    // Active section tracking
    var observerOptions = { rootMargin: '-20% 0px -80% 0px' };
    var observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                navItems.forEach(function(item) { item.classList.remove('active'); });
                var id = entry.target.id;
                var activeLink = document.querySelector('.nav-item[href=\"#' + id + '\"]');
                if (activeLink) activeLink.classList.add('active');
            }
        });
    }, observerOptions);
    sections.forEach(function(section) { observer.observe(section); });

    // Scroll to top button
    window.addEventListener('scroll', function() {
        if (window.scrollY > 500) {
            scrollTop.classList.add('visible');
        } else {
            scrollTop.classList.remove('visible');
        }
    });
    scrollTop.addEventListener('click', function() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Prism re-highlight
    if (typeof Prism !== 'undefined') {
        Prism.highlightAll();
    }
});
"""


def main():
    print("Discovering modules...")
    modules = discover_modules()
    print(f"Found {len(modules)} modules")

    print("Building HTML...")
    html_content = build_html(modules)

    # Inject CSS and JS
    html_content = html_content.replace("{css}", CSS)
    html_content = html_content.replace("{js}", JS)

    output_path = REPO_ROOT / "docs" / "index.html"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Generated: {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
