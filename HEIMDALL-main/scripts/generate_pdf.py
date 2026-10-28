"""
HEIMDALL Academic Report PDF Generator
=======================================
Converts `reports/HEIMDALL_PROJECT_REPORT.md` into a publication-grade,
beautifully styled PDF report using MathJax for formulas, clean typography,
and Google Chrome / Microsoft Edge headless PDF rendering.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
import markdown

ROOT_DIR = Path(__file__).resolve().parents[1]
MD_PATH = ROOT_DIR / "reports" / "HEIMDALL_PROJECT_REPORT.md"
HTML_PATH = ROOT_DIR / "reports" / "HEIMDALL_PROJECT_REPORT.html"
PDF_PATH = ROOT_DIR / "reports" / "HEIMDALL_PROJECT_REPORT.pdf"

# Find Chrome or Edge
CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

browser_exe = None
for p in CHROME_PATHS:
    if os.path.exists(p):
        browser_exe = p
        break

if not browser_exe:
    raise RuntimeError("Neither Google Chrome nor Microsoft Edge was found on this system.")

print(f"[*] Selected rendering engine: {browser_exe}")

# Read Markdown
md_content = MD_PATH.read_text(encoding="utf-8")

# Convert Markdown to HTML
html_body = markdown.markdown(
    md_content,
    extensions=["tables", "fenced_code", "toc", "attr_list"],
)

# Publication-grade CSS styling
HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HEIMDALL: Host Environment Intelligence for Monitoring, Diagnostics and Anomaly Learning</title>
<!-- MathJax for rendering LaTeX formulas -->
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

@page {{
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
  @bottom-right {{
    content: counter(page);
    font-family: 'Inter', sans-serif;
    font-size: 8.5pt;
    color: #64748b;
  }}
  @bottom-left {{
    content: "24CYS305 Artificial Intelligence and Neural Networks — Project Report";
    font-family: 'Inter', sans-serif;
    font-size: 8pt;
    color: #64748b;
  }}
}}

*, *::before, *::after {{
  box-sizing: border-box;
}}

body {{
  font-family: 'Crimson Pro', Georgia, serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #1e293b;
  background-color: #ffffff;
  margin: 0;
  padding: 0;
}}

h1, h2, h3, h4, h5, h6 {{
  font-family: 'Inter', sans-serif;
  color: #0f172a;
  page-break-after: avoid;
  page-break-inside: avoid;
}}

h1 {{
  font-size: 20pt;
  font-weight: 700;
  line-height: 1.25;
  margin-top: 0;
  margin-bottom: 6px;
  text-align: center;
  color: #091e42;
}}

h2 {{
  font-size: 13.5pt;
  font-weight: 600;
  border-bottom: 1.5px solid #cbd5e1;
  padding-bottom: 4px;
  margin-top: 22px;
  margin-bottom: 10px;
  color: #0f2744;
}}

h3 {{
  font-size: 11.5pt;
  font-weight: 600;
  margin-top: 14px;
  margin-bottom: 6px;
  color: #1e3a8a;
}}

h4 {{
  font-size: 10.5pt;
  font-weight: 600;
  margin-top: 10px;
  margin-bottom: 4px;
  color: #334155;
}}

p {{
  margin-top: 0;
  margin-bottom: 9px;
  text-align: justify;
  text-justify: inter-word;
}}

.subtitle {{
  text-align: center;
  font-family: 'Inter', sans-serif;
  font-size: 12pt;
  font-weight: 500;
  color: #334155;
  margin-bottom: 12px;
}}

.meta-box {{
  font-family: 'Inter', sans-serif;
  font-size: 8.5pt;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 10px 14px;
  margin-bottom: 18px;
  line-height: 1.6;
  color: #475569;
}}

.meta-box strong {{
  color: #0f172a;
}}

.abstract-box {{
  background-color: #f1f5f9;
  border-left: 3.5px solid #0284c7;
  padding: 10px 14px;
  margin: 14px 0 18px 0;
  font-size: 9.5pt;
  line-height: 1.5;
  text-align: justify;
  border-radius: 0 4px 4px 0;
}}

.abstract-box strong {{
  font-family: 'Inter', sans-serif;
  color: #0369a1;
  text-transform: uppercase;
  font-size: 8.5pt;
  letter-spacing: 0.5px;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0 16px 0;
  font-family: 'Inter', sans-serif;
  font-size: 8.5pt;
  line-height: 1.4;
  page-break-inside: avoid;
}}

th, td {{
  padding: 5px 8px;
  text-align: left;
  vertical-align: middle;
  border-bottom: 1px solid #e2e8f0;
}}

th {{
  background-color: #f8fafc;
  color: #0f172a;
  font-weight: 600;
  border-top: 1.5px solid #cbd5e1;
  border-bottom: 1.5px solid #94a3b8;
  font-size: 8.5pt;
}}

tr:hover td {{
  background-color: #f8fafc;
}}

code {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 8.5pt;
  background-color: #f1f5f9;
  color: #0f172a;
  padding: 1px 4px;
  border-radius: 3px;
  border: 1px solid #e2e8f0;
}}

pre {{
  background-color: #0f172a;
  color: #f8fafc;
  padding: 10px 14px;
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 7.8pt;
  line-height: 1.4;
  overflow-x: auto;
  margin: 10px 0 14px 0;
  page-break-inside: avoid;
  border: 1px solid #1e293b;
}}

pre code {{
  background-color: transparent;
  color: inherit;
  padding: 0;
  border: none;
}}

ul, ol {{
  margin-top: 0;
  margin-bottom: 10px;
  padding-left: 20px;
}}

li {{
  margin-bottom: 4px;
  text-align: justify;
}}

hr {{
  border: 0;
  border-top: 1px solid #e2e8f0;
  margin: 18px 0;
}}

blockquote {{
  margin: 10px 0;
  padding: 6px 14px;
  border-left: 3px solid #64748b;
  background: #f8fafc;
  font-style: italic;
  color: #334155;
}}
</style>
</head>
<body>

{html_body}

</body>
</html>
"""

HTML_PATH.write_text(HTML_TEMPLATE, encoding="utf-8")
print(f"[*] Generated styled HTML: {HTML_PATH}")

# Render to PDF using Headless Chrome / Edge
print("[*] Launching headless browser for PDF compilation...")
cmd = [
    browser_exe,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-extensions",
    "--run-all-compositor-stages-before-draw",
    "--print-to-pdf-no-header",
    f"--print-to-pdf={PDF_PATH}",
    str(HTML_PATH.resolve()),
]

res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
if res.returncode != 0:
    print(f"[!] Warning during headless invocation: {res.stderr}")

# Check result
if PDF_PATH.exists() and PDF_PATH.stat().st_size > 1000:
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"[+] SUCCESS! Generated publication-quality PDF:")
    print(f"    Path: {PDF_PATH}")
    print(f"    Size: {size_kb:.1f} KB")
else:
    raise RuntimeError(f"PDF creation failed or output was empty: {res.stderr}")
