#!/usr/bin/env python
"""Build the JPEG AI project report PDF.

Markdown -> HTML (python-markdown) -> PDF (weasyprint).

The HTML is written to the PROJECT ROOT so that relative image paths like
results/bench_kodak.png resolve without rewriting.

Run:
    .venv/bin/python report/build.py
then:
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
        /opt/anaconda3/bin/weasyprint JPEG-AI-Report.html JPEG-AI-Report.pdf
"""
from __future__ import annotations

import pathlib
import sys

import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "report"
HTML_OUT = ROOT / "JPEG-AI-Report.html"

ORDER = [
    "00-front.md",
    "01-part1.md",
    "02-part2-maths.md",
    "03-part2-arch.md",
    "04-part3-provenance.md",
    "05-part4-build.md",
    "06-part5-results.md",
    "07-part6-failures.md",
    "08-part7-next.md",
    "09-appendices.md",
]

CSS = r"""
@page {
  size: A4;
  margin: 20mm 17mm 20mm 17mm;
  @top-center {
    content: "JPEG AI — Building a Learning-Based Image Codec from a Research Paper";
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 7.5pt;
    color: #8a8a8a;
    padding-bottom: 3mm;
    border-bottom: 0.4pt solid #dcdcdc;
    width: 100%;
  }
  @bottom-center {
    content: counter(page);
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 8.5pt;
    color: #6a6a6a;
    padding-top: 3mm;
  }
}
@page :first {
  @top-center { content: none; border-bottom: none; }
  @bottom-center { content: none; }
}

html { font-size: 10pt; }
body {
  font-family: "Palatino", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  line-height: 1.46;
  color: #16181c;
  text-align: left;
  hyphens: none;
}

/* ---------- cover ---------- */
.cover { page-break-after: always; padding-top: 34mm; text-align: center; }
.cover .cover-title {
  font-size: 46pt; letter-spacing: 3pt; margin: 0 0 2mm 0;
  color: #0b3d66; font-weight: 700;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cover .cover-sub {
  font-size: 17pt; font-weight: 400; margin: 0 0 4mm 0;
  color: #24303a; line-height: 1.32;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cover .cover-tag {
  font-size: 10.5pt; font-weight: 400; font-style: italic;
  color: #5a6570; margin: 0 auto 14mm auto; max-width: 128mm;
}
.cover-meta {
  text-align: left; max-width: 138mm; margin: 0 auto 12mm auto;
  font-size: 9pt; line-height: 1.55; color: #2a3238;
  border-top: 0.8pt solid #cfd6dc; border-bottom: 0.8pt solid #cfd6dc;
  padding: 5mm 0;
}
.cover-meta p { margin: 0 0 2.2mm 0; }
.cover-note {
  text-align: left; max-width: 138mm; margin: 0 auto;
  font-size: 8.6pt; line-height: 1.5; color: #43505a;
}
.cover-note p { margin: 0 0 3mm 0; }

/* ---------- headings ---------- */
h1 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 21pt; font-weight: 700; color: #0b3d66;
  margin: 0 0 6mm 0; padding-bottom: 2.5mm;
  border-bottom: 1.4pt solid #0b3d66;
  page-break-after: avoid; page-break-before: auto;
}
h2 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 14pt; font-weight: 700; color: #12314d;
  margin: 8mm 0 3mm 0; page-break-after: avoid;
}
h3 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 11pt; font-weight: 700; color: #1d3a52;
  margin: 6mm 0 2.2mm 0; page-break-after: avoid;
}
h4 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9.6pt; font-weight: 700; color: #35424e;
  margin: 4.5mm 0 1.6mm 0; page-break-after: avoid;
}

p { margin: 0 0 2.6mm 0; orphans: 3; widows: 3; }

/* ---------- part-opening italic abstracts ---------- */
h1 + p em, h1 + em {
  color: #4a5560;
}

/* ---------- lists ---------- */
ul, ol { margin: 0 0 3mm 0; padding-left: 6.5mm; }
li { margin-bottom: 1.1mm; }

/* ---------- code ---------- */
code {
  font-family: "SF Mono", "Menlo", "DejaVu Sans Mono", monospace;
  font-size: 8.4pt; background: #f2f4f6; padding: 0.3mm 0.9mm;
  border-radius: 1.2pt; color: #0f2d47;
}
pre {
  background: #f7f9fa; border: 0.5pt solid #dbe2e8;
  border-left: 2.2pt solid #0b3d66;
  padding: 2.6mm 3.2mm; margin: 0 0 3.4mm 0;
  font-size: 8.1pt; line-height: 1.36;
  overflow-wrap: break-word; white-space: pre-wrap;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8.1pt; color: #16283a; }

/* ---------- tables ---------- */
table {
  border-collapse: collapse; width: 100%;
  margin: 0 0 4mm 0; font-size: 8.3pt;
  page-break-inside: avoid;
}
thead { background: #0b3d66; color: #ffffff; }
thead th {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-weight: 700; text-align: left;
  padding: 1.5mm 2mm; border: 0.4pt solid #0b3d66;
  font-size: 8pt;
}
tbody td {
  padding: 1.3mm 2mm; border: 0.4pt solid #d3dae0;
  vertical-align: top;
}
tbody tr:nth-child(even) { background: #f5f7f9; }
table code { font-size: 7.8pt; background: #e8edf1; }

/* ---------- blockquote ---------- */
blockquote {
  margin: 0 0 3.4mm 0; padding: 2.4mm 3.4mm;
  background: #f4f8fb; border-left: 2.4pt solid #2d6fa3;
  color: #1b3348; font-size: 9.4pt;
}
blockquote p { margin: 0 0 1.6mm 0; }
blockquote p:last-child { margin-bottom: 0; }

/* ---------- images ---------- */
img {
  max-width: 100%; height: auto; display: block;
  margin: 3mm auto 2mm auto;
  border: 0.5pt solid #d3dae0;
  page-break-inside: avoid;
}

/* ---------- horizontal rule ---------- */
hr { border: none; border-top: 0.6pt solid #cfd6dc; margin: 6mm 0; }

/* ---------- page breaks ---------- */
.page-break { page-break-before: always; }

/* ---------- table of contents ---------- */
.toc { font-size: 9pt; }
.toc ul { list-style: none; padding-left: 4mm; margin: 0; }
.toc > ul { padding-left: 0; }
.toc > ul > li {
  margin-top: 1.6mm; font-weight: 700;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9.2pt; color: #0b3d66;
}
.toc > ul > li > ul > li {
  font-weight: 400; font-family: "Palatino", Georgia, serif;
  font-size: 8.7pt; color: #24303a;
}
.toc > ul > li > ul > li > ul { display: none; }
.toc a { text-decoration: none; color: inherit; }
.toc li a::after {
  content: " " leader('.') " " target-counter(attr(href), page);
  color: #7c8791; font-weight: 400;
}

/* keep short sections together where we can */
h2, h3 { page-break-inside: avoid; }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>JPEG AI — Building a Learning-Based Image Codec from a Research Paper</title>
<style>
{css}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> int:
    missing = [n for n in ORDER if not (SRC / n).exists()]
    if missing:
        print("missing source files: " + ", ".join(missing))
        return 1

    chunks = []
    for name in ORDER:
        text = (SRC / name).read_text(encoding="utf-8")
        chunks.append(text.rstrip() + "\n")
    combined = "\n\n".join(chunks)

    md_path = SRC / "_combined.md"
    md_path.write_text(combined, encoding="utf-8")

    md = markdown.Markdown(
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "attr_list",
            "md_in_html",
            "sane_lists",
        ],
        extension_configs={"toc": {"toc_depth": "1-3"}},
    )
    body = md.convert(combined)

    HTML_OUT.write_text(
        HTML_TEMPLATE.format(css=CSS, body=body), encoding="utf-8"
    )

    words = len(combined.split())
    print(f"source   {len(ORDER)} files, {len(combined.splitlines()):,} lines, {words:,} words")
    print(f"markdown {md_path.relative_to(ROOT)}")
    print(f"html     {HTML_OUT.relative_to(ROOT)}  ({HTML_OUT.stat().st_size:,} bytes)")
    print()
    print("now run:")
    print("  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \\")
    print("    /opt/anaconda3/bin/weasyprint JPEG-AI-Report.html JPEG-AI-Report.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
