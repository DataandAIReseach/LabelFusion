# LabelFusion arXiv Paper Package

This folder contains a minimal arXiv-friendly LaTeX setup for a ~5 page paper about LabelFusion.

## Files
- `main.tex` — LaTeX source (uses standard `article` with 1-inch margins, common for arXiv submissions)
- `references.bib` — Bibliography entries for cited works

## Build
Use a LaTeX toolchain (macOS example shown):

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The output `main.pdf` is suitable for arXiv upload. If arXiv flags fonts, include the generated `.bbl` and consider `-interaction=nonstopmode`.

## Notes
- Update authors, affiliations, and date as needed.
- If you prefer the official arXiv `neurips`/`lncs`/`ieee` styles, replace the preamble accordingly, but arXiv accepts standard `article` class.
- Keep figures (if added) under ~10MB total; prefer vector PDF or high-quality PNG.
