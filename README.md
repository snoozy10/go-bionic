Sample pdf: "Bible de Genève, 1564 (fonts and typography)"
<br>
Source: [Github page of pdf2htmlEX](https://github.com/pdf2htmlEX/pdf2htmlEX?tab=readme-ov-file)
# Result
Input: [geneve_1564.pdf](/sample_pdf/geneve_1564.pdf)
<br>
Output: [boldened.pdf](boldened.pdf)
# Initial idea:
raster pdf -> high-fidelity html (pdf2htmlEX) -> bionified html
## Issues:
pdf2htmlEX simply encodes ocr data behind raster pdf.
While this makes the pdf/html text selectable, it doesn't help with going bionic.
Detecting font is difficult for non-standard fonts.
<br><br>
# Modified idea:
raster pdf -> bionified raster pdf
<br><br>
# Future Steps:
- add concurrency/parallelism as boldening of words are inherently independent
- robust handling of pages with both horizontal and vertical orientations (currently handles only horizontal text)
  -- maybe OSD per ROI instead of entire page? perhaps both??
- handle colored boldening
<br><br>
### P.S. boldening = bolding. Personally prefer the term boldening :)
  


