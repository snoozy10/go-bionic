# Goal:
Apply bionic font to a PDF while maintaining layout fidelity.
<br>
Sample pdf: "Bible de Genève, 1564 (fonts and typography)"
<br>
Source: [Github page of pdf2htmlEX](https://github.com/pdf2htmlEX/pdf2htmlEX?tab=readme-ov-file)
# Result
Input: [geneve_1564.pdf](/sample_pdf/geneve_1564.pdf)
<br>
Output: [boldened.pdf](boldened.pdf)
# Initial idea:
`raster pdf` -> `high-fidelity html` (pdf2htmlEX) -> `html with bionic font`
## Issues:
- pdf2htmlEX simply encodes OCR data in a layer independent of the pdf.
While this makes the pdf/html text selectable, it doesn't really help with going bionic.
- Resulting HTML is messy and difficult to parse (words split by random white spaces, difficult to detect words).
- Detecting font is challenging for non-standard fonts. So maintaining high-fidelity with proper word/paragraph separation is a tough problem to solve.
<br><br>
# Modified idea:
`raster pdf` -> `raster pdf with bionic font`
<br><br>
# Future Steps:
- add concurrency/parallelism as boldening of words are inherently independent
- robust handling of pages with both horizontal and vertical orientations (currently handles only horizontal text)
  -- maybe OSD per ROI instead of entire page? perhaps both??
- handle colored boldening
<br><br>
> [!Note]
> boldening = bolding. Personally prefer the term boldening 😄
  


