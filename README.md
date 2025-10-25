# Table of Contents
- [Goal](#goal)
- [Result](#result)
- [Initial idea](#initial-idea)
  - [Issues with the initial idea](#issues-with-the-initial-idea)
- [Current implementation](#current-implementation)
- [Future improvements](#future-improvements)
<br><br> 
# Goal
Apply bionic reading font (font that aids faster reading; also known as adhd font or fast font) to a PDF _while maintaining layout fidelity_.
<br><br>
Sample pdf: "Bible de Genève, 1564 (fonts and typography)"
<br>
Source: [Github page of pdf2htmlEX](https://github.com/pdf2htmlEX/pdf2htmlEX?tab=readme-ov-file)
<br><br>
# Result
Input: [geneve_1564.pdf](/sample_pdf/geneve_1564.pdf)
<br>
Output: [boldened.pdf](boldened.pdf)
<br><br>
# Initial idea
`pdf` -> `high-fidelity html` (`pdf2htmlEX`/`tesseract`) -> `html with bionic font`
## Issues with the initial idea
- `pdf2htmlEX` simply encodes OCR data in a layer independent of the pdf.
While this makes the pdf/html text selectable, it doesn't really help with going bionic.
- Resulting HTML when using `pdf2htmlEX` is messy and difficult to parse (words split by random white spaces, difficult to detect words).
- Maintaining fidelity was a key goal of this program, so using the HTML output of `pyslibtesseract` or `tesseract` didn't solve the issue for the following reasons:
  - `tesseract` doesn't provide font information.
  - Detecting font is challenging for non-standard fonts (although, this is an issue I plan to revisit)
<br><br>
# Current implementation
`pdf` -> `pdf with partially boldened words`
<br><br>
# Future improvements
- Add concurrency/parallelism as boldening of words are inherently independent
- Robust handling of pages with both horizontal and vertical orientations (currently handles only horizontal text)
  - Maybe OSD per ROI instead of entire page? Perhaps both??
- Handle colored boldening
- Filter out ROI with high overlap (i.e. child ROI)
- Accurate character counting and boldening. Currently boldens exactly first-(ratio) of a word, disregarding character edges.
<br><br>
> [!Note]
> boldening = bolding. Boldening used for personal preference 😄
  


