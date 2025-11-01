# Table of Contents
- [Goal](#goal)
- [Result](#result)
- [Initial idea](#initial-idea)
  - [Issues with the initial idea](#issues-with-the-initial-idea)
- [Current implementation](#current-implementation)
- [Future improvements](#future-improvements)
<br><br> 
# Goal
Apply text-formatting that mimics the bionic reading font (also known as ADHD font or fast font) to a PDF _while maintaining layout fidelity_.
<br><br>
Sample pdf 1: "Bible de Genève, 1564 (fonts and typography)"
<br>
Source: [Github page of pdf2htmlEX](https://github.com/pdf2htmlEX/pdf2htmlEX?tab=readme-ov-file)
<br><br>
Sample pdf 2: "Deep Learning: An Introduction for Applied Mathematicians"
<br>
Source: [Link to Paper](https://arxiv.org/pdf/1801.05894)
<br><br>
> [!Note]
> This project was created specifically for raster PDFs e.g. scanned pages saved as PDFs. There are much better ways to handle "true"/"vector" PDFs that are faster, and can apply character-wise boldening.
# Result
Input 1: [geneve_1564.pdf](/sample_pdfs/geneve_1564.pdf)
<br>
Output 1: [geneve_1564_boldened.pdf](geneve_1564_boldened.pdf)
<br><br>
Input 2: [deep_learning.pdf](/sample_pdfs/deep_learning.pdf)
<br>
Output 2: [deep_learning_boldened.pdf](deep_learning_boldened.pdf)
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
- Filter out ROI with high overlap (i.e. child ROI)
- Accurate character counting and boldening. Currently boldens exactly first-(ratio) of a word, disregarding character edges.
<br><br>
> [!Note]
> boldening = bolding. Boldening used for personal preference 😄
  


