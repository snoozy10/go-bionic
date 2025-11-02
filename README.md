# Table of Contents
- [Goal](#goal)
- [Results](#results)
- [Initial idea](#initial-idea)
  - [Issues with the initial idea](#issues-with-the-initial-idea)
- [Current implementation](#current-implementation)
- [Future improvements](#future-improvements)
<br><br>

# Goal
Apply text-formatting that mimics the bionic reading font (also known as ADHD font or fast font) to a PDF _while maintaining layout fidelity_.
<br><br>

> [!Note]
> <p>This project was created specifically for raster PDFs e.g. scanned pages saved as PDFs. Although the current implementation <em>can</em> handle vector PDFs, there are much faster/simpler ways to do so.</p>
&nbsp;
# Results
### 1.
Input: [geneve_1564.pdf](/sample_pdfs/geneve_1564.pdf)
<br>
Output: [geneve_1564_boldened.pdf](/sample_output/geneve_1564_boldened.pdf)
<br>
Bible de Genève, 1564 (fonts and typography)
<br>
Source: [Github: pdf2htmlEX](https://github.com/pdf2htmlEX/pdf2htmlEX?tab=readme-ov-file)
<br><br>

### 2.
Input: [deep_learning.pdf](/sample_pdfs/deep_learning.pdf)
<br>
Output: [deep_learning_boldened.pdf](/sample_output/deep_learning_boldened.pdf)
<br>
Deep Learning: An Introduction for Applied Mathematicians
<br>
Source: [Link to Paper](https://arxiv.org/pdf/1801.05894)
<br><br>

### 3.
Input: [public_water_mass_mailing.pdf](/sample_pdfs/public_water_mass_mailing.pdf)
<br>
Output: [public_water_mass_mailing_boldened.pdf](/sample_output/public_water_mass_mailing_boldened.pdf)
<br>
Sample Scanned Document: ScanSnap SV600 (Ricoh)
<br>
Source: [Link to Document](https://www.pfu.ricoh.com/global/scanners/downloads/v5/sv600_c_normal.pdf)
<br><br>

### 4.
Input: [sv600_c_normal.pdf](/sample_pdfs/sv600_c_normal.pdf)
<br>
Output: [sv600_c_normal_boldened.pdf](/sample_output/sv600_c_normal_boldened.pdf)
<br>
Sample Scanned Document: Public Water Mass Mailing
<br>
Source: [Github: Text-Extraction-Scanned-Pdf](https://github.com/fraponyo94/Text-Extraction-Scanned-Pdf/tree/master/sample-scanned-pdfs)
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
- Robust handling of pages with both horizontal and vertical orientations (currently handles only horizontal text)
  - Maybe OSD per ROI instead of entire page? Perhaps both??
- Filter out ROI with high overlap (i.e. child ROI)
- Option to choose output color (grayscale/RGB); implement simpler pipeline for grayscale output
- Accurate character counting and boldening. Currently boldens exactly first-(ratio) of a word, disregarding character edges.
<br><br>
> [!Note]
> boldening = bolding. Boldening used for personal preference 😄
  


