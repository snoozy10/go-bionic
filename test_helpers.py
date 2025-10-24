from PIL import Image
import os
from pathlib import Path
import pymupdf
import numpy as np
import pytesseract
import cv2
from main import get_rois, bolden_roi, show_image


def save_og_page_imgs() -> None:
    """
    Helper function for testing purposes. Converts target pdf pages to images
    """
    root = Path(__file__).resolve().parent
    pdf_path = os.path.join(root, "sample_pdf", "geneve_1564.pdf")
    doc = pymupdf.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]

        # get RGB pixmap
        pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csRGB)
        og_page_img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        Image.fromarray(og_page_img).save(f"page_{page_index}.png")


def test_orientation() -> None:
    """
    Helper function to test pytesseract's OSD and rotate behavior
    """
    img = Image.open("page_0.png").convert('RGB')

    osd = pytesseract.image_to_osd(img, output_type='dict')

    rotate = osd['rotate']
    print(rotate)

    # Rotate the image to correct the orientation
    im_fixed = img.copy().rotate(360 - rotate, expand=True)

    # convert from RGB to BGR
    og_page_img = cv2.cvtColor(np.array(im_fixed), cv2.COLOR_RGB2BGR)

    segmented_image, rois = get_rois(og_page_img)

    # Display image with marked regions to test
    # show_image(segmented_image)

    for bbox in rois:
        bolden_roi(og_page_img, bbox)

    # Display image with bold regions to test
    show_image(og_page_img)
    og_page_img = cv2.cvtColor(og_page_img, cv2.COLOR_BGR2RGB)

    Image.fromarray(og_page_img).save(f"page_0_ori.png")
