from PIL import Image
import os
from pathlib import Path
import pymupdf
import numpy as np
import pytesseract
import cv2
from main import get_rois, bolden_roi, bolden_image, show_image, get_roi_data, LANGUAGE


def run_data_first():
    root = Path(__file__).resolve().parent
    pdf_path = os.path.join(root, "sample_pdf", "test.pdf")

    doc = pymupdf.open(pdf_path)
    page_index = 0
    page = doc[page_index]

    # get RGB pixmap
    pix = page.get_pixmap(dpi=300)
    og_page_img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    segmented_image, rois = get_rois(og_page_img)
    data = pytesseract.image_to_data(og_page_img, output_type=pytesseract.Output.DICT, lang=LANGUAGE)

    for roi in rois:
        test_result = get_roi_data(data, roi)
        print(test_result["text"])


def save_og_page_imgs(pdf_name="geneve_1564.pdf", page_count=1) -> None:
    """
    Helper function for testing purposes. Converts target pdf pages to images
    """
    root = Path(__file__).resolve().parent
    pdf_path = os.path.join(root, "sample_pdf", pdf_name)
    doc = pymupdf.open(pdf_path)
    page_count = min(page_count, len(doc))
    for page_index in range(page_count):
        page = doc[page_index]

        # get RGB pixmap
        pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csRGB)
        og_page_img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        Image.fromarray(og_page_img).save(f"test_page_{page_index}.png")


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
    data = pytesseract.image_to_data(og_page_img,  output_type=pytesseract.Output.DICT, lang=LANGUAGE)
    # Display image with marked regions to test
    # show_image(segmented_image)

    for bbox in rois:
        roi_data = get_roi_data(data, bbox)
        bolden_roi(og_page_img, roi_data)

    # Display image with bold regions to test
    show_image(og_page_img)
    og_page_img = cv2.cvtColor(og_page_img, cv2.COLOR_BGR2RGB)

    Image.fromarray(og_page_img).save(f"page_0_ori.png")


if __name__ == "__main__":
    save_og_page_imgs(pdf_name="test.pdf", page_count=1)
    og_page_img = Image.open("test_page_0.png").convert('RGB')
    og_page_img = bolden_image(np.array(og_page_img))
    Image.fromarray(og_page_img).save(f"test_page_0.png")
