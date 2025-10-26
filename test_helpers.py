from PIL import Image
import pymupdf
import numpy as np
import pytesseract
import cv2
from main import get_path, get_rois, bolden_roi, bolden_image, show_image, get_roi_data, LANGUAGE
TEST_FOLDER = "test_folder"


def run_data_first(
        pdf_path: str
):
    doc = pymupdf.open(pdf_path)
    page_index = 0
    page = doc[page_index]

    # get RGB pixmap
    pix = page.get_pixmap(dpi=300)
    og_page_img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    rois = get_rois(image=og_page_img)
    data = pytesseract.image_to_data(og_page_img, output_type=pytesseract.Output.DICT, lang=LANGUAGE)

    for roi in rois:
        test_result = get_roi_data(data=data, roi=roi)
        print(test_result["text"])


def save_og_page_imgs(
        pdf_path: str,
        page_count: int = 1
) -> str:
    """
    Helper function for testing purposes. Converts target pdf pages to images
    """
    doc = pymupdf.open(pdf_path)
    page_count = min(page_count, len(doc))
    for page_index in range(page_count):
        page = doc[page_index]

        # get RGB pixmap
        pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csRGB)
        og_page_img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        filepath = get_path(folder=TEST_FOLDER, filename="test_page.png", input_mode=False)
        Image.fromarray(og_page_img).save(filepath)
        return filepath


def test_orientation(
        image_path: str
) -> None:
    """
    Helper function to test pytesseract's OSD and rotate behavior
    """
    img = Image.open(image_path).convert('RGB')

    osd = pytesseract.image_to_osd(img, output_type='dict')

    rotate = osd['rotate']
    print(rotate)

    # Rotate the image to correct the orientation
    im_fixed = img.copy().rotate(360 - rotate, expand=True)

    # convert from RGB to BGR
    og_page_img = cv2.cvtColor(np.array(im_fixed), cv2.COLOR_RGB2BGR)

    rois = get_rois(image=og_page_img)
    data = pytesseract.image_to_data(og_page_img,  output_type=pytesseract.Output.DICT, lang=LANGUAGE)
    # Display image with marked regions to test
    # show_image(segmented_image)

    for roi in rois:
        roi_data = get_roi_data(data=data, roi=roi)
        bolden_roi(image=og_page_img, roi_data=roi_data)

    # Display image with bold regions to test
    show_image(image=og_page_img)
    og_page_img = cv2.cvtColor(og_page_img, cv2.COLOR_BGR2RGB)
    filepath = get_path(folder=TEST_FOLDER, filename="test_ori.png", input_mode=False)

    Image.fromarray(og_page_img).save(filepath)


if __name__ == "__main__":
    pdf_path = get_path(folder=TEST_FOLDER, filename="test.pdf", input_mode=True)
    filepath = save_og_page_imgs(pdf_path, page_count=1)
    og_page_img = Image.open(filepath).convert('RGB')
    og_page_img = bolden_image(image=np.array(og_page_img))
    Image.fromarray(og_page_img).save(filepath)
