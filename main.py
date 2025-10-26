from __future__ import annotations

import typing
from typing import Any

import cv2  # openCV https://docs.opencv.org/4.12.0/
import pytesseract
import pymupdf
import numpy as np
from PIL import Image

import os
from pathlib import Path

from numpy import floating, ndarray

# threshold of regions to ignore
THRESHOLD_REGION_IGNORE = 40

# max number of words to consider for mean/median text height in a text-region
MAX_WORD_SAMPLES = 10

# check and correct the orientation of pdfs before going bionic. set to True for skewed layouts
CHECK_ORIENTATION = False

# ratio of word-width getting boldened
BOLD_RATIO = 0.5

# languages in the pdf to process by pytesseract
# https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
LANGUAGE = "deu+eng"

# magic number to scale kernel size wrt text height.
# useful for large texts. higher = weaker darkening wrt height.
KERNEL_MAGIC_NUMBER = 16

# multiplier for alpha during darkening, [0, 1]. higher = darker.
ALPHA_MAGIC_NUMBER = 0.85

# opacity of the lightened half (right half) of words
LIGHTENING_OPACITY = 0.80


def get_path(
        filename: str,
        folder: str = None,
        input_mode: bool = True
) -> str:
    """
    Returns a valid filepath for io operations
    :param filename: name of the io file
    :param folder: folder inside root containing the io file. can be None if file is in root folder
    :param input_mode: boolean indicating if the filepath is for input operation, or output
    :returns:
        resulting filepath that is valid for io operation
    """
    root = Path(__file__).resolve().parent

    if folder is not None:
        # possibility for error if folder name is nested
        # maybe allow nested folders and use mkdirs instead of mkdir?
        # To-do
        folder_path = os.path.join(root, folder)
    else:
        folder_path = root

    filepath = os.path.join(folder_path, filename)

    if input_mode:
        assert os.path.exists(filepath)
    else:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
    return filepath


def resize_with_aspect_ratio(
        image: ndarray[tuple[typing.Any, ...], np.dtype],
        width: int | None = None,
        height: int | None = None,
        inter: int = cv2.INTER_AREA
) -> ndarray[tuple[typing.Any, ...], np.dtype]:
    """
    Resizes the opencv image viewing window
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def show_image(
        image: ndarray[tuple[typing.Any, ...], np.dtype],
        width: int = 450,
        resize: bool = True,
        title: str = "Show Image"
) -> None:
    """
    Displays an image using opencv's image viewer
    """
    if resize:
        image = resize_with_aspect_ratio(image=image, width=width)
    cv2.imshow(title, image)
    cv2.waitKey()


def bolden_crop_region(
        image: ndarray[tuple[typing.Any, ...], np.dtype],
        text_height: floating[Any],

) -> ndarray[tuple[typing.Any, ...], np.dtype]:
    """
    Increases stroke width in the cropped region
    :param image: cropped image of the word-region to bolden
    :param text_height: mean/median text height in roi
    :returns:
        cropped image with thicker strokes
    """

    # convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -- threshold mask, inverted
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # choose kernel size by text height. use a *magic number*
    k = max(1, int(text_height / KERNEL_MAGIC_NUMBER))

    # use a circular or disk-shaped structuring element for morphological operations
    # useful for smoothing out corners, removing small circular noise, etc
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # bolden the first-(ratio) of the word
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # apply gaussian blur for smoothening
    blurred_mask = cv2.GaussianBlur(dilated, (3, 3), 0)

    alpha = (blurred_mask.astype(np.float32) / 255)[:, :, None]

    # bolden. use color of nearest text pixel (using morphological reconstruction)
    text_color = cv2.inpaint(image, (255 - thresh).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    sub_final = (image * (1 - alpha) + text_color * alpha).astype(np.uint8)

    # bolden. do not preserve color. magic number present, test out values
    # sub_final = (image * (1 - ALPHA_MAGIC_NUMBER * alpha)).astype(np.uint8)
    return sub_final


def get_text_height(
        data: Any,
        sample_limit: int = MAX_WORD_SAMPLES,
        mean: bool = True
) -> floating[Any] | None:
    """
    Calculates the mean/median height of text in regional text

    :param data: extracted data within roi using pytesseract
    :param sample_limit: how many words to sample from the region (to keep it fast)
    :param mean: boolean indicating if function should return mean. true:false = mean:median
    :returns:
        mean/median height of the text in data["text"]
    """
    heights = []
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        h = data["height"][i]
        heights.append(h)
        if len(heights) >= sample_limit:
            break

    if not heights:
        return None  # no text found
    if mean:
        return np.mean(heights)
    return np.median(heights)


# Note: current implementation cannot handle a page with both horizontal and vertical texts
# Note: anti-pattern. blob.
# To-do
def bolden_roi(
        image: ndarray[tuple[typing.Any, ...], np.dtype],
        roi_data: dict
) -> None:
    """
    Boldens all the words within one roi
    :param image: image of the container page
    :param roi_data: data belonging to this specific roi i.e. pytesseract's image_to_data for current roi
    """
    # get the mean height of available text within the region, or None if no text
    mean_height = get_text_height(data=roi_data, sample_limit=20, mean=False)

    # if mean height is None, i.e. no text in roi, return
    if mean_height is None:
        return

    # iterate through the words
    for i, word in enumerate(roi_data["text"]):
        word = word.strip()
        if not word:
            continue

        # get word width, height, ...
        w = roi_data["width"][i]
        h = roi_data["height"][i]
        x = roi_data["left"][i]
        y = roi_data["top"][i]

        # if text too small, skip boldening. 6pt text in 300dpi ~ 24 pixels
        if h < 5:
            continue

        # map word coordinates to full (uncropped) image
        abs_x0 = x
        abs_y0 = y
        abs_x1 = x + w
        abs_y1 = y + h

        # first-(ratio) region of the word
        ratio = BOLD_RATIO

        # edge where boldening ends
        mid_x = int(abs_x0 + ratio * (abs_x1 - abs_x0))

        # in case ratio is negative. lol.
        if mid_x <= abs_x0:
            continue

        # Crop the region from full image
        word_left_crop = image[abs_y0:abs_y1, abs_x0:mid_x]
        sub_final = bolden_crop_region(image=word_left_crop, text_height=mean_height)

        # show_image(word_left_crop, title="original left crop")

        # replace the unedited pixels in original image
        image[abs_y0:abs_y1, abs_x0:mid_x] = sub_final

        # To-do lighten out the right portion of the word
        opacity = LIGHTENING_OPACITY

        word_right_crop = image[abs_y0:abs_y1, mid_x:abs_x1]

        # Blend with white
        lightened = cv2.addWeighted(
            word_right_crop, opacity,  # original image weight
            255 * np.ones_like(word_right_crop, dtype=np.uint8), 1 - opacity,  # white overlay
            0
        )

        # Replace region
        image[abs_y0:abs_y1, mid_x:abs_x1] = lightened


# Adding function to decrease pytesseract's image_to_data overhead
def get_roi_data(
        data: dict,
        roi: tuple[int, int, int, int]
) -> dict:
    """
    Filters pytesseract's image_to_data dictionary and returns dictionary specific to one roi
    :param data: pytesseract's image_to_data dictionary for the entire image
    :param roi: bounding box of the roi
    :returns:
        data dictionary specific to the input roi
    """
    x1, y1, x2, y2 = roi
    indices = []
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        x, y, w, h = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        if x >= x1 and x + w <= x2 and y >= y1 and y + h <= y2:
            indices.append(i)
    return {k: [v[i] for i in indices] for k, v in data.items()}


def get_roi_marked_image(
        og_page_img: ndarray[tuple[typing.Any, ...], np.dtype],
        rois: list[tuple[int, int, int, int]]
) -> ndarray[tuple[typing.Any, ...], np.dtype]:
    """
    Returns a copy of the og_page_img with rois marked in
    :param og_page_img: the image to be marked
    :param rois: a list of the four bounding co-ordinates of rois
    :returns:
        a copy of the input image with rois marked in
    """
    image_copy = og_page_img.copy()
    for roi in rois:
        (x0, y0, x1, y1) = roi
        image_copy = cv2.rectangle(image_copy, (x0, y0), (x1, y1), color=(255, 0, 255), thickness=3)
    return image_copy


# Source: https://gist.github.com/akash-ch2812/d42acf86e4d6562819cf4cd37d1195e7
# Edited before use
def get_rois(
        image: ndarray[tuple[typing.Any, ...], np.dtype]
) -> list[tuple[int, int, int, int]]:
    """
    Gets the bounding co-ordinates of sections within an image
    :param image: numpy array representation of the image
    :returns:
        rois: a list of the four bounding co-ordinates of rois
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    rois = []

    for c in contours:
        x0, y0, w, h = cv2.boundingRect(c)

        if w < THRESHOLD_REGION_IGNORE or h < THRESHOLD_REGION_IGNORE:
            continue

        # make sure region is within image bounds
        x1 = x0 + w
        y1 = y0 + h
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)

        roi_bbox = (x0, y0, x1, y1)

        rois.append(roi_bbox)

    return rois


def rotate_image(
        image: ndarray[tuple[typing.Any, ...], np.dtype]
) -> ndarray[tuple[typing.Any, ...], np.dtype]:
    """
    Correct image's orientation issues
    :param image: original page image
    :returns:
        og_page_img: rotated (horizontal text) page image
    """
    image = Image.fromarray(image)
    osd = pytesseract.image_to_osd(image, output_type='dict')
    rotate = osd['rotate']

    if rotate > 0:
        image = image.rotate(360 - rotate, expand=True)

    # convert from RGB to BGR
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image


def bolden_image(
        image: ndarray[tuple[typing.Any, ...], np.dtype]
) -> ndarray[tuple[typing.Any, ...], np.dtype]:
    """Boldens the first-(BOLD_RATIO) of each word encountered in png
    Works properly on horizontally aligned text for now
    :param image: image to be processed
    :returns:
        processed image
    """
    if CHECK_ORIENTATION:
        image = rotate_image(image=image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    rois = get_rois(image=image)

    # Display image with marked regions to test
    # segmented_image = get_roi_marked_image(og_page_img, rois)
    # show_image(segmented_image)

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=LANGUAGE)

    for roi in rois:
        roi_data = get_roi_data(data=data, roi=roi)
        bolden_roi(image=image, roi_data=roi_data)

    # Display image with bold regions to test
    # show_image(og_page_img)

    # To-do: handle colored bolding
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def bolden_doc(
        pdf_path: str
) -> None:
    """
    Boldens the first-(BOLD_RATIO) of each word encountered in pdf
    Works properly on horizontally aligned text for now
    """
    doc = pymupdf.open(pdf_path)
    images = []

    for page_index in range(len(doc)):
        page = doc[page_index]

        # get RGB pixmap
        pix = page.get_pixmap(dpi=300)
        og_page_img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)

        og_page_img = bolden_image(image=og_page_img)

        # save boldened image to list
        images.append(Image.fromarray(og_page_img))

    # Save all images to one continuous PDF
    if images:
        filepath = get_path(filename="boldened.pdf", input_mode=False)

        images[0].save(
            filepath,
            save_all=True,
            append_images=images[1:],
            resolution=300.0,
        )


if __name__ == "__main__":
    pdf_path = get_path(folder="sample_pdf", filename="geneve_1564.pdf", input_mode=True)
    bolden_doc(pdf_path=pdf_path)
