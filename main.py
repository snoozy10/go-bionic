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
# check and correct the orientation of pdfs before going bionic
CHECK_ORIENTATION = False
# ratio of word-width getting boldened
BOLD_RATIO = 0.5
# languages in the pdf to process by pytesseract
# https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
LANGUAGE = "deu+eng"
# magic numbers
KERNEL_MAGIC_NUMBER = 16
ALPHA_MAGIC_NUMBER = 0.85


def get_text_height(
        data: Any, sample_limit: int = MAX_WORD_SAMPLES, mean: bool = True
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


# Source: https://gist.github.com/akash-ch2812/d42acf86e4d6562819cf4cd37d1195e7
# Edited before use
def get_rois(
        image: ndarray[tuple[typing.Any, ...]]
) -> tuple[ndarray[tuple[Any, ...]], list[tuple[int, int, int, int]]]:
    """
    Gets the bounding co-ordinates of sections within an image
    :param image: numpy array representation of the image
    :returns:
        image_copy: a copy of the image with bounding boxes drawn in
        rois: a list of the four bounding co-ordinates of rois
    """

    # only tinker with a copy of the og image, and draw rectangles in a copied image
    image_copy = image.copy()

    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
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
        x1, y1 = min(image_copy.shape[1], x1), min(image_copy.shape[0], y1)

        # draw the bounding boxes
        image_copy = cv2.rectangle(image_copy, (x0, y0), (x1, y1), color=(255, 0, 255), thickness=3)
        roi_bbox = (x0, y0, x1, y1)

        rois.append(roi_bbox)

    return image_copy, rois


def resize_with_aspect_ratio(
        image: ndarray[tuple[typing.Any, ...]],
        width: int | None = None,
        height: int | None = None,
        inter: int = cv2.INTER_AREA
) -> ndarray[tuple[Any, ...]]:
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
        image: ndarray[tuple[typing.Any, ...]],
        width: int = 450,
        resize: bool = True
) -> None:
    """
    Displays an image using opencv's image viewer
    """
    if resize:
        image = resize_with_aspect_ratio(image, width=width)
    cv2.imshow("show_image", image)
    cv2.waitKey()


# Note: current implementation cannot handle a page with both horizontal and vertical texts
# To-do
def bolden_roi(
        image: ndarray[tuple[Any, ...]],
        roi_data: dict
) -> None:
    """
    Boldens all the words within one roi
    :param image: image of the container page
    :param roi_data: data belonging to this specific roi i.e. pytesseract's image_to_data for current roi
    """
    # get the mean height of available text within the region, or None if no text
    mean_height = get_text_height(roi_data, 20, False)

    # if mean height is None, i.e. no text in roi, return
    if mean_height is None:
        return

    # iterate through the words
    for i, word in enumerate(roi_data["text"]):
        word = word.strip()
        if not word:
            continue
        # print(word)

        # get word width, height, ...
        w = roi_data["width"][i]
        h = roi_data["height"][i]
        x = roi_data["left"][i]
        y = roi_data["top"][i]

        # if text too small, ignore
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

        # in case ratio is negative lol
        if mid_x <= abs_x0:
            continue

        # Crop the region from full image
        word_left_crop = image[abs_y0:abs_y1, abs_x0:mid_x]

        # if cropping yields no region, skip
        if word_left_crop.size == 0:
            continue

        # start adaptive, smooth boldening
        # -- convert to gray
        gray = cv2.cvtColor(word_left_crop, cv2.COLOR_BGR2GRAY)

        # -- threshold, inverted
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # choose kernel size by text height. use a *magic number*
        k = max(1, int(mean_height / KERNEL_MAGIC_NUMBER))

        # use a circular or disk-shaped structuring element for morphological operations
        # useful for smoothing out corners, removing small circular noise, etc
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # bolden the first-(ratio) of the word
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # apply gaussian blur for smoothening
        blurred_mask = cv2.GaussianBlur(dilated, (3, 3), 0)
        alpha = (blurred_mask.astype(np.float32) / 255)[:, :, None]

        # magic numbers present. test out values
        sub_final = (word_left_crop * (1 - ALPHA_MAGIC_NUMBER * alpha)).astype(np.uint8)

        # replace the unedited pixels in original image
        image[abs_y0:abs_y1, abs_x0:mid_x] = sub_final


def rotate_page(
        og_page_img: ndarray[tuple[typing.Any, ...]]
) -> ndarray[tuple[typing.Any, ...]]:
    """
    Correct image's orientation issues
    :param og_page_img: original page image
    :returns:
        og_page_img: rotated (horizontal text) page image
    """
    og_page_img = Image.fromarray(og_page_img)
    osd = pytesseract.image_to_osd(og_page_img, output_type='dict')
    rotate = osd['rotate']

    if rotate > 0:
        og_page_img = og_page_img.rotate(360 - rotate, expand=True)

    # convert from RGB to BGR
    og_page_img = cv2.cvtColor(np.array(og_page_img), cv2.COLOR_RGB2BGR)
    return og_page_img


def bolden_doc() -> None:
    """
    Boldens the first-(BOLD_RATIO) of each word encountered
    Works properly on horizontally aligned text for now
    """
    root = Path(__file__).resolve().parent
    pdf_path = os.path.join(root, "sample_pdf", "geneve_1564.pdf")

    doc = pymupdf.open(pdf_path)
    images = []

    for page_index in range(len(doc)):
        page = doc[page_index]

        # get RGB pixmap
        pix = page.get_pixmap(dpi=300)
        og_page_img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)

        if CHECK_ORIENTATION:
            og_page_img = rotate_page(og_page_img)
        else:
            og_page_img = cv2.cvtColor(og_page_img, cv2.COLOR_RGB2BGR)

        segmented_image, rois = get_rois(og_page_img)

        # Display image with marked regions to test
        show_image(segmented_image)
        data = pytesseract.image_to_data(og_page_img, output_type=pytesseract.Output.DICT, lang=LANGUAGE)

        for bbox in rois:
            roi_data = get_roi_data(data, bbox)
            bolden_roi(og_page_img, roi_data)

        # Display image with bold regions to test
        # show_image(og_page_img)

        # To-do: handle colored bolding
        og_page_img = cv2.cvtColor(og_page_img, cv2.COLOR_BGR2RGB)

        # save boldened image to list
        images.append(Image.fromarray(og_page_img))

    # Save all images to one continuous PDF
    if images:
        images[0].save(
            "boldened.pdf",
            save_all=True,
            append_images=images[1:],
            resolution=300.0,
        )


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


if __name__ == "__main__":
    bolden_doc()
