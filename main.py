from __future__ import annotations

import typing
from typing import Any

import cv2  # openCV https://docs.opencv.org/4.12.0/
import pytesseract
import pymupdf

import numpy as np
from numpy import floating, ndarray

import os
from pathlib import Path
import time
from tqdm import tqdm

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

# threshold of regions to ignore
THRESHOLD_REGION_IGNORE = 40

# max number of words to consider for mean/median text height in a text-region
MAX_WORD_SAMPLES = 10

# check and correct the orientation of pdfs before going bionic. set to True for skewed layouts
CHECK_ORIENTATION = True

# ratio of word-width getting boldened
BOLD_RATIO = 0.5

# languages in the pdf to process by pytesseract
# https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
LANGUAGE = "deu"  # for the wild umlauts :)

# magic number to scale kernel size wrt text height.
# useful for large texts. higher = weaker darkening wrt height.
KERNEL_MAGIC_NUMBER = 16

# opacity of the lightened half (right half) of words
LIGHTENING_OPACITY = 0.70

# Precompute 256 look-up-tables for alpha calculation
LUT_NORM = np.arange(256, dtype=np.float32) / 255.0


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


def bolden_word_crop(
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

    # -- threshold mask, inverted
    _, thresh = cv2.threshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # choose kernel size by text height
    # set KERNEL_MAGIC_NUMBER >= 15 (based on dpi=300 and minimum height of legible text)
    k = max(1, int(text_height / KERNEL_MAGIC_NUMBER))

    # use a circular or disk-shaped structuring element for morphological operations
    # useful for smoothing out corners, removing small circular noise, etc
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # dilate the thresh mask. then apply gaussian blur for smoother edges
    blurred_mask = cv2.GaussianBlur(
        cv2.dilate(thresh, kernel, iterations=1),
        (3, 3),
        0
    )
    alpha = cv2.LUT(blurred_mask, LUT_NORM).astype(np.float32)[:, :, None]

    # bolden AND darken. use mean color of nearest text pixel
    text_color = cv2.mean(image, mask=thresh)[0:3]
    text_color = np.full_like(image, text_color)
    sub_final = (image * (1 - alpha) + text_color * alpha).astype(np.uint8)

    # bolden. do not preserve color. higher alpha_multiplier = more darkening
    # alpha_multiplier = 0.85
    # sub_final = (image * (1 - alpha_multiplier * alpha)).astype(np.uint8)
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
        if h < KERNEL_MAGIC_NUMBER:
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
        sub_final = bolden_word_crop(image=word_left_crop, text_height=mean_height)

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
def get_roi_bounds(
        image: ndarray[tuple[typing.Any, ...], np.dtype]
) -> list[tuple[int, int, int, int]]:
    """
    Gets the bounding co-ordinates of sections within an image
    :param image: numpy array representation of the image
    :returns:
        rois: a list of the four bounding co-ordinates of rois
    """
    # Next chain of operations:
    # 1. gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2. blur = cv2.GaussianBlur(gray, (9, 9), 0)
    # 3. thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    thresh = cv2.adaptiveThreshold(
        cv2.GaussianBlur(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            ),
            (9, 9),
            0
        ),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        30
    )

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # Find contours, highlight text areas
    contours = cv2.findContours(
        cv2.dilate(thresh, kernel, iterations=4),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
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
    osd = pytesseract.image_to_osd(image, output_type='dict')
    rotate = osd['rotate']

    if rotate > 0:
        height, width = image.shape[:2]

        center_x = width//2
        center_y = height//2

        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), 360-rotate, 1.0)

        cos = np.abs(rotation_matrix[0][0])
        sin = np.abs(rotation_matrix[0][1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)

        rotation_matrix[0][2] += (new_width/2 - center_x)
        rotation_matrix[1][2] += (new_height/2 - center_y)

        image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
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
    rois = get_roi_bounds(image=image)

    # Display image with marked regions to test
    # segmented_image = get_roi_marked_image(og_page_img, rois)
    # show_image(segmented_image)

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        lang=LANGUAGE,
    )

    for roi in rois:
        roi_data = get_roi_data(data=data, roi=roi)
        bolden_roi(image=image, roi_data=roi_data)

    # Display image with bold regions to test
    # show_image(og_page_img)

    # handle colored bolding
    return image


def process_page(
        pdf_path: str,
        page_index: int,
) -> tuple[int, int, bytes]:
    """
    Worker method for processing each page in a separate process
    :param pdf_path: path of the input pdf
    :param page_index: index of the page being worked with
    :returns:
        a tuple containing the width, height, and processed page image as bytes
    """
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        page_width = page.rect.width
        page_height = page.rect.height
        pix = doc.get_page_pixmap(pno=page_index, dpi=250, alpha=False)  # lower DPI for speed

    image = np.frombuffer(pix.samples, np.uint8).reshape((pix.height, pix.width, pix.n))
    if CHECK_ORIENTATION:
        image = rotate_image(image=image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = bolden_image(image)

    _, img_png_encoded = cv2.imencode(".png", image)
    result = page_width, page_height, bytes(img_png_encoded)

    return result


def bolden_doc_mp(
        input_pdf_path: str,
        output_pdf_path: str,
) -> None:
    """
    Boldens the first-(BOLD_RATIO) of each word encountered in pdf
    Works properly on horizontally aligned text for now
    """
    with pymupdf.open(input_pdf_path) as doc:
        n_pages = len(doc)

    max_workers = max(1, min(n_pages, cpu_count() - 1))

    with ProcessPoolExecutor(max_workers) as executor:
        futures = {
            executor.submit(
                process_page, input_pdf_path, i
            ): i for i in range(n_pages)
        }
        processed_pages = [None] * n_pages

        def save_using_pymupdf():
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pages", unit="page"):
                i = futures[future]
                processed_pages[i] = future.result()

            with pymupdf.open() as new_doc:
                for w, h, img_bytes in tqdm(processed_pages, total=len(processed_pages), desc="Adding pages", unit="page"):
                    new_doc.new_page(width=w, height=h).insert_image(pymupdf.Rect(0, 0, w, h), stream=img_bytes)
                tqdm.write("Saving PDF...")
                new_doc.ez_save(output_pdf_path)
                tqdm.write("Boldening complete!")

        # (pymupdf version) ^ OR v (PIL version)
        def save_using_PIL():
            tqdm.write("Saving PDF...")
            if processed_pages:
                from PIL import Image
                pil_images = [
                    Image.frombytes(
                        mode="RGB",
                        size=(int(w), int(h)),
                        data=img
                    ) for (w, h, img) in processed_pages]
                pil_images[0].save(
                    fp=save_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    resolution=300.0
                )
            tqdm.write("Boldening complete!")

        save_using_pymupdf()


def get_input_pdf_path_by_prompting() -> str | None:
    """
    A simple filedialog to prompt for the pdf to process
    :return:
        absolute filepath as string
    """
    from tkinter import filedialog as fd

    root_path = Path(__file__).resolve().parent

    filetypes = (
        ('pdf files', '*.pdf'),
    )
    filepath = fd.askopenfilename(
        title="go-bionic: Select input PDF",
        filetypes=filetypes,
        initialdir=root_path
    )
    return filepath


if __name__ == "__main__":
    pdf_path = get_input_pdf_path_by_prompting()
    if not pdf_path:
        print("Nothing to convert. Exiting application...")
        exit()

    filename_wo_extension = Path(pdf_path).stem

    output_folder = "sample_output"
    output_filename_w_extension = filename_wo_extension + "_boldened.pdf"

    save_path = get_path(
        folder=output_folder,
        filename=output_filename_w_extension,
        input_mode=False
    )

    start = time.time()
    bolden_doc_mp(
        input_pdf_path=pdf_path,
        output_pdf_path=save_path
    )
    end = time.time()
    print("Elapsed time using multiprocessing: ", end-start)




