import datetime
import cv2
import imutils
import imutils.contours
import numpy as np

from img2table.ocr import PaddleOCR

from dataclasses import dataclass
from typing import List

from .config import first_line_mask, second_line_mask

ocr = PaddleOCR()

translation_tables = {
    'rus': str.maketrans({
        'A': 'А', 'B': 'Б', 'V': 'В', 'G': 'Г', 'D': 'Д', 'E': 'Е', '2': 'Ё', 'J': 'Ж', 'Z': 'З', 'I': 'И', 'Q': 'Й',
        'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'R': 'Р', 'S': 'С', 'T': 'Т', 'U': 'У', 'F': 'Ф',
        'H': 'Х', 'C': 'Ц', '3': 'Ч', '4': 'Ш', 'W': 'Щ', 'X': 'Ъ', 'Y': 'Ы', '9': 'Ь', '6': 'Э', '7': 'Ю', '8': 'Я'
    })
}


@dataclass
class MRZResult:
    passport_type: str
    country_code: str
    surname: str
    names: List[str]
    number: str
    citizenship: str
    date_of_birth: datetime.date
    sex: str


def prepare(image):
    final_wide = 1200
    r = float(final_wide) / image.shape[1]
    dim = (final_wide, int(image.shape[0] * r))

    img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    area_thresh = 0
    big_contour = contours[0]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_thresh:
            area_thresh = area
            big_contour = contour

    page = np.zeros_like(img)

    cv2.drawContours(page, [big_contour], 0, (255, 255, 255), -1)

    peri = cv2.arcLength(big_contour, True)
    corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)

    polygon = img.copy()

    cv2.polylines(polygon, [corners], True, (0, 0, 255), 1, cv2.LINE_AA)

    nr = np.empty((0, 2), dtype="int32")

    yarr = []
    xarr = []
    for a in corners:
        for b in a:
            nr = np.vstack([nr, b])

    for i in nr:
        yarr.append(i[0])
        xarr.append(i[1])

    return img[min(xarr):max(xarr), min(yarr):max(yarr)]


def recognize_mrz_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray.shape

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    min_val, max_val = (np.min(grad), np.max(grad))
    grad = (grad - min_val) / (max_val - min_val)
    grad = (grad * 255).astype("uint8")
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect_kernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
    thresh = cv2.erode(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = imutils.contours.sort_contours(cnts, method="bottom-to-top")[0]

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        percent_width = w / float(image_width)
        percent_height = h / float(image_height)
        if percent_width > 0.29 and percent_height > 0.005:
            mrz_box = (x, y, w, h)
            break
    else:
        return

    x, y, w, h = mrz_box
    px = int((x + w) * 0.03)
    py = int((y + h) * 0.1)
    x, y = (x - px, y - py)
    w, h = (w + (px * 2), h + (py * 2))
    mrz = image[y:y + h, x:x + w]

    results = ocr.ocr(mrz)

    mrz_text = ''
    for line, confidence in results[1]:
        if confidence < 0.6:
            continue

        if len(line) < 20:
            continue

        mrz_text += line + '\n'

    return mrz_text.replace(' ', '').strip()


def _follow_to_sep(text: str, sep='<'):
    if sep not in text:
        return text, -1

    i = text.index(sep)

    return text[:i], i


def _parse_line(line: str, sep='<'):
    parts = []
    while sep in line:
        part, i = _follow_to_sep(line, sep=sep)
        parts.append(part)

        line = line[i + 1:]

    parts.append(line)

    return parts


def _replace_non_alpha_chars(text: str):
    return text.replace('0', 'O').replace('1', 'I').replace('5', 'S')


def _replace_non_digit_chars(text: str):
    return text.replace('O', '0').replace('I', '1').replace('S', '5')


def decode_mrz_text(mrz_text: str, lang='rus', verify_checksum: bool = False):
    lang = lang.lower()

    def translate(text: str):
        if lang not in translation_tables.keys():
            return text

        return text.translate(translation_tables[lang])

    while '\n\n' in mrz_text:
        mrz_text = mrz_text.replace('\n\n', '\n')

    lines = mrz_text.upper().split('\n')

    sorted_lines = sorted(sorted(lines, key=len, reverse=True)[:2], key=lines.index)

    first_line, second_line = sorted_lines

    first_line = first_line_mask.parse(first_line, verify_checksum=False)

    names = [
        translate(_replace_non_alpha_chars(name)).title()
        for name in [element for element in _parse_line(first_line[3]) if element.strip()][:3]
    ]
    second_line = second_line_mask.parse(second_line, verify_checksum=verify_checksum)

    number = second_line[0][:3] + second_line[8][:1] + second_line[0][3:9]

    date_of_birth = _replace_non_digit_chars(second_line[3])

    year = date_of_birth[:2]
    if int(year) > 24:
        year = '19' + year
    else:
        year = '20' + year

    date_of_birth = datetime.datetime.strptime(
        '.'.join((date_of_birth[4:], date_of_birth[2:4], year)),
        '%d.%m.%Y'
    ).date()

    return MRZResult(
        passport_type=first_line[1],
        country_code=first_line[2],
        surname=names[0],
        names=names[1:],
        number=number,
        citizenship=second_line[2],
        date_of_birth=date_of_birth,
        sex=second_line[5]
    )


def recognize(image, lang='rus', verify_checksum: bool = True):
    return decode_mrz_text(recognize_mrz_text(prepare(image)), lang=lang, verify_checksum=verify_checksum)
