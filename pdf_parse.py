# RANDOM SCRIPT TO PARSE PDF INTO SEPARATE FILES
# ASSIGN FILE NAME USING A TEXT PARSED FROM THE PDF

import numpy as np
import argparse
import pypdfium2 as pdfium
import cv2 
import pytesseract
import re
import skimage.filters as filters

pytesseract.pytesseract.tesseract_cmd = '...tesseract.exe'

def parse_args():
    parser = argparse.ArgumentParser(description='pdf parser scan')
    parser.add_argument('--root', type=str, help='root directory', default='...')
    parser.add_argument('--pdf', type=str, help='original pdf file', default='...pdf')
    args, _ = parser.parse_known_args()
    return args


def render_pdf(file, prefix='output_', pad=5, scale=2, bbox=(245,245,160,45)):
    for image, suffix in pdfium.render_pdf_topil(file, scale=scale):
        file = f'scanned/{prefix}{suffix}.png'
        image.save(file)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # create bounding box
        x, y, w, h = bbox
        crop = img[y:y + h, x:x + w]
        
        # fine tune bounding box
        thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        thresh = 255 - thresh
        kernel = np.ones((3, 225), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            
        crop = crop[y-pad:y + h + pad, x:x + w + pad]
        crop = cv2.resize(crop, None, fx=2, fy=2)

        # preprocess
        blurred_dilation = cv2.GaussianBlur(crop, (61, 61), 0)
        division = cv2.divide(crop, blurred_dilation, scale=255)
        sharp = filters.unsharp_mask(division, radius=16, amount=16, multichannel=False, preserve_range=False)
        sharp = (255 * sharp).clip(0, 255).astype(np.uint8)
        thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # pytesseract
        length = 0
        while length != 9:
            for o in [1,3]:
                for p in range(5, 13):
                    custom_config = r'--psm {} --oem {} -c tessedit_char_whitelist=0123456789'.format(p, o)
                    text = pytesseract.image_to_string(thresh, config=custom_config)
                    text = re.sub("[^0-9]", "", text)
                    text = 'H' + text
                    if len(text) == 9:
                        break
                    print(text)
            length = len(text)
                    
        image.save(f'scanned/{prefix}{text}.pdf')

def main():
    args = parse_args()
    render_pdf(args.pdf, prefix='scan_')
