# RANDOM SCRIPT TO PARSE PDF INTO SEPARATE FILES
# ASSIGN FILE NAME USING A TEXT PARSED FROM THE PDF

import numpy as np
import pandas as pd

import argparse
import pypdfium2 as pdfium
import cv2 
import pytesseract
import re
import skimage.filters as filters

pytesseract.pytesseract.tesseract_cmd = '.../Tesseract-OCR/tesseract.exe'

def parse_args():
    parser = argparse.ArgumentParser(description='W9')
    parser.add_argument('--root', type=str, help='root directory', default='working_dir/')
    parser.add_argument('--pdf', type=str, help='original pdf file', default='.../example.pdf')
    args, _ = parser.parse_known_args()
    return args

args = parse_args()

def align_images(image, template, max_features=800, keep_percent=0.2, debug=False):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(max_features)
    (kps_a, descs_a) = orb.detectAndCompute(image_gray, None)
    (kps_b, descs_b) = orb.detectAndCompute(template_gray, None)

    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descs_a, descs_b, None)
    
    matches = sorted(matches, key=lambda x:x.distance)

    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    if debug:
        matched_vis = cv2.drawMatches(image, kps_a, template, kps_b, matches, None)
        matched_vis = imutils.resize(matched_vis, width=1000)
        cv2.imshow("Matched Keypoints", matched_vis)
        cv2.waitKey(0)
        
    pts_a = np.zeros((len(matches), 2), dtype="float")
    pts_b = np.zeros((len(matches), 2), dtype="float")

    for (i, m) in enumerate(matches):
        pts_a[i] = kps_a[m.queryIdx].pt
        pts_b[i] = kps_b[m.trainIdx].pt
        
    (H, mask) = cv2.findHomography(pts_a, pts_b, method=cv2.RANSAC)
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    return aligned
    
def render_pdf(file, prefix='output_'):
    
    pages = {}
    
    for image, suffix in pdfium.render_pdf_topil(file, scale=2):
        
        #--scan page
        file = f'scanned/{prefix}{suffix}.png'
        image.save(file)
        
        img = cv2.imread(file)
        template = cv2.imread("scanned/template.png")
        img = align_images(img, template, debug=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # create bounding box
        bbox=(245,245,160,45)
        x, y, w, h = bbox
        crop = img[y:y + h, x:x + w]

        thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        thresh = 255 - thresh
        kernel = np.ones((3, 225), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)

        pad = 10
        crop = crop[y-pad:y + h + pad, x:x + w + pad]
        crop = cv2.resize(crop, None, fx=2, fy=2)

        # preprocess
        blurred_dilation = cv2.GaussianBlur(crop, (61, 61), 0)
        division = cv2.divide(crop, blurred_dilation, scale=255)
        sharp = filters.unsharp_mask(division, radius=16, amount=16, channel_axis=False, preserve_range=False)
        sharp = (255 * sharp).clip(0, 255).astype(np.uint8)
        thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
        length = 0
        while length != 9:
            for o in [1,3]:
                for p in range(6, 13):
                    custom_config = r'--psm {} --oem {} -c tessedit_char_whitelist=0123456789'.format(p, o)
                    text = pytesseract.image_to_string(thresh, config=custom_config)
                    text = re.sub("[^0-9]", "", text)
                    text = 'H' + text
                    if len(text) == 9:
                        break
            length = len(text)

        pages[file] = {'nbr': text, 'image':image}

    return pages
        
pages = render_pdf(args.pdf, prefix='scan_')
res = pd.DataFrame.from_dict(pages).T

for nbr in res['nbr']:
    x = res.loc[res['nbr']==nbr].reset_index(drop=True)
    x = list(x['image'])
    if len(x) == 1:
        x[0].save(f'scanned/{nbr}.pdf')
    else:
        img = x.pop(0)
        img.save(f'scanned/{nbr}.pdf', save_all=True, append_images=x)

