from glob import glob
import json
import numpy as np


for path1, path2 in zip(glob("./output_scores/**/*.json", recursive=True), glob("./output_scores2/**/*.json", recursive=True)):
    with open(path2) as f:
        temp = json.load(f)
        # img2 = temp['img_level_score']['__ndarray__']
        pix2 = temp['pix_level_score']['__ndarray__']
    with open(path1) as f:
        temp = json.load(f)
        img1 = temp['img_level_score']['__ndarray__']
        # pix1 = np.array(temp['pix_level_score']['__ndarray__'])
    temp['img_level_score']['__ndarray__'] = img1
    temp['pix_level_score']['__ndarray__'] = pix2
    with open(path1, 'w') as f:
        json.dump(temp, f)