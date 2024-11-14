from tqdm import tqdm
import os
from pathlib import Path
import numpy as np

from evaluation.utils.json_helpers import dict_to_json


# -----------------------------------------------------------------------------
#  This class is a sample code provided to help you with saving your predicted
#  scores as JSON files. We strongly suggest that you use the provided methods,
#  but you are NOT required to follow this structure. Feel free to adapt,
#  modify, or extend this template in favor of your own workflow.
# -----------------------------------------------------------------------------
class DumpScores:
    def __init__(self):
        self.scores_dir = "./output_scores"
        self.save_scores_precision = 4

    def save_scores(self, image_path_list, pred_img_level, pred_pix_level):
        print(
            f"Saving scores at '{self.scores_dir}' with precision: '{self.save_scores_precision}'"
        )
        for i in tqdm(range(len(image_path_list)), desc=f"Saving scores"):
            image_path = image_path_list[i]
            image_score_path = self.get_scores_path_for_image(image_path)
            os.makedirs(os.path.dirname(image_score_path), exist_ok=True)

            vectorized_enforce_precision = np.vectorize(self.enforce_precision)
            d = {
                "img_level_score": vectorized_enforce_precision(
                    pred_img_level[i], self.save_scores_precision
                ),
                "pix_level_score": vectorized_enforce_precision(
                    pred_pix_level[i], self.save_scores_precision
                ),
            }
            dict_to_json(d, image_score_path)

    def get_scores_path_for_image(self, image_path):
        """example image_path: './data/photovoltaic_module/test/good/037.png'"""
        path = Path(image_path)

        category, split, anomaly_type = path.parts[-4:-1]
        image_name = path.stem

        return os.path.join(
            self.scores_dir, category, split, anomaly_type, f"{image_name}_scores.json"
        )

    def enforce_precision(self, x, precision):
        return float(f"{x:.{precision}f}")
