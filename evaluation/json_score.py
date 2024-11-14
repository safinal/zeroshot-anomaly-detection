# -----------------------------------------------------------------------------
#  Do Not Alter This File!
# -----------------------------------------------------------------------------
#  The following code is part of the logic used for loading and evaluating your
#  output scores. Please DO NOT modify this section, as upon your submission,
#  the whole evaluation logic will be overwritten by the original code.
# -----------------------------------------------------------------------------

import warnings
import numpy as np
import torch
from tqdm import tqdm

from evaluation.base_eval import BaseEval
from evaluation.utils.json_helpers import json_to_dict

warnings.filterwarnings("ignore")


class JsonScoreEvaluator(BaseEval):
    """
    Evaluates anomaly detection performance based on pre-computed scores stored in JSON files.

    This class extends the BaseEval class and specializes in reading scores from JSON files,
    computing evaluation metrics, and optionally saving results to CSV or JSON format.

    Notes:
        - Score files are expected to follow the exact dataset structure.
                `{category}/{split}/{anomaly_type}/{image_name}_scores.json`
          e.g., `photovoltaic_module/test/good/037_scores.json`
        - Score files are expected to be at `self.scores_dir`.

    Example usage:
        >>> evaluator = JsonScoreEvaluator(cfg)
        >>> results = evaluator.main()
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def get_scores_for_image(self, image_path):
        image_scores_path = self.get_scores_path_for_image(image_path)
        image_scores = json_to_dict(image_scores_path)

        return image_scores

    def load_category_scores(self, category):
        cls_scores_list = []  # image level prediction
        anomaly_maps = []  # pixel level prediction
        gt_list = []  # image level ground truth
        img_masks = []  # pixel level ground truth

        image_path_list = []
        test_dataset = self.load_datasets(category)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        for image_info in tqdm(test_dataloader):
            if not isinstance(image_info, dict):
                raise ValueError("Encountered non-dict image in dataloader")

            del image_info["image"]

            image_path = image_info["image_path"][0]
            image_path_list.extend(image_path)

            img_masks.append(image_info["mask"])
            gt_list.extend(list(image_info["is_anomaly"].numpy()))

            image_scores = self.get_scores_for_image(image_path)
            cls_scores = image_scores["img_level_score"]
            anomaly_maps_iter = image_scores["pix_level_score"]

            cls_scores_list.append(cls_scores)
            anomaly_maps.append(anomaly_maps_iter)

        pr_sp = np.array(cls_scores_list)
        gt_sp = np.array(gt_list)
        pr_px = np.array(anomaly_maps)
        gt_px = torch.cat(img_masks, dim=0).numpy().astype(np.int32)

        assert pr_px.shape[1:] == (
            1,
            224,
            224,
        ), "Predicted output scores do not meet the expected shape!"
        assert gt_px.shape[1:] == (
            1,
            224,
            224,
        ), "Loaded ground truth maps do not meet the expected shape!"

        return gt_sp, pr_sp, gt_px, pr_px, image_path_list
