# -----------------------------------------------------------------------------
#  Do Not Alter This File!
# -----------------------------------------------------------------------------
#  The following code is part of the logic used for loading and evaluating your
#  output scores. Please DO NOT modify this section, as upon your submission,
#  the whole evaluation logic will be overwritten by the original code.
# -----------------------------------------------------------------------------

import warnings
import os
from pathlib import Path
import csv
import json
import torch

import datasets.rayan_dataset as rayan_dataset
from evaluation.utils.metrics import compute_metrics

warnings.filterwarnings("ignore")


class BaseEval:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            "cuda:{}".format(cfg["device"]) if torch.cuda.is_available() else "cpu"
        )

        self.path = cfg["datasets"]["data_path"]
        self.dataset = cfg["datasets"]["dataset_name"]
        self.save_csv = cfg["testing"]["save_csv"]
        self.save_json = cfg["testing"]["save_json"]
        self.categories = cfg["datasets"]["class_name"]
        if isinstance(self.categories, str):
            if self.categories.lower() == "all":
                if self.dataset == "rayan_dataset":
                    self.categories = self.get_available_class_names(self.path)
            else:
                self.categories = [self.categories]
        self.output_dir = cfg["testing"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.scores_dir = cfg["testing"]["output_scores_dir"]
        self.class_name_mapping_dir = cfg["testing"]["class_name_mapping_dir"]

        self.leaderboard_metric_weights = {
            "image_auroc": 1.2,
            "image_ap": 1.1,
            "image_f1": 1.1,
            "pixel_auroc": 1.0,
            "pixel_aupro": 1.4,
            "pixel_ap": 1.3,
            "pixel_f1": 1.3,
        }

    def get_available_class_names(self, root_data_path):
        all_items = os.listdir(root_data_path)
        folder_names = [
            item
            for item in all_items
            if os.path.isdir(os.path.join(root_data_path, item))
        ]

        return folder_names

    def load_datasets(self, category):
        dataset_classes = {
            "rayan_dataset": rayan_dataset.RayanDataset,
        }

        dataset_splits = {
            "rayan_dataset": rayan_dataset.DatasetSplit.TEST,
        }

        test_dataset = dataset_classes[self.dataset](
            source=self.path,
            split=dataset_splits[self.dataset],
            classname=category,
        )
        return test_dataset

    def get_category_metrics(self, category):
        print(f"Loading scores of '{category}'")
        gt_sp, pr_sp, gt_px, pr_px, _ = self.load_category_scores(category)

        print(f"Computing metrics for '{category}'")
        image_metric, pixel_metric = compute_metrics(gt_sp, pr_sp, gt_px, pr_px)

        return image_metric, pixel_metric

    def load_category_scores(self, category):
        raise NotImplementedError()

    def get_scores_path_for_image(self, image_path):
        """example image_path: './data/photovoltaic_module/test/good/037.png'"""
        path = Path(image_path)

        category, split, anomaly_type = path.parts[-4:-1]
        image_name = path.stem

        return os.path.join(
            self.scores_dir, category, split, anomaly_type, f"{image_name}_scores.json"
        )

    def calc_leaderboard_score(self, **metrics):
        weighted_sum = 0
        total_weight = 0
        for key, weight in self.leaderboard_metric_weights.items():
            metric = metrics.get(key)
            weighted_sum += metric * weight
            total_weight += weight

        if total_weight == 0:
            return 0

        return weighted_sum / total_weight

    def main(self):
        image_auroc_list = []
        image_f1_list = []
        image_ap_list = []
        pixel_auroc_list = []
        pixel_f1_list = []
        pixel_ap_list = []
        pixel_aupro_list = []
        leaderboard_score_list = []
        for category in self.categories:
            image_metric, pixel_metric = self.get_category_metrics(
                category=category,
            )
            image_auroc, image_f1, image_ap = image_metric
            pixel_auroc, pixel_f1, pixel_ap, pixel_aupro = pixel_metric
            leaderboard_score = self.calc_leaderboard_score(
                image_auroc=image_auroc,
                image_f1=image_f1,
                image_ap=image_ap,
                pixel_auroc=pixel_auroc,
                pixel_aupro=pixel_aupro,
                pixel_f1=pixel_f1,
                pixel_ap=pixel_ap,
            )

            image_auroc_list.append(image_auroc)
            image_f1_list.append(image_f1)
            image_ap_list.append(image_ap)
            pixel_auroc_list.append(pixel_auroc)
            pixel_f1_list.append(pixel_f1)
            pixel_ap_list.append(pixel_ap)
            pixel_aupro_list.append(pixel_aupro)
            leaderboard_score_list.append(leaderboard_score)

            print(category)
            print(
                "[image level] auroc:{}, f1:{}, ap:{}".format(
                    image_auroc * 100,
                    image_f1 * 100,
                    image_ap * 100,
                )
            )
            print(
                "[pixel level] auroc:{}, f1:{}, ap:{}, aupro:{}".format(
                    pixel_auroc * 100,
                    pixel_f1 * 100,
                    pixel_ap * 100,
                    pixel_aupro * 100,
                )
            )
            print(
                "leaderboard score:{}".format(
                    leaderboard_score * 100,
                )
            )

        image_auroc_mean = sum(image_auroc_list) / len(image_auroc_list)
        image_f1_mean = sum(image_f1_list) / len(image_f1_list)
        image_ap_mean = sum(image_ap_list) / len(image_ap_list)
        pixel_auroc_mean = sum(pixel_auroc_list) / len(pixel_auroc_list)
        pixel_f1_mean = sum(pixel_f1_list) / len(pixel_f1_list)
        pixel_ap_mean = sum(pixel_ap_list) / len(pixel_ap_list)
        pixel_aupro_mean = sum(pixel_aupro_list) / len(pixel_aupro_list)
        leaderboard_score_mean = sum(leaderboard_score_list) / len(
            leaderboard_score_list
        )

        print("mean")
        print(
            "[image level] auroc:{}, f1:{}, ap:{}".format(
                image_auroc_mean * 100, image_f1_mean * 100, image_ap_mean * 100
            )
        )
        print(
            "[pixel level] auroc:{}, f1:{}, ap:{}, aupro:{}".format(
                pixel_auroc_mean * 100,
                pixel_f1_mean * 100,
                pixel_ap_mean * 100,
                pixel_aupro_mean * 100,
            )
        )
        print(
            "leaderboard score:{}".format(
                leaderboard_score_mean * 100,
            )
        )

        # Save the final results as a csv file
        if self.save_csv:
            with open(self.class_name_mapping_dir, "r") as f:
                class_name_mapping_dict = json.load(f)
            csv_data = [
                [
                    "Category",
                    "pixel_auroc",
                    "pixel_f1",
                    "pixel_ap",
                    "pixel_aupro",
                    "image_auroc",
                    "image_f1",
                    "image_ap",
                    "leaderboard_score",
                ]
            ]
            for i, category in enumerate(self.categories):
                csv_data.append(
                    [
                        class_name_mapping_dict[category],
                        pixel_auroc_list[i] * 100,
                        pixel_f1_list[i] * 100,
                        pixel_ap_list[i] * 100,
                        pixel_aupro_list[i] * 100,
                        image_auroc_list[i] * 100,
                        image_f1_list[i] * 100,
                        image_ap_list[i] * 100,
                        leaderboard_score_list[i] * 100,
                    ]
                )
            csv_data.append(
                [
                    "mean",
                    pixel_auroc_mean * 100,
                    pixel_f1_mean * 100,
                    pixel_ap_mean * 100,
                    pixel_aupro_mean * 100,
                    image_auroc_mean * 100,
                    image_f1_mean * 100,
                    image_ap_mean * 100,
                    leaderboard_score_mean * 100,
                ]
            )

            csv_file_path = os.path.join(self.output_dir, "results.csv")
            with open(csv_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(csv_data)

        # Save the final results as a json file
        if self.save_json:
            json_data = []
            with open(self.class_name_mapping_dir, "r") as f:
                class_name_mapping_dict = json.load(f)
            for i, category in enumerate(self.categories):
                json_data.append(
                    {
                        "Category": class_name_mapping_dict[category],
                        "pixel_auroc": pixel_auroc_list[i] * 100,
                        "pixel_f1": pixel_f1_list[i] * 100,
                        "pixel_ap": pixel_ap_list[i] * 100,
                        "pixel_aupro": pixel_aupro_list[i] * 100,
                        "image_auroc": image_auroc_list[i] * 100,
                        "image_f1": image_f1_list[i] * 100,
                        "image_ap": image_ap_list[i] * 100,
                        "leaderboard_score": leaderboard_score_list[i] * 100,
                    }
                )
            json_data.append(
                {
                    "Category": "mean",
                    "pixel_auroc": pixel_auroc_mean * 100,
                    "pixel_f1": pixel_f1_mean * 100,
                    "pixel_ap": pixel_ap_mean * 100,
                    "pixel_aupro": pixel_aupro_mean * 100,
                    "image_auroc": image_auroc_mean * 100,
                    "image_f1": image_f1_mean * 100,
                    "image_ap": image_ap_mean * 100,
                    "leaderboard_score": leaderboard_score_mean * 100,
                }
            )

            json_file_path = os.path.join(self.output_dir, "results.json")
            with open(json_file_path, mode="w") as file:
                final_json = {
                    "result": leaderboard_score_mean * 100,
                    "metadata": json_data,
                }
                json.dump(final_json, file, indent=4)
