import os
os.environ["TORCH_HOME"] = "./clip"
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.append('./models/backbone')
import models.backbone.open_clip as open_clip
import math
from tqdm import tqdm
import time
import cv2
import json
from pathlib import Path

from datasets import rayan_dataset
from models.backbone import _backbones


class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class Preprocessing(torch.nn.Module):
    def __init__(self, input_layers, output_dim):
        super(Preprocessing, self).__init__()
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_layer in input_layers:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class LNAMD(torch.nn.Module):
    def __init__(self, device, feature_dim=1024, feature_layer=[1,2,3,4], r=3, patchstride=1):
        super(LNAMD, self).__init__()
        self.device = device
        self.r = r
        self.patch_maker = PatchMaker(r, stride=patchstride)
        self.LNA = Preprocessing(feature_layer, feature_dim)

    def _embed(self, features):
        B = features[0].shape[0]

        features_layers = []
        for feature in features:
            # reshape and layer normalization
            feature = feature[:, 1:, :] # remove the cls token
            feature = feature.reshape(feature.shape[0],
                                      int(math.sqrt(feature.shape[1])),
                                      int(math.sqrt(feature.shape[1])),
                                      feature.shape[2])
            feature = feature.permute(0, 3, 1, 2)
            feature = torch.nn.LayerNorm([feature.shape[1], feature.shape[2],
                                          feature.shape[3]]).to(self.device)(feature)
            features_layers.append(feature)

        if self.r != 1:
            # divide into patches
            features_layers = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features_layers]
            patch_shapes = [x[1] for x in features_layers]
            features_layers = [x[0] for x in features_layers]
        else:
            patch_shapes = [f.shape[-2:] for f in features_layers]
            features_layers = [f.reshape(f.shape[0], f.shape[1], -1, 1, 1).permute(0, 2, 1, 3, 4) for f in features_layers]

        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features_layers)):
            patch_dims = patch_shapes[i]
            if patch_dims[0] == ref_num_patches[0] and patch_dims[1] == ref_num_patches[1]:
                continue
            _features = features_layers[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features_layers[i] = _features
        features_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_layers]
        
        # aggregation
        features_layers = self.LNA(features_layers)
        features_layers = features_layers.reshape(B, -1, *features_layers.shape[-2:])   # (B, L, layer, C)

        return features_layers.detach().cpu()

def compute_scores_fast(Z, i, device, topmin_min=0, topmin_max=0.3):
    # speed fast but space large
    # compute anomaly scores
    image_num, patch_num, c = Z.shape
    patch2image = torch.tensor([]).to(device)
    Z_ref = torch.cat((Z[:i], Z[i+1:]), dim=0)
    patch2image = torch.cdist(Z[i:i+1], Z_ref.reshape(-1, c)).reshape(patch_num, image_num-1, patch_num)
    patch2image = torch.min(patch2image, -1)[0]
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)

def compute_scores_slow(Z, i, device, topmin_min=0, topmin_max=0.3):
    # space small but speed slow
    # compute anomaly scores
    patch2image = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            patch2image = torch.cat((patch2image, torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1)), dim=1)
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)

def MSM(Z, device, topmin_min=0, topmin_max=0.3):
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    for i in tqdm(range(Z.shape[0])):
    # for i in range(Z.shape[0]):
        anomaly_scores_i = compute_scores_fast(Z, i, device, topmin_min, topmin_max).unsqueeze(0)
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)    # (N, B)
    return anomaly_scores_matrix

def MMO(W, score, k_list=[1, 2, 3]):
    S_list = []
    for k in k_list:
        _, topk_matrix = torch.topk(W.float(), W.shape[0]-k, largest=False, sorted=True)
        W_mask = W.clone()
        for i in range(W.shape[0]):
            W_mask[i, topk_matrix[i]] = 0
        n = W.shape[-1]
        D_ = torch.zeros_like(W).float()
        for i in range(n):
            D_[i, i] = 1 / (W_mask[i,:].sum())
        P = D_ @ W_mask
        S = score.clone().unsqueeze(-1)
        S = P @ S
        S_list.append(S)
    S = torch.concat(S_list, -1).mean(-1)
    return S

def RsCIN(scores_old, cls_tokens=None, k_list=[0]):
    if cls_tokens is None or 0 in k_list:
        return scores_old
    cls_tokens = np.array(cls_tokens)
    scores = (scores_old - scores_old.min()) / (scores_old.max() - scores_old.min())
    similarity_matrix = cls_tokens @ cls_tokens.T
    similarity_matrix = torch.tensor(similarity_matrix)
    scores_new = MMO(similarity_matrix.clone().float(), score=torch.tensor(scores).clone().float(), k_list=k_list)
    scores_new = scores_new.numpy()
    return scores_new





def find_filter3(image, threshold=90):
    
    h, w,_ = image.shape
    
    
    n_patch = 4
    patch_size = h // 8
    patch_colors = []
    for i in range(n_patch):
        for j in range(n_patch):
            patch = image[patch_size * (2+i): patch_size * (3+i)][:, patch_size * (2+j): patch_size * (3+j)]
            patch_colors.append(patch.mean(0).mean(0))
            

    
    masks = []
    
    for color_vector in patch_colors:
        color_distance = np.linalg.norm(image - color_vector, axis=-1)
        
        masks.append((color_distance > threshold).astype(np.uint8))
    
    final_mask0 = np.sum(masks, 0)
    
    final_mask0 = (final_mask0 > 0.9*len(patch_colors)).astype(np.uint8)
    
    circle = 1 - cv2.circle(np.zeros_like(final_mask0), (h//2, w//2), h//9*4, 1, -1) 
    
    final_mask = circle * final_mask0
    
    return final_mask


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return {
                "__ndarray__": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        else:
            return super(NumpyEncoder, self).default(obj)


def dict_to_json(dct, filename):
    """Save a dictionary to a JSON file"""
    with open(filename, "w") as f:
        json.dump(dct, f, cls=NumpyEncoder)

class DumpScores:
    def __init__(self, scores_dir):
        self.scores_dir = scores_dir
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



class Main():
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.seed = seed
        self.device = torch.device("cuda:{}".format(cfg['device']) if torch.cuda.is_available() else "cpu")

        self.path = cfg['datasets']['data_path']
        self.categories = [_ for _ in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, _))]

        self.model_name = cfg['models']['backbone_name']
        self.image_size = cfg['datasets']['img_resize']
        self.batch_size = cfg['models']['batch_size']
        self.pretrained = cfg['models']['pretrained']
        self.features_list = [l+1 for l in cfg['models']['feature_layers']]
        self.divide_num = cfg['datasets']['divide_num']
        self.r_list = cfg['models']['r_list']
        self.load_backbone()

        if 'dino' in self.model_name:
            self.dump_scores = DumpScores("output_scores2")
        else:
            self.dump_scores = DumpScores("output_scores")


    def load_backbone(self):
        if 'dino' in self.model_name:
            # dino or dino_v2
            self.dino_model = _backbones.load(self.model_name)
            self.dino_model.to(self.device)
            print(self.dino_model)
            self.preprocess = None
        else:
            # clip
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, self.image_size, pretrained=self.pretrained, cache_dir='./clip')
            self.clip_model.to(self.device)
            print(self.clip_model)


    def load_datasets(self, category, divide_num=1, divide_iter=0):
        # dataloader
        test_dataset = rayan_dataset.RayanDataset(source=self.path, classname=category, input_size=self.image_size, output_size=224, split=rayan_dataset.DatasetSplit.TEST, external_transform=None)
        return test_dataset


    def make_category_data(self, category):
        print(category)

        # divide sub-datasets
        divide_num = self.divide_num
        anomaly_maps = torch.tensor([]).double()
        class_tokens = []
        image_path_list = []
        start_time_all = time.time()
        dataset_num = 0
        for divide_iter in range(divide_num):
            test_dataset = self.load_datasets(category, divide_num=divide_num, divide_iter=divide_iter)
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            
            # extract features
            patch_tokens_list = []
            subset_num = len(test_dataset)
            dataset_num += subset_num
            start_time = time.time()
            for image_info in tqdm(test_dataloader):
                if isinstance(image_info, dict):
                    image = image_info["image"]
                    image_path_list.extend(image_info["image_path"])
                with torch.no_grad(), torch.cuda.amp.autocast():
                    input_image = image.to(torch.float).to(self.device)
                    if 'dinov2' in self.model_name:
                        patch_tokens = self.dino_model.get_intermediate_layers(x=input_image, n=[l-1 for l in self.features_list], return_class_token=False)
                        image_features = self.dino_model(input_image)
                        patch_tokens = [patch_tokens[l].cpu() for l in range(len(self.features_list))]
                        fake_cls = [torch.zeros_like(p)[:, 0:1, :] for p in patch_tokens]
                        patch_tokens = [torch.cat([fake_cls[i], patch_tokens[i]], dim=1) for i in range(len(patch_tokens))]
                    elif 'dino' in self.model_name:
                        patch_tokens_all = self.dino_model.get_intermediate_layers(x=input_image, n=max(self.features_list))
                        image_features = self.dino_model(input_image)
                        patch_tokens = [patch_tokens_all[l-1].cpu() for l in self.features_list]
                    else: # clip
                        image_features, patch_tokens = self.clip_model.encode_image(input_image, self.features_list)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        patch_tokens = [patch_tokens[l].cpu() for l in range(len(self.features_list))]
                image_features = [image_features[bi].squeeze().cpu().numpy() for bi in range(image_features.shape[0])]
                class_tokens.extend(image_features)
                patch_tokens_list.append(patch_tokens)  # (B, L+1, C)
            end_time = time.time()
            print('extract time: {}ms per image'.format((end_time-start_time)*1000/subset_num))
            
            # LNAMD
            feature_dim = patch_tokens_list[0][0].shape[-1]
            anomaly_maps_r = torch.tensor([]).double()
            for r in self.r_list:
                start_time = time.time()
                print('aggregation degree: {}'.format(r))
                LNAMD_r = LNAMD(device=self.device, r=r, feature_dim=feature_dim, feature_layer=self.features_list)
                Z_layers = {}
                for im in range(len(patch_tokens_list)):
                    patch_tokens = [p.to(self.device) for p in patch_tokens_list[im]]
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        features = LNAMD_r._embed(patch_tokens)
                        features /= features.norm(dim=-1, keepdim=True)
                        for l in range(len(self.features_list)):
                            # save the aggregated features
                            if str(l) not in Z_layers.keys():
                                Z_layers[str(l)] = []
                            Z_layers[str(l)].append(features[:, :, l, :])
                end_time = time.time()
                print('LNAMD-{}: {}ms per image'.format(r, (end_time-start_time)*1000/subset_num))

                # MSM
                anomaly_maps_l = torch.tensor([]).double()
                start_time = time.time()
                for l in Z_layers.keys():
                    # different layers
                    Z = torch.cat(Z_layers[l], dim=0).to(self.device) # (N, L, C)
                    print('layer-{} mutual scoring...'.format(l))
                    anomaly_maps_msm = MSM(Z=Z, device=self.device, topmin_min=0, topmin_max=0.3)
                    anomaly_maps_l = torch.cat((anomaly_maps_l, anomaly_maps_msm.unsqueeze(0).cpu()), dim=0)
                    torch.cuda.empty_cache()
                anomaly_maps_l = torch.mean(anomaly_maps_l, 0)
                anomaly_maps_r = torch.cat((anomaly_maps_r, anomaly_maps_l.unsqueeze(0)), dim=0)
                end_time = time.time()
                print('MSM: {}ms per image'.format((end_time-start_time)*1000/subset_num))
            anomaly_maps_iter = torch.mean(anomaly_maps_r, 0).to(self.device)
            del anomaly_maps_r
            torch.cuda.empty_cache()

            # interpolate
            B, L = anomaly_maps_iter.shape
            H = int(np.sqrt(L))
            ac_score = anomaly_maps_iter.detach().cpu().numpy().max(-1)
            top_2 = torch.topk(anomaly_maps_iter, k=3)[0]
            max_1 = top_2[:, 0]
            max_2 = top_2[:, 1]
            ac_score = ((max_1 * 3 + max_2) / 4).detach().cpu().numpy()
            anomaly_maps_iter = F.interpolate(anomaly_maps_iter.view(B, 1, H, H),
                                        size=224, mode='bilinear', align_corners=True)
            anomaly_maps = torch.cat((anomaly_maps, anomaly_maps_iter.cpu()), dim=0)

        # save image features for optimizing classification
        # cls_save_path = os.path.join('./image_features/{}_{}.dat'.format(dataset, category))
        # with open(cls_save_path, 'wb') as f:
        #     pickle.dump([np.array(class_tokens)], f)
        end_time_all = time.time()
        print('Final: {}ms per image'.format((end_time_all-start_time_all)*1000/dataset_num))

        anomaly_maps = anomaly_maps.cpu().numpy()
        torch.cuda.empty_cache()

        B = anomaly_maps.shape[0]   # the number of unlabeled test images
        # ac_score = np.array(anomaly_maps).reshape(B, -1).max(-1)
        # RsCIN
        k_score = [1, 8, 9]
            
        scores_cls = RsCIN(ac_score, class_tokens, k_list=k_score)

    
        masks = []
        for path in image_path_list:
            image = cv2.imread(path)
            temp = find_filter3(image)
            temp = F.interpolate(torch.tensor(temp).unsqueeze(0).unsqueeze(0),
                            size=224, mode='bilinear', align_corners=True)
            masks.append(1 - temp.detach().cpu().numpy())

        anomaly_maps = np.where(np.concatenate(masks), anomaly_maps, anomaly_maps.min())

        return image_path_list, list(scores_cls), list(anomaly_maps)


    def main(self):
        image_path_list = []
        pred_img_level = []
        pred_pix_level = []
        for category in self.categories:
            a, b, c = self.make_category_data(category=category,)
            image_path_list += a
            pred_img_level += b
            pred_pix_level += c
        # pred_pix_level = np.expand_dims(np.concatenate(pred_pix_level), axis=1)
        # pred_pix_level = (pred_pix_level - pred_pix_level.min()) / (pred_pix_level.max() - pred_pix_level.min())
        # pred_pix_level = list(pred_pix_level)

        # pred_img_level = np.array(pred_img_level)
        # pred_img_level = (pred_img_level - pred_img_level.min()) / (pred_img_level.max() - pred_img_level.min())
        # pred_img_level = list(pred_img_level)

        self.dump_scores.save_scores(image_path_list, pred_img_level, pred_pix_level)
