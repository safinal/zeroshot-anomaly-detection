import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from tqdm import tqdm

import random
import numpy as np

import torchvision.transforms as transforms
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from AnomalyCLIP_lib.transform import image_transform
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD



def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def get_transform(args):
    preprocess = image_transform(args.image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    return preprocess, target_transform


import os
from PIL import Image
import json

class RayanSolver(object):
    def __init__(self, root):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        
        for cls_name in [_ for _ in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, _))]:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples = anomaly_samples + 1
                            else:
                                normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)
runner = RayanSolver(root='./data')
runner.run()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, target_transform, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)


        self.obj_list = [_ for _ in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, _))]

        self.class_name_map_class_id = {}
        for k, index in zip(self.obj_list, range(len(self.obj_list))):
            self.class_name_map_class_id[k] = index

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}  


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from utils.dump_scores import DumpScores

dump_scores = DumpScores()


def test(args):
    features_list = args.features_list
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list


    results = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['img_path'] = []
        results[obj]['anomaly_maps'] = []

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path, weights_only=True)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)


    model.to(device)
    for batch_idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['img_path'].append(items['img_path'][0])
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            
            anomaly_map = anomaly_map.sum(dim = 0)
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)


    image_path_list = []
    pred_img_level = []
    pred_pix_level = []
    for val in results.values():
        image_path_list += val['img_path']
        pred_img_level += list(map(float, val['pr_sp']))
        pred_pix_level += list(map(lambda x: x.numpy(), val['anomaly_maps']))
    dump_scores.save_scores(image_path_list, pred_img_level, pred_pix_level)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/", help="path to test dataset")
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/9_12_4_multiscale/epoch_15.pth', help='path to checkpoint')
    # model
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
