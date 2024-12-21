import argparse
import os
import sys
sys.path.append(os.getcwd())
from models.deploy import Main
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description='MuSc')
    parser.add_argument('--config', type=str, default='configs/musc.yaml', help='config file path')
    parser.add_argument('--data_path', type=str, default=None, help='dataset path')
    parser.add_argument('--device', type=int, default=None, help='gpu id')
    parser.add_argument('--r_list', type=int, nargs="+", default=None, help='aggregation degrees of LNAMD')
    parser.add_argument('--feature_layers', type=int, nargs="+", default=None, help='feature layers')
    parser.add_argument('--backbone_name', type=str, default=None, help='backbone')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained datasets')
    parser.add_argument('--img_resize', type=int, default=None, help='image size')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--divide_num', type=int, default=None, help='the number of divided subsets')
    args = parser.parse_args()
    return args

def load_args(cfg, args):
    if args.data_path is not None:
        cfg['datasets']['data_path'] = args.data_path
    assert os.path.exists(cfg['datasets']['data_path']), f"The dataset path {cfg['datasets']['data_path']} does not exist."
    if args.device is not None:
        cfg['device'] = args.device
    if isinstance(cfg['device'], int):
        cfg['device'] = str(cfg['device'])
    if args.r_list is not None:
        cfg['models']['r_list'] = args.r_list
    if isinstance(cfg['models']['r_list'], int):
        cfg['models']['r_list'] = [cfg['models']['r_list']]
    if args.feature_layers is not None:
        cfg['models']['feature_layers'] = args.feature_layers
    if isinstance(cfg['models']['feature_layers'], int):
        cfg['models']['feature_layers'] = [cfg['models']['feature_layers']]
    if args.backbone_name is not None:
        cfg['models']['backbone_name'] = args.backbone_name
    if args.pretrained is not None:
        cfg['models']['pretrained'] = args.pretrained
    if args.img_resize is not None:
        cfg['datasets']['img_resize'] = args.img_resize
    if args.batch_size is not None:
        cfg['models']['batch_size'] = args.batch_size
    if args.divide_num is not None:
        cfg['datasets']['divide_num'] = args.divide_num
    return cfg

if __name__ == "__main__":
    args = get_args()
    cfg = {
        'datasets': {
            'img_resize': 518,
            'data_path': 'data',
            'divide_num': 1
        },
        'models': {
            'r_list': [1, 3],
            'feature_layers': [5, 11, 17, 23],
            'backbone_name': '',
            'pretrained': 'openai',
            'batch_size': 4
        },
        'device': 0
    }
    cfg = load_args(cfg, args)
    print(cfg)
    seed = 42
    model = Main(cfg, seed=seed)
    model.main()