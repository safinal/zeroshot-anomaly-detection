#!/bin/sh
python3 main.py --device 0 --data_path ./data --backbone_name ViT-L-14-336 --pretrained openai --feature_layers 5 11 17 23 --img_resize 518 --divide_num 1 --r_list 1 3 --batch_size 4
python3 main.py --device 0 --data_path ./data --backbone_name dinov2_vitl14 --pretrained openai --feature_layers 5 11 17 23 --img_resize 518 --divide_num 1 --r_list 1 3 --batch_size 4
python3 combine.py