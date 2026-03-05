# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Leonardo Gori

import argparse
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import torch
from visual_language_navigation.fcclip.fcclip import add_maskformer2_config, add_fcclip_config, FCCLIP
import numpy as np
import time


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="fcclip demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def _init_fcclip():
    args = get_parser()
    
    args.opts = ["MODEL.WEIGHTS", "visual_language_navigation/fcclip/fcclip_cocopan.pth", "MODEL.DEVICE", "cuda"]
    args.config_file = "/home/user1/ergo-maps/visual_language_navigation/fcclip/configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml"

    cfg = setup_cfg(args)

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    # input_format = cfg.INPUT.FORMAT
    # assert input_format in ["RGB", "BGR"], input_format

    return model, aug


def get_fcclip_feat(
    model: FCCLIP,
    original_image: np.array,
    labels,
    aug,
    device,
    get_preds=False
):
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.

        # if input_format == "RGB":
        #     # whether the model expects BGR inputs or RGB
        #     original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to(device)

        inputs = {"image": image, "height": height, "width": width}

        # start = time.time()
        double_feats, index_image = model.get_image_embeddings([inputs])[0]
        #print(f"feature extraction executed in {time.time() - start}")

        if get_preds:
            category_list = []

            for label in labels:
                category_list.append([label])

            with torch.no_grad():
                # start = time.time()
                max_indices = model.query_image_segmentation(category_list, double_feats.unsqueeze(0), index_image)
                # max_indices = model.get_segmentation(category_list, featured_image)
                #print(f"indexing executed in {time.time() - start}")

        return double_feats, index_image, max_indices