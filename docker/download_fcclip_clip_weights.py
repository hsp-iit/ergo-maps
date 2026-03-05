# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

from visual_language_navigation.utils.fcclip_utils import get_parser, setup_cfg
from detectron2.utils.registry import Registry
from detectron2.modeling import build_model

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def main() -> None:
    args = get_parser()
    
    args.opts = ["MODEL.WEIGHTS", "/home/user1/ergo-maps/visual_language_navigation/fcclip/fcclip_cocopan.pth", "MODEL.DEVICE", "cpu"]
    args.config_file = "/home/user1/ergo-maps/visual_language_navigation/fcclip/configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml"
    cfg = setup_cfg(args)
    model = build_model(cfg)
    
    print("Weights cached!")


if __name__ == "__main__":
    main()
