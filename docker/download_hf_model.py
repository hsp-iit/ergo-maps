# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

from timm.models._registry import model_entrypoint
from timm.layers import set_layer_config

def main() -> None:
    kwargs = {}
    model_name = "vit_large_patch16_384.augreg_in21k_ft_in1k"
    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=None, exportable=None, no_jit=None):
        model = create_fn(
            pretrained=True,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            **kwargs,
        )
    print("Model Downloaded!")


if __name__ == "__main__":
    main()
