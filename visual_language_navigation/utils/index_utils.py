# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Leonardo Gori

import numpy as np
from visual_language_navigation.utils.clip_utils import get_text_feats, multiple_templates


def get_lseg_score(
    clip_model,
    landmarks: list,
    lseg_map: np.array,
    clip_feat_dim: int,
    use_multiple_templates: bool = False,
    avg_mode: int = 0,
    add_other=True,
):
    """
    Inputs:
        landmarks: a list of strings that describe the landmarks
        lseg_map: a numpy array with shape (h, w, clip_dim)
        avg_mode: this is for multiple template. 0 for averaging features, 1 for averaging scores
    """
    landmarks_other = landmarks
    if add_other and landmarks_other[-1] != "other":
        landmarks_other = landmarks + ["other"]

    if use_multiple_templates:
        mul_tmp = multiple_templates.copy()
        multi_temp_landmarks_other = [x.format(lm) for lm in landmarks_other for x in mul_tmp]
        text_feats = get_text_feats(multi_temp_landmarks_other, clip_model, clip_feat_dim)

        # average the features
        if avg_mode == 0:
            text_feats = text_feats.reshape((-1, len(mul_tmp), text_feats.shape[-1]))
            text_feats = np.mean(text_feats, axis=1)

        map_feats = lseg_map.reshape((-1, lseg_map.shape[-1]))

        scores_list = map_feats @ text_feats.T

        # average the features
        if avg_mode == 1:
            scores_list = scores_list.reshape((-1, len(landmarks_other), len(mul_tmp)))
            scores_list = np.mean(scores_list, axis=2)
    else:
        text_feats = get_text_feats(landmarks_other, clip_model, clip_feat_dim)

        map_feats = lseg_map.reshape((-1, lseg_map.shape[-1]))

        scores_list = map_feats @ text_feats.T

    return scores_list

