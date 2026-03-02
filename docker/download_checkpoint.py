# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import sys
import gdown
import os


def main()->None:
    url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
    try:
        dest_folder = sys.argv[1]
    except Exception:
        dest_folder=""
    if dest_folder == "" or dest_folder == None:
        dest_folder = os.path.expanduser('~') + "/visual-language-navigation/visual_language_navigation/lseg/checkpoints"

    dest_path = dest_folder + "/demo_e200.ckpt"
    if not os.path.isfile(dest_path):
        gdown.download(url, output=str(dest_path))

if __name__ == "__main__":
    main()