# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import sys
import os
import gdown
import hashlib


def sha256sum(filename: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    url = "https://drive.google.com/uc?id=1-91PIns86vyNaL3CzMmDD39zKGnPMtvj"
    expected_sha256 = "241c141e0fec7f3be7e06915daee60807502de3b7ae8b8cbef2560d6df346257"

    # Handle arguments
    dest_folder = sys.argv[1] if len(sys.argv) > 1 else ""
    if not dest_folder:
        dest_folder = os.path.expanduser(
            "~/visual-language-navigation/visual_language_navigation/fcclip"
        )

    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, "fcclip_cocopan.pth")

    # Check if file exists and validate checksum
    if os.path.isfile(dest_path):
        print(f"Found existing file at {dest_path}, verifying checksum...")
        if sha256sum(dest_path) == expected_sha256:
            print("Checksum OK, skipping download.")
            return
        else:
            print("Checksum mismatch, re-downloading...")

    # Download if missing or corrupted
    print(f"Downloading model to {dest_path}...")
    gdown.download(url, output=dest_path, quiet=False)

    # Verify again after download
    if sha256sum(dest_path) == expected_sha256:
        print("Download completed and checksum verified")
    else:
        print("Download completed but checksum FAILED")
        print("Please retry or check the source file.")


if __name__ == "__main__":
    main()