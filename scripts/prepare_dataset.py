# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import shutil

from PIL import Image


def main(args):
    raw_image_dir = f"{args.image_dir}/GTmod12"
    hr_image_dir = f"{args.image_dir}/train/HR"
    lrbicx2_image_dir = f"{args.image_dir}/train/LRbicx2"
    lrbicx3_image_dir = f"{args.image_dir}/train/LRbicx3"
    lrbicx4_image_dir = f"{args.image_dir}/train/LRbicx4"

    # Create dataset folder
    if os.path.exists(hr_image_dir):
        shutil.rmtree(hr_image_dir)
    if os.path.exists(lrbicx2_image_dir):
        shutil.rmtree(lrbicx2_image_dir)
    if os.path.exists(lrbicx3_image_dir):
        shutil.rmtree(lrbicx3_image_dir)
    if os.path.exists(lrbicx4_image_dir):
        shutil.rmtree(lrbicx4_image_dir)

    os.makedirs(hr_image_dir)
    os.makedirs(lrbicx2_image_dir)
    os.makedirs(lrbicx3_image_dir)
    os.makedirs(lrbicx4_image_dir)

    # Carry out data enhancement processing on the data set in the GTmod12 catalog in turn
    for file_name in os.listdir(raw_image_dir):
        base_file_name = os.path.basename(file_name)
        raw_image = Image.open(f"{raw_image_dir}/{base_file_name}")

        for scale_ratio in [1.0, 0.9, 0.8, 0.7, 0.6]:
            for rotate_angle in [0, 90, 180, 270]:
                # Process HR image
                new_image = raw_image.resize((int(raw_image.width * scale_ratio), int(raw_image.height * scale_ratio)), Image.BICUBIC)
                new_image = new_image.rotate(rotate_angle, expand=True)
                new_image.save(f"{hr_image_dir}/{base_file_name.split('.')[-2]}_s{scale_ratio}_r{rotate_angle}.png")

                # Process LRbicx2 image
                lrbicx2_image = new_image.resize([new_image.width // 2, new_image.height // 2], Image.BICUBIC)
                lrbicx2_image.save(f"{lrbicx2_image_dir}/{base_file_name.split('.')[-2]}_s{scale_ratio}_r{rotate_angle}.png")

                # Process LRbicx3 image
                lrbicx3_image = new_image.resize([new_image.width // 3, new_image.height // 3], Image.BICUBIC)
                lrbicx3_image.save(f"{lrbicx3_image_dir}/{base_file_name.split('.')[-2]}_s{scale_ratio}_r{rotate_angle}.png")

                # Process LRbicx4 image
                lrbicx4_image = new_image.resize([new_image.width // 4, new_image.height // 4], Image.BICUBIC)
                lrbicx4_image.save(f"{lrbicx4_image_dir}/{base_file_name.split('.')[-2]}_s{scale_ratio}_r{rotate_angle}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts (Use FSRCNN functions).")
    parser.add_argument("--image_dir", type=str, default="T91_General100", help="Path to generator image directory. (Default: `T91_General100`)")
    args = parser.parse_args()

    main(args)
