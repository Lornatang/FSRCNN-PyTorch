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
from tqdm import tqdm


def main():
    raw_image_dir = f"{args.output_dir}/temp"
    new_image_dir = f"{args.output_dir}/train"

    if os.path.exists(raw_image_dir):
        shutil.rmtree(raw_image_dir)
    if os.path.exists(new_image_dir):
        shutil.rmtree(new_image_dir)
    os.makedirs(raw_image_dir)
    os.makedirs(new_image_dir)

    # Carry out data enhancement processing on the data set in the temp catalog in turn
    file_names = os.listdir(args.inputs_dir)
    for file_name in tqdm(file_names, total=len(file_names)):
        raw_image = Image.open(f"{args.inputs_dir}/{file_name}")

        index = 0
        for scale_ratio in [1.0, 0.9, 0.7, 0.5]:
            for rotate_angle in [0, 90, 180]:
                # Process HR image
                hr_image = raw_image.resize((int(raw_image.width * scale_ratio), int(raw_image.height * scale_ratio)), Image.BICUBIC) if scale_ratio != 1.0 else raw_image
                hr_image = hr_image.rotate(rotate_angle) if rotate_angle != 0 else hr_image
                # Save all images
                hr_image.save(f"{raw_image_dir}/{file_name.split('.')[-2]}_{index:04d}.{file_name.split('.')[-1]}")
                index += 1
    print("Data augment successful.")

    file_names = os.listdir(raw_image_dir)
    for file_name in tqdm(file_names, total=len(file_names)):
        # Use PIL to read high-resolution image
        image = Image.open(f"{raw_image_dir}/{file_name}")

        for pos_x in range(0, image.size[0] - args.image_size + 1, args.step):
            for pos_y in range(0, image.size[1] - args.image_size + 1, args.step):
                crop_image = image.crop([pos_x, pos_y, pos_x + args.image_size, pos_y + args.image_size])
                # Save all images
                crop_image.save(f"{new_image_dir}/{file_name.split('.')[-2]}_{pos_x}_{pos_y}.{file_name.split('.')[-1]}")

    shutil.rmtree(raw_image_dir)
    print("Data split successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--inputs_dir", type=str, default="TG191/original", help="Path to input image directory. (Default: `TG191/original`)")
    parser.add_argument("--output_dir", type=str, default="TG191", help="Path to generator image directory. (Default: `TG191`)")
    parser.add_argument("--image_size", type=int, default=33, help="Low-resolution image size from raw image. (Default: 33)")
    parser.add_argument("--step", type=int, default=14, help="Crop image similar to sliding window.  (Default: 14)")
    args = parser.parse_args()

    main()
