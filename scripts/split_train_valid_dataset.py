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
import random
import shutil
import argparse
from tqdm import tqdm
import os


def main(args):
    train_hr_image_dir = f"{args.image_dir}/train/HR/"
    train_lrbicx2_image_dir = f"{args.image_dir}/train/LRbicx2"
    train_lrbicx3_image_dir = f"{args.image_dir}/train/LRbicx3"
    train_lrbicx4_image_dir = f"{args.image_dir}/train/LRbicx4"

    valid_hr_image_dir = f"{args.image_dir}/valid/HR"
    valid_lrbicx2_image_dir = f"{args.image_dir}/valid/LRbicx2"
    valid_lrbicx3_image_dir = f"{args.image_dir}/valid/LRbicx3"
    valid_lrbicx4_image_dir = f"{args.image_dir}/valid/LRbicx4"

    if not os.path.exists(valid_hr_image_dir):
        os.makedirs(valid_hr_image_dir)
    if not os.path.exists(valid_lrbicx2_image_dir):
        os.makedirs(valid_lrbicx2_image_dir)
    if not os.path.exists(valid_lrbicx3_image_dir):
        os.makedirs(valid_lrbicx3_image_dir)
    if not os.path.exists(valid_lrbicx4_image_dir):
        os.makedirs(valid_lrbicx4_image_dir)

    # Divide 10% of the data into the validation dataset
    train_files = os.listdir(train_hr_image_dir)
    valid_files = random.sample(train_files, int(len(train_files) * args.valid_samples_ratio))

    for image_file_name in tqdm(valid_files, total=len(valid_files)):
        train_hr_image_path = f"{train_hr_image_dir}/{image_file_name}"
        train_lrbicx2_image_path = f"{train_lrbicx2_image_dir}/{image_file_name}"
        train_lrbicx3_image_path = f"{train_lrbicx3_image_dir}/{image_file_name}"
        train_lrbicx4_image_path = f"{train_lrbicx4_image_dir}/{image_file_name}"

        valid_hr_image_path = f"{valid_hr_image_dir}/{image_file_name}"
        valid_lrbicx2_image_path = f"{valid_lrbicx2_image_dir}/{image_file_name}"
        valid_lrbicx3_image_path = f"{valid_lrbicx3_image_dir}/{image_file_name}"
        valid_lrbicx4_image_path = f"{valid_lrbicx4_image_dir}/{image_file_name}"

        shutil.copyfile(train_hr_image_path, valid_hr_image_path)
        shutil.copyfile(train_lrbicx2_image_path, valid_lrbicx2_image_path)
        shutil.copyfile(train_lrbicx3_image_path, valid_lrbicx3_image_path)
        shutil.copyfile(train_lrbicx4_image_path, valid_lrbicx4_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train and valid dataset scripts (Use SRCNN functions).")
    parser.add_argument("--image_dir", type=str, default="T91_General100", help="Path to generator image directory. (Default: `T91_General100`)")
    parser.add_argument("--valid_samples_ratio", type=float, default=0.1, help="What percentage of the data is extracted from the training set into the validation set.  (Default: 0.1)")
    args = parser.parse_args()

    main(args)
