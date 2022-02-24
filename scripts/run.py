import os

# Prepare dataset
os.system("python ./augment_dataset.py --images_dir ../data/T91/original --output_dir ../data/T91/FSRCNN/original --num_workers 10")
os.system("python ./prepare_dataset.py --images_dir ../data/T91/FSRCNN/original --output_dir ../data/T91/FSRCNN/train --image_size 32 --step 16 --num_workers 10")

# Split train and valid
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/T91/FSRCNN/train --valid_images_dir ../data/T91/FSRCNN/valid --valid_samples_ratio 0.1")
