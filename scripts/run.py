import os

# Prepare dataset
os.system("python ./create_multiscale_dataset.py --images_dir ../data/TG191/original --output_dir ../data/TG191/FSRCNN/original")
os.system("python ./prepare_dataset.py --images_dir ../data/TG191/FSRCNN/original --output_dir ../data/TG191/FSRCNN/train --image_size 32 --step 16")

# Split train and valid
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/TG191/FSRCNN/train --valid_images_dir ../data/TG191/FSRCNN/valid --valid_samples_ratio 0.1")
