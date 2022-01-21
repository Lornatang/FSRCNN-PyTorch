import os

# Prepare dataset
# create multiscale dataset
# os.system("python ./create_multiscale_dataset.py --images_dir ../data/TG191/original --output_dir ../data/TG191/FSRCNN/original")
# # split image
# os.system("python ./prepare_dataset.py --images_dir ../data/TG191/FSRCNN/original --output_dir ../data/TG191/FSRCNN/train --image_size 32 --step 16")
#
# # Split train and valid
# os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/TG191/FSRCNN/train --valid_images_dir ../data/TG191/FSRCNN/valid --valid_samples_ratio 0.1")

# # Create LMDB database file
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TG191/FSRCNN/train --lmdb_path ../data/train_lmdb/FSRCNN/TG191_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TG191/FSRCNN/train --lmdb_path ../data/train_lmdb/FSRCNN/TG191_LRbicx2_lmdb --upscale_factor 2")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TG191/FSRCNN/train --lmdb_path ../data/train_lmdb/FSRCNN/TG191_LRbicx3_lmdb --upscale_factor 3")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TG191/FSRCNN/train --lmdb_path ../data/train_lmdb/FSRCNN/TG191_LRbicx4_lmdb --upscale_factor 4")

os.system("python ./create_lmdb_dataset.py --images_dir ../data/TG191/FSRCNN/valid --lmdb_path ../data/valid_lmdb/FSRCNN/TG191_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TG191/FSRCNN/valid --lmdb_path ../data/valid_lmdb/FSRCNN/TG191_LRbicx2_lmdb --upscale_factor 2")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TG191/FSRCNN/valid --lmdb_path ../data/valid_lmdb/FSRCNN/TG191_LRbicx3_lmdb --upscale_factor 3")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TG191/FSRCNN/valid --lmdb_path ../data/valid_lmdb/FSRCNN/TG191_LRbicx4_lmdb --upscale_factor 4")
