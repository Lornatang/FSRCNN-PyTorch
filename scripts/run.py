import os

# Prepare dataset
os.system("python3 ./prepare_dataset.py --inputs_dir ../data/TG191/original --output_dir ../data/TG191/FSRCNN/")

# Split train and valid
os.system("python3 ./split_train_valid_dataset.py --inputs_dir ../data/TG191/FSRCNN")

# Create LMDB database file
os.system("python3 ./create_lmdb_database.py --image_dir ../data/TG191/FSRCNN/train --lmdb_path ../data/train_lmdb/FSRCNN/TG191_HR_lmdb --upscale_factor 1")
os.system("python3 ./create_lmdb_database.py --image_dir ../data/TG191/FSRCNN/train --lmdb_path ../data/train_lmdb/FSRCNN/TG191_LRbicx2_lmdb --upscale_factor 2")
os.system("python3 ./create_lmdb_database.py --image_dir ../data/TG191/FSRCNN/train --lmdb_path ../data/train_lmdb/FSRCNN/TG191_LRbicx3_lmdb --upscale_factor 3")
os.system("python3 ./create_lmdb_database.py --image_dir ../data/TG191/FSRCNN/train --lmdb_path ../data/train_lmdb/FSRCNN/TG191_LRbicx4_lmdb --upscale_factor 4")

os.system("python3 ./create_lmdb_database.py --image_dir ../data/TG191/FSRCNN/valid --lmdb_path ../data/valid_lmdb/FSRCNN/TG191_HR_lmdb --upscale_factor 1")
os.system("python3 ./create_lmdb_database.py --image_dir ../data/TG191/FSRCNN/valid --lmdb_path ../data/valid_lmdb/FSRCNN/TG191_LRbicx2_lmdb --upscale_factor 2")
os.system("python3 ./create_lmdb_database.py --image_dir ../data/TG191/FSRCNN/valid --lmdb_path ../data/valid_lmdb/FSRCNN/TG191_LRbicx3_lmdb --upscale_factor 3")
os.system("python3 ./create_lmdb_database.py --image_dir ../data/TG191/FSRCNN/valid --lmdb_path ../data/valid_lmdb/FSRCNN/TG191_LRbicx4_lmdb --upscale_factor 4")
