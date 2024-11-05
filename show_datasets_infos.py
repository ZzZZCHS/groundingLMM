import h5py
import glob
import os
import json

data_dir = '/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_1013'

for filepath in glob.glob(os.path.join(data_dir, "*.hdf5")):
    with h5py.File(filepath, 'r') as f:
        print(json.loads(f['data'].attrs['env_args'])['env_name'], len(f['data'].keys()))