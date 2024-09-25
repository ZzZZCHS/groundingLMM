import os
import h5py
from tqdm import tqdm
import numpy as np

data_root = 'robocasa_datasets'
ori_data_dir = os.path.join(data_root, 'v0.1/single_stage')

obs_keys = ["masked_robot0_agentview_left_image", "masked_robot0_agentview_right_image", "masked_robot0_eye_in_hand_image"]

src_name = 'demo_gentex_im128_randcams_addmask.hdf5'

for root, dirs, files in tqdm(os.walk(ori_data_dir)):
    for file_name in files:
        if file_name != src_name:
            continue
        file_path = os.path.join(root, src_name)
        print(file_path)
        f = h5py.File(file_path, 'r+')
        data = f['data']
        for demo_id in tqdm(data.keys()):
            tmp_demo = data[demo_id]
            tmp_obs = tmp_demo['obs']
            for k in obs_keys:
                tmp_array = tmp_obs[k][()].copy()
                if tmp_array.ndim == 3:
                    del tmp_obs[k]
                    dset = tmp_obs.create_dataset(k, data=np.expand_dims(tmp_array, axis=0))
        f.close()