import os
import cv2
import json
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import math
from pycocotools import mask
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import GCG_QUESTIONS
from dataset.gcg_datasets.GranDf_gcg_ds import GCGBaseDataset
from tqdm import tqdm
import tools.lang_utils as LangUtils
import tools.torch_utils as TorchUtils
import tools.action_utils as AcUtils
import tools.tensor_utils as TensorUtils
import tools.obs_utils as ObsUtils
from tools.prompt_templates import GVLA_PROMPT_TEMPLATE
import h5py
from contextlib import contextmanager
from collections import OrderedDict


class GroundedVLADataset(GCGBaseDataset):
    def __init__(
        self,
        hdf5_path,
        raw_data_dir,
        obs_keys,
        action_keys,
        dataset_keys,
        action_config,
        frame_stack=10,
        seq_length=10,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=True,
        **kwargs
    ):
        self.hdf5_path = hdf5_path
        self.dataset_name = os.path.basename(hdf5_path).split('.hdf5')[0]
        self.hdf5_use_swmr = True
        self._hdf5_file = None
        
        self.image_folder = os.path.join(raw_data_dir, "images")
        if not os.path.exists(self.image_folder):
            os.mkdir(self.image_folder)
        
        self.obs_keys = obs_keys
        self.action_keys = tuple(action_keys)
        self.dataset_keys = tuple(dataset_keys)
        self.camera_names = tuple(["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"])
        
        self.action_config = action_config

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask
        
        self.load_demo_info()
        
        # maybe prepare for observation normalization
        self.obs_normalization_stats = None
        # if self.hdf5_normalize_obs:
        #     self.obs_normalization_stats = self.normalize_obs()

        # prepare for action normalization
        self.action_normalization_stats = None
        
        self.hdf5_cache = None
        
        self.shuffled_obs_key_groups = list()

        self.close_and_delete_hdf5_handle()
        
        super().__init__(**kwargs)
    
    def load_demo_info(self, filter_by_attribute=None, demos=None):
        """
        Args:
            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load

            demos (list): list of demonstration keys to load from the hdf5 file. If 
                omitted, all demos in the file (or under the @filter_by_attribute 
                filter key) are used.
        """
        # filter demo trajectory by mask
        if demos is not None:
            self.demos = demos
        elif filter_by_attribute is not None:
            self.demos = [elem.decode("utf-8") for elem in np.array(self.hdf5_file["mask/{}".format(filter_by_attribute)][:])]
        else:
            self.demos = list(self.hdf5_file["data"].keys())

        # sort demo keys
        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        inds = inds[:-len(inds)//20]  # leave 1/20 for validation
        self.demos = [self.demos[i] for i in inds]

        self.n_demos = len(self.demos)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()
        self._demo_id_to_demo_lang_str = dict() # language annotation per demo id
        self._demo_id_to_demo_lang_emb = dict() # language embedding per demo id
        self._demo_id_to_demo_target_obj_phrase = dict()
        self._demo_id_to_demo_target_place_phrase = dict()

        # determine index mapping
        self.total_num_sequences = 0
        for ep in self.demos:
            demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length

            ep_meta = self.hdf5_file["data/{}".format(ep)].attrs.get("ep_meta", None)
            if ep_meta is not None:
                ep_meta = json.loads(ep_meta)
                lang = ep_meta.get("lang", None)
                target_obj_phrase = ep_meta.get("target_obj_phrase", None)
                target_place_phrase = ep_meta.get("target_place_phrase", None)
                self._demo_id_to_demo_lang_str[ep] = lang
                self._demo_id_to_demo_target_obj_phrase[ep] = target_obj_phrase
                self._demo_id_to_demo_target_place_phrase[ep] = target_place_phrase

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1

        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        lang_encoder = LangUtils.LangEncoder(
            device=device,
        )
        
        if len(self._demo_id_to_demo_lang_str) > 0:
            print("getting language embeddings...")
            for ep_batch in tqdm(np.array_split(self.demos, int(math.ceil(len(self.demos) / 64)))):
                # get language embedding
                lang_batch = [self._demo_id_to_demo_lang_str[ep] for ep in ep_batch]
                emb_batch = lang_encoder.get_lang_emb(lang_batch)
                # emb_batch = torch.randn(50, 768)
                emb_batch = TensorUtils.to_numpy(emb_batch)
                for batch_idx, ep in enumerate(ep_batch):
                    self._demo_id_to_demo_lang_emb[ep] = emb_batch[batch_idx]

        del lang_encoder
    
    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None
    
    @contextmanager
    def hdf5_file_opened(self):
        """
        Convenient context manager to open the file on entering the scope
        and then close it on leaving.
        """
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()
    
    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in 
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences

    def get_action_traj(self, ep):
        action_traj = dict()
        for key in self.action_keys:
            action_traj[key] = self.hdf5_file["data/{}/{}".format(ep, key)][()].astype('float32')
        return action_traj
    
    def get_action_stats(self):
        ep = self.demos[0]
        action_traj = self.get_action_traj(ep)
        action_stats = _compute_traj_stats(action_traj)
        # print("SequenceDataset: normalizing actions...")
        for ep in self.demos[1:]:
            action_traj = self.get_action_traj(ep)
            traj_stats = _compute_traj_stats(action_traj)
            action_stats = _aggregate_traj_stats(action_stats, traj_stats)
        return action_stats
    
    def set_action_normalization_stats(self, action_normalization_stats):
        self.action_normalization_stats = action_normalization_stats

    def get_action_normalization_stats(self):
        """
        Computes a dataset-wide min, max, mean and standard deviation for the actions 
        (per dimension) and returns it.
        """
        
        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        if self.action_normalization_stats is None:
            action_stats = self.get_action_stats()
            self.action_normalization_stats = action_stats_to_normalization_stats(
                action_stats, self.action_config)
        return self.action_normalization_stats
    
    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """
        hd5key = "data/{}/{}".format(ep, key)
        ret = self.hdf5_file[hd5key]
        return ret
    
    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        gcg_data, output = self.get_item(index)

        return gcg_data, output
    
    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1, # note: need to decrement self.n_frame_stack by one
            seq_length=self.seq_length
        )

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )

        # get action components
        ac_dict = OrderedDict()
        for k in self.action_keys:
            ac = meta[k]
            # expand action shape if needed
            if len(ac.shape) == 1:
                ac = ac.reshape(-1, 1)
            ac_dict[k] = ac
       
        # normalize actions
        action_normalization_stats = self.get_action_normalization_stats()
        ac_dict = ObsUtils.normalize_dict(ac_dict, normalization_stats=action_normalization_stats)

        # concatenate all action components
        meta["actions"] = AcUtils.action_dict_to_vector(ac_dict)

        # also return the sampled index
        meta["index"] = index

        if demo_id in self._demo_id_to_demo_lang_emb:
            # language embedding
            T = meta["actions"].shape[0]
            meta["obs"]["lang_emb"] = np.tile(
                self._demo_id_to_demo_lang_emb[demo_id],
                (T, 1)
            )
        
        gcg_data = self.process_gcg_data(demo_id)

        return gcg_data, meta
    
    def process_gcg_data(self, demo_id):
        cam_name = self.camera_names[0] # left
        image_key = f"{cam_name}_image"
        mask_key = f"{cam_name}_mask"
        image_data = self.get_dataset_for_ep(demo_id, f"obs/{image_key}")
        mask_data = self.get_dataset_for_ep(demo_id, f"obs/{mask_key}")
        # image_path = os.path.join(self.image_folder, f"{self.dataset_name}_{demo_id}_{image_key}.jpg")
        # if not os.path.exists(image_path):
        #     image = Image.fromarray(image_data[0])
        #     image.save(image_path)
            
        
        lang = self._demo_id_to_demo_lang_str[demo_id]
        target_obj_phrase = "the " + self._demo_id_to_demo_target_obj_phrase[demo_id]  # check if the phrase starts with "the "
        target_place_phrase = "the " + self._demo_id_to_demo_target_place_phrase[demo_id] if self._demo_id_to_demo_target_place_phrase[demo_id] is not None else None
        
        caption = f"The target object is {target_obj_phrase}."
        data_labels = [target_obj_phrase]
        st_idx = caption.find(target_obj_phrase)
        tokens_positive = [[st_idx, st_idx + len(target_obj_phrase)]]
        masks = [(mask_data[0] == 1).astype(np.uint8)]
        if target_place_phrase is not None:
            caption += f" The target placement area is {target_place_phrase}."
            data_labels.append(target_place_phrase)
            st_idx = caption.find(target_place_phrase)
            tokens_positive.append([st_idx, st_idx + len(target_place_phrase)])
            masks.append((mask_data[0] == 2).astype(np.uint8))
        else:
            caption += f" No target placement area."
        
        prompt = GVLA_PROMPT_TEMPLATE.format(lang)
        
        image = image_data[0]
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        bboxes = None
        questions, conversations = self.create_conversations(caption, tokens_positive, prompt)
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1:], dtype=torch.long) * self.IGNORE_LABEL
        
        return ("", global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize, questions, data_labels)
        

    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        masked_keys = [
            "obs/robot0_agentview_left_mask",
            "obs/robot0_agentview_right_mask",
            "obs/robot0_eye_in_hand_mask"
        ]
        depth_keys = [
            "obs/robot0_agentview_left_depth",
            "obs/robot0_agentview_right_depth",
            "obs/robot0_eye_in_hand_depth"
        ]
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            if k in masked_keys:
                seq[k] = data[:1].repeat(seq_end_index-seq_begin_index, axis=0)
            else:
                seq[k] = data[seq_begin_index: seq_end_index]
        
        if ObsUtils.MASK_CHANNEL == 1:
            for k in masked_keys:
                image_key = k.replace('_mask', '_image')
                if image_key not in keys:
                    continue
                data = self.get_dataset_for_ep(demo_id, k)
                mask_data = data[:1].repeat(seq_end_index-seq_begin_index, axis=0) * 127
                seq[image_key] = np.concatenate([seq[image_key], mask_data], axis=-1)
        
        if ObsUtils.DEPTH_CHANNEL == 1:
            for k in depth_keys:
                image_key = k.replace('_depth', '_image')
                if image_key not in keys:
                    continue
                data = self.get_dataset_for_ep(demo_id, k)
                depth_data = data[seq_begin_index: seq_end_index] * 255
                depth_data = depth_data.astype(np.uint8)
                seq[image_key] = np.concatenate([seq[image_key], depth_data], axis=-1)

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return seq, pad_mask
    
    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, prefix="obs"):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
            prefix (str): one of "obs", "next_obs"

        Returns:
            a dictionary of extracted items.
        """
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        obs = {'/'.join(k.split('/')[1:]): obs[k] for k in obs}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        return obs

    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).
        
        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def get_trajectory_at_index(self, index):
        """
        Method provided as a utility to get an entire trajectory, given
        the corresponding @index.
        """
        demo_id = self.demos[index]
        demo_length = self._demo_id_to_demo_length[demo_id]

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1, # note: need to decrement self.n_frame_stack by one
            seq_length=demo_length
        )
        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.obs_keys,
            seq_length=demo_length
        )
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=self.obs_keys,
                seq_length=demo_length,
                prefix="next_obs"
            )

        meta["ep"] = demo_id
        return meta
    

class CustomWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """
    WeightedRandomSampler except allows for more than 2^24 samples to be sampled
    copied from https://github.com/pytorch/pytorch/issues/2576#issuecomment-831780307
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


class GroundedVLAMetaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets,
        ds_weights=None,
        normalize_weights_by_ds_size=False,
    ):
        super(GroundedVLAMetaDataset, self).__init__()
        self.datasets = datasets
        ds_lens = np.array([len(ds) for ds in self.datasets])
        if ds_weights is None:
            ds_weights = [1.0] * len(datasets)
        if normalize_weights_by_ds_size:
            self.ds_weights = np.array(ds_weights) / ds_lens
        else:
            self.ds_weights = ds_weights
        self._ds_ind_bins = np.cumsum([0] + list(ds_lens))

        # TODO: comment
        action_stats = self.get_action_stats()
        self.action_normalization_stats = action_stats_to_normalization_stats(
            action_stats, self.datasets[0].action_config)
        self.set_action_normalization_stats(self.action_normalization_stats)
    
    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

    def __getitem__(self, idx):
        ds_ind = np.digitize(idx, self._ds_ind_bins) - 1
        ind_in_ds = idx - self._ds_ind_bins[ds_ind]
        gcg_data, meta = self.datasets[ds_ind].__getitem__(ind_in_ds)
        meta["index"] = idx
        return *gcg_data, meta

    def get_ds_label(self, idx):
        ds_ind = np.digitize(idx, self._ds_ind_bins) - 1
        ds_label = self.ds_labels[ds_ind]
        return ds_label
    
    def get_ds_id(self, idx):
        ds_ind = np.digitize(idx, self._ds_ind_bins) - 1
        ds_label = self.ds_labels[ds_ind]
        return self.ds_labels_to_ids[ds_label]

    def __repr__(self):
        str_output = '\n'.join([ds.__repr__() for ds in self.datasets])
        return str_output

    def get_dataset_sampler(self):
        if np.all(np.array(self.ds_weights) == 1):
            """
            if all weights are 1, then no need to use weighted sampler
            """
            return None
        
        weights = np.ones(len(self))
        for i, (start, end) in enumerate(zip(self._ds_ind_bins[:-1], self._ds_ind_bins[1:])):
            weights[start:end] = self.ds_weights[i]

        # sampler = torch.utils.data.WeightedRandomSampler(
        sampler = CustomWeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def get_action_stats(self):
        meta_action_stats = self.datasets[0].get_action_stats()
        for dataset in self.datasets[1:]:
            ds_action_stats = dataset.get_action_stats()
            meta_action_stats = _aggregate_traj_stats(meta_action_stats, ds_action_stats)
            
        return meta_action_stats
    
    def set_action_normalization_stats(self, action_normalization_stats):
        self.action_normalization_stats = action_normalization_stats
        for ds in self.datasets:
            ds.set_action_normalization_stats(self.action_normalization_stats)

    def get_action_normalization_stats(self):
        """
        Computes a dataset-wide min, max, mean and standard deviation for the actions 
        (per dimension) and returns it.
        """
        
        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        if self.action_normalization_stats is None:
            action_stats = self.get_action_stats()
            self.action_normalization_stats = action_stats_to_normalization_stats(
                action_stats, self.datasets[0].action_config)
        return self.action_normalization_stats

    
def _compute_traj_stats(traj_obs_dict):
    """
    Helper function to compute statistics over a single trajectory of observations.
    """
    traj_stats = { k : {} for k in traj_obs_dict }
    for k in traj_obs_dict:
        traj_stats[k]["n"] = traj_obs_dict[k].shape[0]
        traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=0, keepdims=True) # [1, ...]
        traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0, keepdims=True) # [1, ...]
        traj_stats[k]["min"] = traj_obs_dict[k].min(axis=0, keepdims=True)
        traj_stats[k]["max"] = traj_obs_dict[k].max(axis=0, keepdims=True)
    return traj_stats


def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
    """
    Helper function to aggregate trajectory statistics.
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    for more information.
    """
    merged_stats = {}
    for k in traj_stats_a:
        n_a, avg_a, M2_a, min_a, max_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"], traj_stats_a[k]["min"], traj_stats_a[k]["max"]
        n_b, avg_b, M2_b, min_b, max_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"], traj_stats_b[k]["min"], traj_stats_b[k]["max"]
        n = n_a + n_b
        mean = (n_a * avg_a + n_b * avg_b) / n
        delta = (avg_b - avg_a)
        M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
        min_ = np.minimum(min_a, min_b)
        max_ = np.maximum(max_a, max_b)
        merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2, min=min_, max=max_)
    return merged_stats


def action_stats_to_normalization_stats(action_stats, action_config):
    action_normalization_stats = OrderedDict()
    for action_key in action_stats.keys():
        # get how this action should be normalized from config, default to None
        norm_method = action_config[action_key].get("normalization", None)
        if norm_method is None:
            # no normalization, unit scale, zero offset
            action_normalization_stats[action_key] = {
                "scale": np.ones_like(action_stats[action_key]["mean"], dtype=np.float32),
                "offset": np.zeros_like(action_stats[action_key]["mean"], dtype=np.float32)
            }
        elif norm_method == "min_max":
            # normalize min to -1 and max to 1
            range_eps = 1e-4
            input_min = action_stats[action_key]["min"].astype(np.float32)
            input_max = action_stats[action_key]["max"].astype(np.float32)
            # instead of -1 and 1 use numbers just below threshold to prevent numerical instability issues
            output_min = -0.999999
            output_max = 0.999999
            
            # ignore input dimentions that is too small to prevent division by zero
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min    

            # expected usage of scale and offset
            # normalized_action = (raw_action - offset) / scale
            # raw_action = scale * normalized_action + offset

            # eq1: input_max = scale * output_max + offset
            # eq2: input_min = scale * output_min + offset

            # solution for scale and offset
            # eq1 - eq2: 
            #   input_max - input_min = scale * (output_max - output_min)
            #   (input_max - input_min) / (output_max - output_min) = scale <- eq3
            # offset = input_min - scale * output_min <- eq4
            scale = input_range / (output_max - output_min)
            offset = input_min - scale * output_min

            offset[ignore_dim] = input_min[ignore_dim] - (output_max + output_min) / 2

            action_normalization_stats[action_key] = {
                "scale": scale,
                "offset": offset
            }
        elif norm_method == "gaussian":
            # normalize to zero mean unit variance
            input_mean = action_stats[action_key]["mean"].astype(np.float32)
            input_std = np.sqrt(action_stats[action_key]["sqdiff"] / action_stats[action_key]["n"]).astype(np.float32)

            # ignore input dimentions that is too small to prevent division by zero
            std_eps = 1e-6
            ignore_dim = input_std < std_eps
            input_std[ignore_dim] = 1.0

            action_normalization_stats[action_key] = {
                "scale": input_mean,
                "offset": input_std
            }
        else:
            raise NotImplementedError(
                'action_config.actions.normalization: "{}" is not supported'.format(norm_method))
    
    return action_normalization_stats
