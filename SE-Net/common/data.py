import torch
import random
import torchvision.transforms as T
from . import utils
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data


class Siamese_Triplet_Gaze(Dataset):
    """
    Human gaze data for two-pathway dense transformer
    """
    def __init__(self,
                 root_dir,
                 fix_labels,
                 bbox_annos,
                 pa,
                 transform,
                 catIds,
                 device, 
                 blur_action=False,
                 acc_foveal=True):
        self.root_dir = root_dir
        self.pa = pa
        self.transform = transform
        self.to_tensor = T.ToTensor()
        self.device = device

        
        # Remove fixations longer than max_traj_length
        self.fix_labels = list(
            filter(lambda x: len(x[3]) <= pa.max_traj_length, fix_labels))
        

        self.catIds = catIds
        self.blur_action = blur_action
        self.acc_foveal = acc_foveal
        self.bboxes = bbox_annos

        self.fv_tid = 0 if self.pa.TAP == 'FV' else len(self.catIds)

        if self.pa.name == 'COCO-Search18':
            task_emb_dict = np.load(f'{self.root_dir}/coco_search18_embeddings.npy', allow_pickle=True).item()
        else:
            task_emb_dict = np.load(f'{self.root_dir}/osie_embeddings.npy', allow_pickle=True).item()
        self.task_emb_dict = task_emb_dict

    def __len__(self):
        return len(self.fix_labels)
    

    def __getitem__(self, idx):
        anchor_data = self.fix_labels[idx]
        anchor_img_name = anchor_data[0]
        anchor_subject_id = anchor_data[-3]
        anchor = self.process_data(idx)
        if self.pa.num_fewshot == 1:
            positive = anchor
        else:
            positive_idx = random.choice(
                [i for i, data in enumerate(self.fix_labels) if data[-3] == anchor_subject_id and i != idx]
            )
            positive = self.process_data(positive_idx)

        # Find a negative example (different subject_id, same img_name)
        if self.pa.num_fewshot == 1:
            negative = positive
        else:
            negative_idx = random.choice(
                # [i for i, data in enumerate(self.fix_labels) if data[-3] != anchor_subject_id and anchor_img_name!=data[0]]
                [i for i, data in enumerate(self.fix_labels) if data[-3] != anchor_subject_id])
            negative = self.process_data(negative_idx)

        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }
        
    def process_data(self, idx):
        img_name, cat_name, condition, fixs, action, is_last, sid, dura, dataset = self.fix_labels[
            idx]
        imgId = cat_name + '_' + img_name
        if imgId in self.bboxes.keys():
            bbox = torch.Tensor(self.bboxes[imgId])
        else:
            bbox = torch.Tensor([1, 2, 3, 4])

        if self.pa.name =='OSIE':
            # im_path = "{}/{}/{}".format(self.root_dir, self.pa.image_path, img_name)
            im_path = "{}/{}".format(self.pa.image_path, img_name)
        else:
            if cat_name == 'none':  # coco-fv
                # im_path = "{}/{}/{}".format(self.root_dir, self.pa.image_path, img_name)
                im_path = "{}/{}".format(self.pa.image_path, img_name)
            else:  # coco-search18
                c = cat_name.replace(' ', '_')
                # im_path = "{}/{}/{}/{}".format(self.root_dir, self.pa.image_path, c, img_name)
                im_path = "{}/{}/{}".format(self.pa.image_path, c, img_name)
        im = Image.open(im_path).convert('RGB')
        im_tensor = self.transform(im.copy())
        assert im_tensor.shape[-1] == self.pa.im_w and im_tensor.shape[-2] == self.pa.im_h, "wrong image size."

        IOR_weight_map = np.zeros((self.pa.im_h, self.pa.im_w), dtype=np.float32)
        IOR_weight_map += 1  # Set base weight to 1

        scanpath_length = len(fixs)
        if scanpath_length == 0:
            fixs = [(0, 0)]
        # Pad fixations to max_traj_lenght
        fixs = fixs + [fixs[-1]] * (self.pa.max_traj_length - len(fixs))
        is_padding = torch.zeros(self.pa.max_traj_length)
        is_padding[scanpath_length:] = 1

        fixs_tensor = torch.FloatTensor(fixs)
        original_fixs = fixs_tensor.clone()
        # Normalize to 0-1 (avoid 1 by adding 1 pixel).
        fixs_tensor /= torch.FloatTensor([self.pa.im_w + 1, self.pa.im_h + 1])

        next_fixs_tensor = fixs_tensor.clone()
        if not is_last:
            x, y = utils.action_to_pos(action, [1, 1],
                                       [self.pa.im_w, self.pa.im_h])
            next_fix = torch.FloatTensor([x, y]) / torch.FloatTensor(
                [self.pa.im_w, self.pa.im_h])
            next_fixs_tensor[scanpath_length:] = next_fix

        target_fix_map = np.zeros(self.pa.im_w * self.pa.im_h,
                                  dtype=np.float32)

        is_fv = condition == 'freeview'

        # process duration
        if len(dura) > self.pa.max_traj_length:
            dura = dura[0:self.pa.max_traj_length]
        else:
            dura = dura + [0] * (self.pa.max_traj_length - len(dura))
        dura = torch.Tensor(dura)

        # process handcraft features
        # if not is_fv:
        #     task_emb = self.task_emb_dict[cat_name.replace(' ', '_')]
        # else:
        #     task_emb = list(self.task_emb_dict.values())[0]
        task_emb = list(self.task_emb_dict.values())[0]

        ret = {
            "task_id": self.fv_tid if is_fv else self.catIds[cat_name],
            "true_state": im_tensor,
            "true_action": torch.tensor([action], dtype=torch.long),
            'img_name': img_name,
            'task_name': cat_name,
            'normalized_fixations': fixs_tensor,
            'is_padding': is_padding,
            'scanpath_length': scanpath_length,
            'duration': dura,
            'subject_id': sid,
            'original_fixs': original_fixs,
            'task_emb': task_emb,
            'bbox': bbox
        }
        return ret
    