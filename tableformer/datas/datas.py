import os
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
import json
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms as T

from .utils import TagConverter


class SynthTabNetDataset(Dataset):
    def __init__(self, img_dir: str, anno_path: str, type: str, input_size=448, *args, **kwargs):
        super().__init__()

        assert type in ['train', 'val', 'test']
        self.img_dir = img_dir

        self.transforms = torch.nn.Sequential(
            T.Resize((input_size, input_size)),
            #T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        )
        # T.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.))

        self.input_size = input_size
        self.max_boxes = 500

        self.converter = TagConverter()

        with open(anno_path, 'r') as json_file:
            json_list = list(json_file)
        print(f'loading dataset: annotation number: {len(json_list)}')
        self.data_dict = dict()
        data_idx = 0
        for j in tqdm(json_list, total=len(json_list)):
            anno = json.loads(j)
            if anno['split'] == type:
                imgid = anno['imgid']
                self.data_dict[data_idx] = {
                    'img_path': os.path.join(img_dir, anno['split'], anno['filename']),
                    'tags': anno['html']['structure']['tokens'],
                    'boxes': [i['bbox'][:4] for i in anno['html']['cells']], # boxes数量不相同会导致batch拼接时出错
                    'classes': [i['bbox'][-1] for i in anno['html']['cells']]
                }
                # boxes_padding = [[0, 0, 0, 0] for _ in range(self.max_boxes - len(anno['html']['cells']))]
                # classes_padding = [0 for _ in range(self.max_boxes - len(anno['html']['cells']))]
                # self.data_dict[data_idx]['boxes'] += boxes_padding
                # self.data_dict[data_idx]['classes'] += classes_padding
                data_idx += 1
        print(f'data number for {type}: {len(self.data_dict)}')
    def preprocess(self, img_path, tags, boxes, classes):
        img = read_image(img_path)
        tags = self.converter.encode(tags)  # 被填充至max_len+2
        scale = (self.input_size / img.shape[1], self.input_size / img.shape[2])
        img = self.transforms(img)
        boxes_ = torch.tensor(boxes)
        boxes_[:, ::2] = boxes_[:, ::2] * scale[0]
        boxes_[:, 1::2] = boxes_[:, 1::2] * scale[1]
        return img, tags, torch.tensor(boxes, dtype=torch.long), torch.tensor(classes, dtype=torch.long)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        try:
            anno = self.data_dict[idx]
            img_path, tags, boxes, classes = anno['img_path'], anno['tags'], anno['boxes'], anno['classes']
            return self.preprocess(img_path, tags, boxes, classes)
        except:
            print(f"failed to load {idx}'s data")
            return None, None, None, None


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        if args.phase == 'train':
            self.train_data = SynthTabNetDataset(
                img_dir=args.img_path,
                anno_path=args.gt_path,
                type='train',
                input_size=args.input_size,
            )
        elif args.phase =='test':
            self.test_data = SynthTabNetDataset(
                img_dir=args.img_path,
                anno_path=args.gt_path,
                type='test',
                input_size=args.input_size,
            )

        self.num_workers = args.num_workers
        self.batch_size = args.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers, batch_size=self.batch_size)