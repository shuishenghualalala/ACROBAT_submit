import torch
import torch.utils.data as data
from torchvision import transforms
import os
import numpy as np
import cv2
class ACROBATDataset(data.Dataset):
    def __init__(self,args,phase='train'):
        if  phase == 'train':
            pairs_path = args.Pair_path
        elif phase == 'validation':
            pairs_path = args.Pair_path_validate
        elif phase == 'test':
            pairs_path = args.Pair_path_test
        self.pair_list = [os.path.join(pairs_path,i) for i in os.listdir(pairs_path)]
        self.patch_size = args.patch_size
    def __len__(self):
        return len(self.pair_list)
    def __getitem__(self, indix):
        pair = self.pair_list[indix]
        imgs = [os.path.join(pair,i) for i in os.listdir(pair)]
        indicator = int(imgs[0].split('/')[-2]) # pair id
        if 'source' in imgs[0]:
            source_path = imgs[0]
            target_path = imgs[1]
        else:
            source_path = imgs[1]
            target_path = imgs[0]

        source , target = cv2.imread(source_path),cv2.imread(target_path)
        source = torch.from_numpy(source).permute(2, 0, 1)
        target = torch.from_numpy(target).permute(2, 0, 1)
        transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        source = transform(source)
        target = transform(target)
        return {'source':source,'target':target,'id':indicator}


def load_dataloader(args,phase):
    if phase == 'train':
        Train_dataset = ACROBATDataset(args,phase)
        Train_dataloader = data.DataLoader(Train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers, drop_last=True)
        return Train_dataset, Train_dataloader

    elif phase == 'validation':
        Val_dataset = ACROBATDataset(args,phase)
        Val_dataloader = data.DataLoader(Val_dataset, batch_size=args.batch_size_eval,
                                             shuffle=False, num_workers=args.num_workers)
        return Val_dataset, Val_dataloader
    elif phase == 'test':
        Test_dataset = ACROBATDataset(args, phase)
        Test_dataloader = data.DataLoader(Test_dataset, batch_size=args.batch_size_eval,
                                          shuffle=False, num_workers=args.num_workers)
        return Test_dataset, Test_dataloader
