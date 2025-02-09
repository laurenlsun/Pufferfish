import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from lxml import etree
import shutil

def data_loader(batch_size, is_train=True):
    #imagnet                                                                                                                 
    

    if is_train:
        traindir = os.path.join('/data/imagenet/content/ILSVRC/Data/CLS-LOC/train')
        # print("TRAINDIR:", traindir)
        # for filename in os.listdir(traindir):
        #     file_path = os.path.join(traindir, filename)
        #     print(file_path)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                 std = [ 0.229, 0.224, 0.225 ]),
        ])
    else:
        traindir = os.path.join('/data/imagenet/content/ILSVRC/Data/CLS-LOC/val')
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                 std = [ 0.229, 0.224, 0.225 ]),
        ])

    # "data/imagenet/content/ILSVRC/Annotations/CLS-LOC/val"

    train = datasets.ImageFolder(traindir, transform)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader

def parse_xmls():
    dict = {}
    dir = os.path.join("/data/imagenet/content/ILSVRC/Annotations/CLS-LOC/val")
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        tree = etree.parse(file_path)
        foldername = tree.xpath("//name/text()")[0] # tree.xpath returns list; just take 1st item because they're all the same
        jpegname = filename.replace("xml", "JPEG")
        dict[jpegname] = foldername
        # print(jpegname, "--------> folder", foldername)
    return dict

def move_files(dict):
    source_dir = os.path.join("/data/imagenet/content/ILSVRC/Data/CLS-LOC/val")
    target_base_dir = os.path.join("/data/imagenet/content/ILSVRC/Data/CLS-LOC/val")
    for filename, folder in dict.items():
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(target_base_dir, folder, filename)
        os.makedirs(os.path.join(target_base_dir, folder), exist_ok=True)
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
            print(f"Moved: {filename} --> {dest_path}")
        else:
            print(f"Warning: {filename} not found in {source_dir}")


if __name__=="__main__":
    # print all filenames in val
    dir = os.path.join("/data/imagenet/content/ILSVRC/Data/CLS-LOC/val")
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        print(file_path)

    # mapping = parse_xmls()
    # move_files(mapping)
    
