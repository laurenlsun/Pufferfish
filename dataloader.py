import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
# from lxml import etree # only needed the one time you parse xml
import shutil

def data_loader(batch_size, is_train=True):
    #imagnet                                                                                                                 
    if is_train:
        traindir = os.path.join('/home/yongqin/imagenet/content/ILSVRC/Data/CLS-LOC/train')
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
        traindir = os.path.join('/home/yongqin/imagenet/content/ILSVRC/Data/CLS-LOC/val')
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

def test_loader(batch_size):
    #imagnet                                                                                                                 
    testdir = os.path.join('/home/yongqin/imagenet/content/ILSVRC/Data/CLS-LOC/test')
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                std = [ 0.229, 0.224, 0.225 ]),
    ])

    test = datasets.ImageFolder(testdir, transform)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=0)
    return test_loader

def parse_xmls(dir):
    dict = {}
    dir = os.path.join(dir)
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        tree = etree.parse(file_path)
        # parse for <name> tag (class name)
        foldername = tree.xpath("//name/text()")[0] # tree.xpath returns list; just take 1st item because they're all the same
        jpegname = filename.replace("xml", "JPEG")
        dict[jpegname] = foldername
        print(jpegname, "--------> folder", foldername)
    return dict

def move_files(dict, source_dir, target_base_dir):
    source_dir = os.path.join(source_dir)
    target_base_dir = os.path.join(target_base_dir)
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
    # dir = os.path.join("/home/yongqin/imagenet/content/ILSVRC/Data/CLS-LOC/val/n02447366")
    # for filename in os.listdir(dir):
    #     file_path = os.path.join(dir, filename)
    #     print(file_path)

    valanndir="/home/yongqin/imagenet/content/ILSVRC/Annotations/CLS-LOC/val"
    # mapping = parse_xmls(valanndir)
    valdatadir = "/home/yongqin/imagenet/content/ILSVRC/Data/CLS-LOC/val"
    # move_files(mapping, valsdatadir, valdatadir)
    
    # files in test set not sorted into classes

    testanndir = "/home/yongqin/imagenet/content/ILSVRC/Annotations/CLS-LOC/test"
    testdatadir = "/home/yongqin/imagenet/content/ILSVRC/Data/CLS-LOC/test"
    # mapping = parse_xmls(testanndir)
    # print(mapping)


    dir = os.path.join("/home/yongqin/imagenet/content/ILSVRC/ImageSets/CLS-LOC")
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        print(file_path)