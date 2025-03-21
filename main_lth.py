# this script is specificly for the comparison with LTH: https://github.com/facebookresearch/open_lth

from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch.backends.cudnn as cudnn

import logging
import time
import random
import numpy as np
#import models

from vgg import *
#import models
from lowrank_vgg import LowrankVGG19LTH, VGG19LTH
from resnet_cifar10 import *

from ptflops import get_model_complexity_info

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))


best_acc = 0  # best test accuracy


# helper function because otherwise non-empty strings
# evaluate as True
def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


# we track the norm of the model weights:
def norm_calculator(model):
    model_norm = 0
    for param_index, param in enumerate(model.parameters()):
        model_norm += torch.norm(param) ** 2
    return torch.sqrt(model_norm).item()


def param_counter(model):
    num_params = 0
    for param_index, (param_name, param) in enumerate(model.named_parameters()):
        num_params += param.numel()
    return num_params


def decompose_weights(model, low_rank_model, rank_factor, args):
    # SVD version
    reconstructed_aggregator = []
    
    if args.arch == "vgg19":
        for item_index, (param_name, param) in enumerate(model.state_dict().items()):
            if len(param.size()) == 4 and item_index not in range(0, 54):
                # resize --> svd --> two layer
                param_reshaped = param.view(param.size()[0], -1)
                rank = min(param_reshaped.size()[0], param_reshaped.size()[1])
                u, s, v = torch.svd(param_reshaped)

                sliced_rank = int(rank/rank_factor)
                u_weight = u * torch.sqrt(s) # alternative implementation: u_weight_alt = torch.mm(u, torch.diag(torch.sqrt(s)))
                v_weight = torch.sqrt(s) * v # alternative implementation: v_weight_alt = torch.mm(torch.diag(torch.sqrt(s)), v.t())
                # sanity check: print("dist u u_alt:{}, dist v v_alt: {}".format(torch.dist(u_weight, u_weight_alt), torch.dist(v_weight.t(), v_weight_alt)))
                # print("## v weight size: {}, v weight alt size: {}".format(v_weight.size(), v_weight_alt.size()))
                #print("layer indeix: {}, dist u u_alt:{}, dist v v_alt: {}".format(item_index, torch.dist(u_weight, u_weight_alt), torch.dist(v_weight.t(), v_weight_alt)))
                #print("layer indeix: {}, dist u u_alt:{}, dist v v_alt: {}".format(item_index, torch.equal(u_weight, u_weight_alt), torch.equal(v_weight.t(), v_weight_alt)))
                #print("dist param impl: {}, dist param impl alt: {}".format(torch.dist(param_reshaped, torch.mm(u_weight_alt, v_weight_alt)), torch.dist(param_reshaped, torch.mm(u_weight, v_weight.t()))))
                #print("dist: {}".format(torch.dist(torch.mm(u_weight_alt, v_weight_alt), torch.mm(u_weight, v_weight.t()))))


                u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]

                u_weight_sliced_shape, v_weight_sliced_shape = u_weight_sliced.size(), v_weight_sliced.size()

                model_weight_v = u_weight_sliced.view(u_weight_sliced_shape[0],
                                                      u_weight_sliced_shape[1], 1, 1)
                
                model_weight_u = v_weight_sliced.t().view(v_weight_sliced_shape[1], 
                                                          param.size()[1], 
                                                          param.size()[2], 
                                                          param.size()[3])

                reconstructed_aggregator.append(model_weight_u)
                reconstructed_aggregator.append(model_weight_v)    
            else:
                reconstructed_aggregator.append(param)
                
                
        model_counter = 0
        reload_state_dict = {}
        for item_index, (param_name, param) in enumerate(low_rank_model.state_dict().items()):
            #print("#### {}, {}, recons agg: {}， param: {}".format(item_index, param_name, 
            #                                                                        reconstructed_aggregator[model_counter].size(),
            #                                                                       param.size()))
            assert (reconstructed_aggregator[model_counter].size() == param.size())
            reload_state_dict[param_name] = reconstructed_aggregator[model_counter]
            model_counter += 1

    elif args.arch == "resnet18":
        for item_index, (param_name, param) in enumerate(model.state_dict().items()):
            if len(param.size()) == 4 and item_index not in range(0, 13) and ".shortcut." not in param_name:
                # resize --> svd --> two layer
                param_reshaped = param.view(param.size()[0], -1)
                rank = min(param_reshaped.size()[0], param_reshaped.size()[1])
                u, s, v = torch.svd(param_reshaped)

                sliced_rank = int(rank/rank_factor)
                u_weight = u * torch.sqrt(s) # alternative implementation: u_weight_alt = torch.mm(u, torch.diag(torch.sqrt(s)))
                v_weight = torch.sqrt(s) * v # alternative implementation: v_weight_alt = torch.mm(torch.diag(torch.sqrt(s)), v.t())
                # sanity check: print("dist u u_alt:{}, dist v v_alt: {}".format(torch.dist(u_weight, u_weight_alt), torch.dist(v_weight.t(), v_weight_alt))

                u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]

                u_weight_sliced_shape, v_weight_sliced_shape = u_weight_sliced.size(), v_weight_sliced.size()

                model_weight_v = u_weight_sliced.view(u_weight_sliced_shape[0],
                                                      u_weight_sliced_shape[1], 1, 1)
                
                model_weight_u = v_weight_sliced.t().view(v_weight_sliced_shape[1], 
                                                          param.size()[1], 
                                                          param.size()[2], 
                                                          param.size()[3])

                reconstructed_aggregator.append(model_weight_u)
                reconstructed_aggregator.append(model_weight_v)           
            else:
                reconstructed_aggregator.append(param)
                
        model_counter = 0
        reload_state_dict = {}
        for item_index, (param_name, param) in enumerate(low_rank_model.state_dict().items()):
            #print("#### {}, {}, recons agg: {}， param: {}".format(item_index, param_name, 
            #                                                                        reconstructed_aggregator[model_counter].size(),
            #                                                                       param.size()))
            assert (reconstructed_aggregator[model_counter].size() == param.size())
            reload_state_dict[param_name] = reconstructed_aggregator[model_counter]
            model_counter += 1
    else:
        raise NotImplementedError("Unsupported model arch ...")
    
    low_rank_model.load_state_dict(reload_state_dict)
    return low_rank_model



def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    epoch_total_time = 0.0
    epoch_comp_time = 0.0
    epoch_data_time = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        comp_start = torch.cuda.Event(enable_timing=True)
        comp_end = torch.cuda.Event(enable_timing=True)

        data_start = torch.cuda.Event(enable_timing=True)
        data_end = torch.cuda.Event(enable_timing=True)

        iter_start.record()
        data_start.record()
        data, target = data.to(device), target.to(device)
        data_end.record()
        torch.cuda.synchronize()
        iter_data_dur = float(data_start.elapsed_time(data_end))/1000.0
        epoch_data_time += iter_data_dur

        optimizer.zero_grad()

        comp_start.record() # only contain comp time
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() 

        comp_end.record() # only contain comp time
        torch.cuda.synchronize()
        iter_comp_dur = float(comp_start.elapsed_time(comp_end))/1000.0
        epoch_comp_time += iter_comp_dur

        if batch_idx % 40 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))

        iter_end.record()
        torch.cuda.synchronize()
        iter_total_dur = float(iter_start.elapsed_time(iter_end))/1000.0
        epoch_total_time += iter_total_dur
        
    return {"total_time":epoch_total_time,
            "comp_time":epoch_comp_time,
            "data_time":epoch_data_time}


def validate(test_loader, model, criterion, epoch, args, device):
    global best_acc

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output), target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    assert total == len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    test_loss /= len(test_loader.dataset)
    logger.info('\nEpoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch, 
        test_loss, correct, len(test_loader.dataset),
        acc))

    if not args.evaluate:
        if acc > best_acc:
            logger.info('###### Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}_best_lth.pth'.format(args.arch))
            best_acc = acc


def seed(seed):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seeded everything")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
    parser.add_argument('--mode', type=str, default='vanilla',
                    help='use full rank or low rank models')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=300, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--full-rank-warmup', type=bool_string, default=True,
                            help='if or not to use full-rank warmup')
    parser.add_argument('--fr-warmup-epoch', type=int, default=15,
                            help='number of full rank epochs to use')
    parser.add_argument('-rf', '--rank-factor', default=4, type=int,
                    metavar='N', help='the rank factor that is going to use in the low rank models')
    parser.add_argument('-re', '--resume', default=False, type=bool_string,
                    help='wether or not to resume from a checkpoint.')
    parser.add_argument('-eva', '--evaluate', type=bool_string, default=False,
                    help='wether or not to evaluate the model after loading the checkpoint.')
    parser.add_argument('-cp', '--ckpt_path', type=str, default="./checkpoint/vgg19_best.pth",
                    help='path to the checkpoint to resume.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Benchmarking over device: {}".format(device))
    logger.info("#### Args: {}".format(args))

    # let's enable cudnn benchmark
    seed(seed=args.seed)

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(
                            Variable(x.unsqueeze(0), requires_grad=False),
                            (4,4,4,4),mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    # data prep for test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    # load training and test set here:
    training_set = datasets.CIFAR10(root='./cifar10_data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                              num_workers=4,
                                              shuffle=True,
                                              pin_memory=True)
    testset = datasets.CIFAR10(root='./cifar10_data', train=False,
                                           download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             num_workers=4,
                                             shuffle=False,
                                             pin_memory=True)


    if args.arch == "resnet18":
        # if args.mode == "vanilla":
        #     model = ResNet18().to(device)
        # elif args.mode == "lowrank":
        #     model = LowrankResNet18().to(device)
        #     vanilla_model = ResNet18().to(device)
        # else:
        raise NotImplementedError("unsupported mode ...")
    elif args.arch == "vgg19":
        if args.mode == "vanilla":
            model = VGG19LTH().to(device)
        elif args.mode == "lowrank":
            model = LowrankVGG19LTH().to(device)
            vanilla_model = VGG19LTH().to(device)
        else:
            raise NotImplementedError("unsupported mode ...")

    with torch.cuda.device(0):
        lowrank_macs, lowrank_params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
        vanilla_macs, vanilla_params = get_model_complexity_info(vanilla_model, (3, 32, 32), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
    logger.info("============> Lowrank Model info: {}, num params: {}, Macs: {}".format(model, param_counter(model), lowrank_macs))
    logger.info("============> Vanilla Model info: {}, num params: {}, Macs: {}".format(vanilla_model, param_counter(vanilla_model), vanilla_macs))

    criterion = nn.CrossEntropyLoss()

    if args.resume:
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        if args.evaluate:
            validate(
                     test_loader=test_loader,
                     model=model, 
                     criterion=criterion, 
                     epoch=start_epoch,
                     args=args,
                     device=device)
            exit()


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    vanilla_optimizer = torch.optim.SGD(vanilla_model.parameters(), args.lr,
                                        momentum=args.momentum, weight_decay=1e-4)
    
    # switching off the weight decay for batch norm layers
    #parameters = add_weight_decay(model, 0.0001)
    #weight_decay = 0.
    # optimizer = torch.optim.SGD(parameters, args.lr,
    #                             momentum=args.momentum,
    #                             #weight_decay=args.weight_decay)
    #                             weight_decay=weight_decay)
    init_lr = args.lr

    epoch_norm = norm_calculator(model)
    logger.info("###### Norm of the Model in Epoch: {}, is: {}".format(0, epoch_norm))

    for epoch in range(0, args.epochs):
        # adjusting lr schedule
        if epoch < 80:
            for group in optimizer.param_groups:
                group['lr'] = init_lr

            for group in vanilla_optimizer.param_groups:
                group['lr'] = init_lr
        elif (epoch >= 80 and epoch < 120):
            for group in optimizer.param_groups:
                group['lr'] = init_lr/10.0

            for group in vanilla_optimizer.param_groups:
                group['lr'] = init_lr/10.0
        elif epoch >= 120:
            for group in optimizer.param_groups:
                group['lr'] = init_lr/100.0

            for group in vanilla_optimizer.param_groups:
                group['lr'] = init_lr/100.0

        for group in optimizer.param_groups:
            logger.info("### Epoch: {}, Current effective lr: {}".format(epoch, group['lr']))

        if args.full_rank_warmup and epoch in range(args.fr_warmup_epoch):
            logger.info("Epoch: {}, Warmuping ...".format(epoch))

            epoch_time = train(train_loader, vanilla_model, criterion, vanilla_optimizer, epoch, device=device)
        elif args.full_rank_warmup and epoch == args.fr_warmup_epoch:
            logger.info("Epoch: {}, swtiching to low rank model ...".format(epoch))

            decompose_start = torch.cuda.Event(enable_timing=True)
            decompose_end = torch.cuda.Event(enable_timing=True)

            decompose_start.record()
            model = decompose_weights(model=vanilla_model, 
                              low_rank_model=model, 
                              rank_factor=args.rank_factor,
                              args=args)
            decompose_end.record()
            torch.cuda.synchronize()
            decompose_dur = float(decompose_start.elapsed_time(decompose_end))/1000.0
            logger.info("#### Cost for decomposing the weights: {} ....".format(decompose_dur))


            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
            #optimizer = optim.SGD(model.parameters(), lr=(args.lr/2), momentum=args.momentum, weight_decay=1e-4)
            #init_lr = args.lr/2
            epoch_time = train(train_loader, model, criterion, optimizer, epoch, device=device)
        else:
            logger.info("Epoch: {}, low rank training ...".format(epoch))
            epoch_time = train(train_loader, model, criterion, optimizer, epoch, device=device)

        logger.info("####### Time Cost for Epoch: {} ===> Total: {}, Comp: {}, Data: {}".format(
                                            epoch, epoch_time["total_time"],
                                            epoch_time["comp_time"],
                                            epoch_time["data_time"]))

        # eval
        if args.full_rank_warmup and epoch in range(args.fr_warmup_epoch):
            # validate(test_loader, model, criterion, epoch, device)
            validate(
                     test_loader=test_loader,
                     model=vanilla_model, 
                     criterion=criterion, 
                     epoch=epoch,
                     args=args,
                     device=device)
        else:
            validate(
                     test_loader=test_loader,
                     model=model, 
                     criterion=criterion, 
                     epoch=epoch,
                     args=args,
                     device=device)


        epoch_norm = norm_calculator(model)
        logger.info("###### Norm of the Model in Epoch: {}, is: {}".format(epoch, epoch_norm))

    # we save the final model for future use
    #with open("trained_model_resnet18", "wb") as f_:
    #    torch.save(model.state_dict(), f_)

if __name__ == '__main__':
    main()