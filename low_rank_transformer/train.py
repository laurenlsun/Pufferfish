'''
This script handles the training process.
'''
import argparse
import math
import time
import os
import dill as pickle
from tqdm import tqdm
import logging
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import Transformer, LowRankTransformer
from transformer.Optim import ScheduledOptim


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# "credit goes to Yu-Hsiang, modified by Hongyi Wang"
__author__ = "Yu-Hsiang Huang"

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, gold, opt.trg_pad_idx, smoothing=smoothing) 
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(vanilla_model, lowrank_model, training_data, validation_data, vanilla_optimizer, device, opt):
    ''' Start training '''

    log_train_file, log_valid_file = None, None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        logger.info('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, loss, accu, start_time):
        logger.info('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=math.exp(min(loss, 100)),
                  accu=100*accu, elapse=(time.time()-start_time)/60))

    #valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        logger.info('[ Epoch {} ]'.format(epoch_i))

        start = time.time()
        if epoch_i in range(opt.fr_warmup_epoch):
            logger.info("### Warming up Training, epoch: {}".format(epoch_i))
            train_loss, train_accu = train_epoch(
                vanilla_model, training_data, vanilla_optimizer, opt, device, smoothing=opt.label_smoothing)
        elif epoch_i == opt.fr_warmup_epoch:
            logger.info("### Switching to low-rank Training, epoch: {}".format(epoch_i))

            decompose_start = torch.cuda.Event(enable_timing=True)
            decompose_end = torch.cuda.Event(enable_timing=True)

            decompose_start.record()
            lowrank_model = decompose_vanilla_model(vanilla_model=vanilla_model, 
                                                    low_rank_model=lowrank_model,
                                                    rank_ratio=0.25)
            decompose_end.record()
            torch.cuda.synchronize()
            decompose_dur = float(decompose_start.elapsed_time(decompose_end))/1000.0
            logger.info("#### Cost for decomposing the weights: {} ....".format(decompose_dur))

            lowrank_optimizer = ScheduledOptim(
               optim.Adam(lowrank_model.parameters(), betas=(0.9, 0.98), eps=1e-09),
               2.0, opt.d_model, opt.n_warmup_steps)
            train_loss, train_accu = train_epoch(
                lowrank_model, training_data, lowrank_optimizer, opt, device, smoothing=opt.label_smoothing)
        else:
            logger.info("### Low-rank Training, epoch: {}".format(epoch_i))
            train_loss, train_accu = train_epoch(
                lowrank_model, training_data, lowrank_optimizer, opt, device, smoothing=opt.label_smoothing)            

        print_performances('Training', train_loss, train_accu, start)

        start = time.time()
        if epoch_i in range(opt.fr_warmup_epoch):
            valid_loss, valid_accu = eval_epoch(vanilla_model, validation_data, device, opt)
        else:
            valid_loss, valid_accu = eval_epoch(lowrank_model, validation_data, device, opt)
        print_performances('Validation', valid_loss, valid_accu, start)

        valid_losses += [valid_loss]

        if opt.fr_warmup_epoch >= opt.epoch: # vanilla 
            checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': vanilla_model.state_dict()}
        else:
            checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': lowrank_model.state_dict()}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                if opt.fr_warmup_epoch >= opt.epoch: # vanilla
                    model_name = opt.save_model + '_vanilla_seed{}.chkpt'.format(opt.seed)
                else:
                    model_name = opt.save_model + '_pufferfish_seed{}.chkpt'.format(opt.seed)

                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    logger.info('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))



def decompose_vanilla_model(vanilla_model, low_rank_model, rank_ratio=0.25):
    collected_weights = []
    for p_index, (name, param) in enumerate(vanilla_model.state_dict().items()):
        if len(param.size()) == 2 and p_index not in range(0, 14) and p_index not in range(76, 96) and p_index != 188:
            rank = min(param.size()[0], param.size()[1])
            sliced_rank = int(rank * rank_ratio)
            u, s, v = torch.svd(param)
            u_weight = u * torch.sqrt(s)
            v_weight = torch.sqrt(s) * v
            u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]
            #collected_weights.append(u_weight_sliced)
            collected_weights.append(v_weight_sliced.t())
            collected_weights.append(u_weight_sliced)
        else:
            collected_weights.append(param)
            
    #for cw_index, cw in enumerate(collected_weights):
    #     print("cw_index: {}, cw: {}".format(cw_index, cw.size()))
         
    reconstructed_state_dict = {}
    model_counter = 0
    for p_index, (name, param) in enumerate(low_rank_model.state_dict().items()):
        #print("p_index: {}, name: {}, param size: {}, collected weight size: {}".format(p_index,
        #                                                                                name,
        #                                                                                param.size(), collected_weights[model_counter].size()))
        assert param.size() == collected_weights[model_counter].size()
        reconstructed_state_dict[name] = collected_weights[model_counter]
        model_counter += 1
    low_rank_model.load_state_dict(reconstructed_state_dict)
    return low_rank_model


def num_params_counter(model):
    num_elems = 0
    for p_index, (p_name, p) in enumerate(model.named_parameters()):
        num_elems += p.numel()
    return num_elems


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
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)   # bpe encoded data
    parser.add_argument('-val_path', default=None)     # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    parser.add_argument('-seed', type=int, default=0, 
                        help='the random seed to use for the experiment.')

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-fr_warmup_epoch', type=int, default=150)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    if not opt.log and not opt.save_model:
        logger.info('No experiment result will be saved.')
        raise

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        logger.info('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')
    seed(seed=opt.seed)

    #========= Loading Dataset =========#

    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise

    logger.info(opt)

    vanilla_transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    lowrank_transformer = LowRankTransformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)
    logger.info("Full rank Transformer: Number of Parameters: {}".format(num_params_counter(vanilla_transformer)))
    logger.info("Low rank Transformer: Number of Parameters: {}".format(num_params_counter(lowrank_transformer)))


    # optimizer
    vanilla_optimizer = ScheduledOptim(
        optim.Adam(vanilla_transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, opt.d_model, opt.n_warmup_steps)
    #lowrank_optimizer = ScheduledOptim(
    #    optim.Adam(lowrank_transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
    #    2.0, opt.d_model, opt.n_warmup_steps)

    train(vanilla_transformer, lowrank_transformer, training_data, validation_data, vanilla_optimizer, device, opt)


def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    MIN_FREQ = 2
    if not opt.embs_share_weight:
        raise

    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = data['settings'].max_len
    field = data['vocab']
    fields = (field, field)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train = TranslationDataset(
        fields=fields,
        path=opt.train_path, 
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path, 
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    return train_iterator, val_iterator


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    print("@@@@ vocab size: {}".format(opt.src_vocab_size))

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    print("@@@@ train set examples: {}".format(len(train.examples)))
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()
