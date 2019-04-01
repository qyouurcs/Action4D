import argparse
import torch
import pickle
import torch.nn as nn
import numpy as np
import os
import glob
import pdb


from data_loader import SeqVolumeDataset

from model import ActionNet

from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
import logging
#from colorlog import ColoredFormatter
import horovod.torch as hvd

from loss import ContrastiveLoss
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name, average = False)
    return avg_tensor.item()


def main(args):

    #if not os.path.exists(args.model_path):
    #    os.makedirs(args.model_path)

    hvd.init()
    logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                                    datefmt='%m-%d %H:%M',
                                                    filename='{0}/training.log'.format(args.model_path),
                                                    filemode='a')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter( "%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    data_loader_val = None
    data_set_training = SeqVolumeDataset(args.data_dir, args.list_fn, args.seq_len, w = 61, h = 61, d= 2100//25 + 1)
    data_sampler = torch.utils.data.distributed.DistributedSampler(data_set_training, num_replicas=hvd.size(), rank=hvd.rank())
    data_loader = torch.utils.data.DataLoader(
        data_set_training, batch_size=args.batch_size, shuffle=(data_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=data_sampler)

    if args.list_fn_val:
        data_set_val = SeqVolumeDataset(args.data_dir_val, args.list_fn_val, args.seq_len, w = 61, h = 61, d= 2100//25 + 1)
        data_sampler_val = torch.utils.data.distributed.DistributedSampler(data_set_val, num_replicas=hvd.size(), rank=hvd.rank())
        data_loader_val = torch.utils.data.DataLoader(data_set_val, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory = True, sampler = data_sampler_val)

    net = ActionNet(args.hidden_size, args.class_num)

    print(net) 

    logging.info('HVD size %d, HVD rank %d', hvd.size(), hvd.rank())
    if torch.cuda.is_available():
        net.cuda()

    net = nn.DataParallel(net)
    if args.model_fn:
        logging.info('Loading from %s', args.model_fn)
        net.load_state_dict( torch.load(args.model_fn) )

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_gc = ContrastiveLoss(0.5)
    params = net.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())
    hvd.broadcast_parameters(net.state_dict(), root_rank=0)

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        data_sampler.set_epoch(epoch)
        net.train()
        for i, (data, lbls) in enumerate(data_loader):
            # Set mini-batch dataset
            data = Variable(data)
            lbls = Variable(lbls)
            if torch.cuda.is_available():
                data = data.cuda()
                lbls = lbls.cuda()
            if data.size(0) !=  args.batch_size:
                continue
            net.zero_grad()
            cur_batch_size = data.size(0)
            outputs, hs = net(data)

            lbls = lbls.expand(-1, outputs.size(1)).contiguous()
            outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
            lbls = lbls.view(-1)
            loss_gc = criterion_gc(hs)
            loss_c = criterion(outputs, lbls)
            loss = loss_gc + loss_c
            loss.backward()
            accuracy = (lbls.data == outputs.data.max(dim = 1)[1]).sum().item() * 1.0 / lbls.size(0)
            optimizer.step()

            if i % args.log_step == 0:
                logging.info('Epoch [%d/%d], Step [%d/%d], Rank[%d], Loss_gc: %.4f, Loss_c: %.4f, Loss: %.4f, Accuracy: %.4f'
                      ,epoch, args.num_epochs, i, total_step, hvd.rank(), loss_gc.data.item(), loss_c.data.item(), loss.data.item(), accuracy)
                
            if i == 0:
                os.system('nvidia-smi')

        if hvd.rank() == 0:
            torch.save(net.state_dict(), 
                os.path.join(args.model_path, 
                   'action-net-%d.pkl' %(epoch+1)))
        # Now testing.
        if data_loader_val:
            net.eval()
            val_total = 0
            val_correct = 0
            with torch.no_grad():
                for i, (data, lbls) in enumerate(data_loader_val):
                    if True:
                        data = Variable(data)
                        lbls = Variable(lbls)
                        if torch.cuda.is_available():
                            data = data.cuda()
                            lbls = lbls.cuda()

                        cur_batch_size = data.size(0)
                        if cur_batch_size != args.batch_size:
                            continue
                        outputs, hs = net(data)
                        #lbls = lbls.unsqueeze(1)
                        lbls = lbls.expand(-1, outputs.size(1)).contiguous()
                        outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
                        lbls = lbls.view(-1)
                        loss_c = criterion(outputs, lbls)
                        loss_gc = criterion_gc(hs)
                        loss = loss_c + loss_gc

                        correct = (lbls.data == outputs.data.max(dim = 1)[1]).sum().item()
                        accuracy = correct * 1.0 / lbls.size(0)
                        val_correct += correct
                        val_total += lbls.size(0)

                        # Print log info
                        if i % args.log_step == 0:
                            logging.info('Testing Epoch [%d/%d], Step [%d/%d], Rank[%d], Loss_gc: %.4f, Loss_c: %.4f, Loss: %.4f, Accuracy: %.4f'
                                  ,epoch, args.num_epochs, i, len(data_loader_val), hvd.rank(), loss_gc.item(), loss_c.item(), loss.item(), accuracy)


            val_t = metric_average(val_total, 'sum_total')
            val_c = metric_average(val_correct, 'sum_correct')
            if hvd.rank() == 0:
                logging.info('Testing Epoch [%d/%d], Val Accuracy: %.4f', epoch, args.num_epochs, val_c * 1.0 / val_t)
            #logging.info('Testing Epoch [%d/%d], Val Accuracy: %.4f', epoch, args.num_epochs, val_correct * 1.0 / val_total)

    if hvd.rank() == 0:
        torch.save(net.state_dict(), 
            os.path.join(args.model_path, 
               'action-net-%d.pkl' %(epoch+1)))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--model_fn', type = str, default = '')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_dir_val', type=str)
    parser.add_argument('--list_fn', type=str,
                        help='list of the video clips')
    parser.add_argument('--list_fn_val', type=str,
                        help='list of the video clips')
    parser.add_argument('--log_step', type=int , default=100,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=100,
                        help='step size for saving trained models')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--class_num', type=int , default=17,
                        help='number of class')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='number of class')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    args = parser.parse_args()
    print(args)
    main(args)
