import argparse
import torch
import torch.nn as nn 
import numpy as np
import os
import glob
import pdb

import logging

from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from model import ActionNet
import scipy
import scipy.ndimage

import cv2

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'bold',
        'size': 'xx-large',
}

actions = [
        "None",
        "Drink",
        "Clapping",
        "Reading",
        "Phone call",
        "Interacting phone",
        "Bend",
        "Squad",
        "Wave",
        "Sitting",
        "Pointing to sth",
        "Lift/hold box",
        "Open drawer",
        "Pull/Push sth",
        "Eat from a plate",
        "Yarning /Stretch",
        "Kick"]

def main(args):

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    net = ActionNet(args.hidden_size, args.class_num)

    sf = nn.Softmax(dim = 1)
    net = nn.DataParallel(net)

    radius = (args.w - 1) // 2

    net.cuda()
    if args.model_fn:
        logging.info("Loading from %s", args.model_fn)
        net.load_state_dict(torch.load(args.model_fn))
    
    key = os.path.basename(args.test_dir).replace('/','_')

    with torch.no_grad():
        total_cnt = 0
        correct_cnt = 0
        net.eval()

        if args.test_dir[-1] == '/':
            args.test_dir = args.test_dir[0:-1]

        
        dict_imgs = {}

        fns = glob.glob(os.path.join(args.test_dir, '*.npz'))
        pid = os.path.splitext(os.path.basename(fns[0]))[0].split('-')[-1]
        max_frame = -1
        for fn in fns:
            frame = os.path.basename(fn)
            frame = int(frame.split('-')[-2])
            max_frame = max(max_frame, frame)

        cur_clip = []
        idx = 1
        wfid = open(os.path.join(args.save_dir, 'pred-v-' + key + '.txt'),'w')
        while idx < max_frame:
            if idx % 500 == 0:
                print (idx,':', len(fns))
            fn = os.path.join(args.test_dir, 'v--d-{0}-q-{1}-frame-{2}-{3}.npz'.format(30,25,idx, pid))
            
            if os.path.isfile(fn):
                d = np.load(fn)['data'].astype('float32')
                d = np.reshape(d, (args.w, args.h, args.d))
                cur_clip.append(d)
            else:
                cur_clip = []
            
            if len(cur_clip) == args.seq_len:

                ipt = np.stack(cur_clip, axis = 0)
                ipt[ipt > 0 ] = 1.0
                ipt *= 20

                ipt = ipt.astype('float32')

                ipt = torch.from_numpy(ipt)
                ipt = ipt.unsqueeze(0)
                
                ipt = Variable(ipt)

                if torch.cuda.is_available():
                    ipt = ipt.cuda()

                outputs = net(ipt)
                if len(outputs) == 2:
                    outputs = outputs[0]
                outputs = outputs.squeeze()
                outputs = sf(outputs)
                outputs = torch.sum(outputs, dim = 0)
                cur_clip.pop(0)
                _, lbl = torch.max(outputs.data, dim = 0)
                wfid.write(str(idx) + ' ' + str(lbl.item()) + '\n')

            idx += 1
        wfid.close()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        help='path for saving trained models')
    parser.add_argument('--model_fn', type=str,
                        help='path for saving trained models')
    parser.add_argument('--test_dir', type=str,
                        help='directory for resized images')
    parser.add_argument('--save_dir', type=str,
                        help='directory for resized images')
    parser.add_argument('--w', type=int , default=61)
    parser.add_argument('--h', type=int , default=61)
    parser.add_argument('--d', type=int , default=85)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--class_num', type=int , default=17,
                        help='number of class')

    parser.add_argument('--hidden_size', type=int , default=512,
                        help='number of class')

    args = parser.parse_args()
    print(args)
    main(args)
