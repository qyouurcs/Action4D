import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import pdb
import numpy as np
from PIL import Image

from transformations import rotation_matrix, translation_matrix
import scipy
import scipy.ndimage
import scipy.sparse

import tarfile
from io import BytesIO


class SeqVolumeDataset(data.Dataset):
    def __init__(self, tar_fn, list_fn, seq_len, w = 61, h = 61, d = 85):
        self.tar_fn = tar_fn
        self.list_fn = list_fn
        self.seq_len = seq_len
        self.samples = self.load_list(list_fn)
        self.w = w
        self.h = h
        self.d = d
        self.rotation = True
        self.tar_fid = tarfile.open(self.tar_fn)
        self.name2member = {}
        for idx, member in enumerate(self.tar_fid.getmembers()):
            if idx % 3000 == 0:
                print(idx)
            if member.name.endswith("npz"):
                array_file = BytesIO()

                array_file.write(self.tar_fid.extractfile(member).read())
                array_file.seek(0)
                d = scipy.sparse.load_npz(array_file)
                self.name2member[member.name] = d


    def load_list(self, list_fn):
        samples = []
        with open(list_fn, 'r') as fid:
            for aline in fid:
                parts = aline.strip().split()
                samples.append( (parts[0], int(parts[1]), int(parts[2])))
        return samples
                
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        sample = self.samples[index]
        lbl = sample[2]
        vs = []
        if self.rotation:
            alpha = 2 * np.pi * np.random.rand()

        for i in range(self.seq_len):
            ffn = os.path.join(sample[0], '{:06d}'.format(sample[1] + i) + '.npz')
            d = self.name2member[ffn]
            d = d.toarray()
            data = np.reshape(d, (self.w, self.h, self.d)).astype('float32')
            data[data > 0 ] = 1.0
            data *= 20
	
            if self.rotation:
                c = np.meshgrid(np.arange(self.w), np.arange(self.h), np.arange(self.d))
                xyz = np.vstack([c[0].reshape(-1) - float(self.w) / 2,
                         c[1].reshape(-1) - float(self.h) / 2,
                         c[2].reshape(-1) - float(self.d) / 2,
                         np.ones((self.w, self.h, self.d)).reshape(-1) ] )

                mat = rotation_matrix(alpha, (0, 0, 1))
                t_xyz = np.dot(mat, xyz)

                x = t_xyz[0,:] + float(self.w) /2
                y = t_xyz[1,:] + float(self.h) /2
                z = t_xyz[2,:] + float(self.d) /2

                x = x.reshape((self.w, self.h, self.d))
                y = y.reshape((self.w, self.h, self.d))
                z = z.reshape((self.w, self.h, self.d))

                #n_xyz = [ x, y, z]
                # Need to swap the x and y axis.
                n_xyz = [ y, x, z]
                data = scipy.ndimage.map_coordinates(data, n_xyz,  order=1, mode='nearest')
            
            p = np.random.uniform(0.7,1)
            r_idx = np.random.randint(6)
            mask = np.random.binomial(1, p, data.shape)
            data = mask * data
            if r_idx > 0:
                data[:,:,0:-r_idx] = data[:,:,r_idx:]
                data[:,:,-r_idx+1:] =  0
            data = data.astype('float32')
            vs.append(torch.from_numpy(data))

        return torch.stack(vs, dim = 0), torch.LongTensor([lbl])

    def __len__(self):
        return len(self.samples)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, lbls = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    return images, torch.stack(lbls, 0).squeeze(1)

def get_loader(tar_fn, list_fn, seq_len, num_workers, batch_size, w = 61, h = 61, d = 85, shuffle = True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    video_ds = SeqVolumeDataset(tar_fn = tar_fn,
                       list_fn = list_fn, 
                       seq_len = seq_len,
                       w = w, 
                       h = h, 
                       d = d)
    
    data_loader = torch.utils.data.DataLoader(dataset=video_ds,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

if __name__ == '__main__':
    video_ds = SeqVolumeDataset(tar_fn = 'all_f16_old_f9_train.tar',
                       list_fn = 'all_f16_old_f9_train.lst',
                       seq_len = 10)
     
    for i in range(len(video_ds)):
        video_ds[i]
    pass
