import numpy as np
import torch
import h5py
import os

from torch.utils.data import Dataset
from torchvision import transforms, utils

class Table(Dataset):
    def __init__(self, h5_file, is_train=False):
        self.infile = h5_file
        f = h5py.File(self.infile, mode = 'r')

        assert 'cprof' in f.keys()
        grp_scat = f['cprof']
        
        assert 'table' in grp_scat.keys()        
        self.dataset = np.array(grp_scat['table']).squeeze()
        
        #event normalize for each momentum
        #w = self.dataset.sum(-1).sum(-1)
        #self.weight = np.ones(self.dataset.shape)/w.reshape(-1,1,1)

        self.weight = np.ones(self.dataset.shape)

        self.binning = [np.array(grp_scat['binning0']),
                        np.array(grp_scat['binning1']),
                        np.array(grp_scat['binning2'])]
        
        self.tableshape = self.dataset.shape
        
        # normalize axis scale (mom, cos, track) as [-1,1]
        normrange= np.array([[50, 1000],[-1,1],[0,600]])
        mean=[]
        width=[]
        for axis in range(3):
            m = (normrange[axis,-1] + normrange[axis,0])/2
            w = normrange[axis,-1] - m
            mean.append(m)
            width.append(w)
        self.mean = np.array(mean)
        self.width = np.array(width)
        
        f.close()

        
    def __len__(self): 
        sum_dim = 1
        for dim in self.tableshape:
            sum_dim *= dim
        return sum_dim
       
    def __getitem__(self, index):
        datasets = self.dataset.reshape(-1,1)[index,:]
        
        weight = self.weight.reshape(-1,1)[index,:]
        
        assert self.tableshape is not None

        # transform global sample number (index i) to indices of array component (I,J,K)
        unrav_indices = np.array(np.unravel_index(index, self.tableshape))#.T

        # normalize
        indices = []
        for axis in range(3):
            binvalue = self.binning[axis][unrav_indices[axis]]
            indices.append( self.normalize(axis, binvalue) )
        indices = np.array(indices)
        
        #[0,...,n] -> [-1,...,1]
        #unrav_indices = unrav_indices*2./(np.array(self.table_shape)-1) - 1.
        
        return indices.astype(np.float32), datasets, weight

    def normalize(self, axis, value):
        return (value-self.mean[axis])/self.width[axis]
