from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os

from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset

class color_dataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(path,max_readers=32,readonly=True,lock=False,readahead=False,meminit=False,)
        #self.env = lmdb.open(path, map_size=1099511627776,max_readers=32, readonly=True)

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            #self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            self.length = 300000
            #self.length = txn.stat()['entries']

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img