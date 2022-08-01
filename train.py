import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
import glob
import matplotlib

from UNet import *
from UNet import TrainableUNet

#Load data 
files_d = glob.glob('./MultiRay/density_one_*.h5')
files_x = glob.glob('./MultiRay/xi_one_*.h5')
density = []
xi = []
files = []
for i,file in enumerate(files_d):
    try:
        f = ''.join(file[23:])
        d = h5py.File('./MultiRay/density_one_'+f)['dataset_1'][:].astype(np.float32)
        x = h5py.File('./MultiRay/xi_one_'+f)['dataset_1'][:].astype(np.float32)

        files.append(f)
      
        density.append(d)
        density.append(np.swapaxes(d, 0,1))
        density.append(np.swapaxes(d, 1,2))
        density.append(np.swapaxes(d, 0,2))
        xi.append(x)
        xi.append(np.swapaxes(x, 0,1))
        xi.append(np.swapaxes(x, 1,2))
        xi.append(np.swapaxes(x, 0,2))
    except:
        pass


files_d = glob.glob('./MultiRay/data_multiRay/density_one_*.h5', )
files_x = glob.glob('./MultiRay/data_multiRay/xi_one_*.h5')
for i,file in enumerate(files_d):
  try: 
    f = ''.join(file[40:])
    d = h5py.File('./MultiRay/data_multiRay/density_one_'+f)['dataset_1'][:].astype(np.float32)
    x = np.swapaxes(h5py.File('./MultiRay/data_multiRay/xi_one_'+f)['dataset_1'][:].astype(np.float32), 0, 1)
    files.append(f)

    density.append(d)
    density.append(np.swapaxes(d, 0,1))
    density.append(np.swapaxes(d, 1,2))
    density.append(np.swapaxes(d, 0,2))
    xi.append(x)
    xi.append(np.swapaxes(x, 0,1))
    xi.append(np.swapaxes(x, 1,2))
    xi.append(np.swapaxes(x, 0,2))    
  except:
    pass

density = np.array(density)
xi = np.array(xi)
half = int(len(density[0,:])/2)

in_chan = 1
out_chan = 1

X = torch.tensor(-(density-np.mean(density))/(np.std(density)))
x_i = xi[:,20:-20,20:-20,20:-20]
y = torch.tensor((x_i-np.mean(x_i))/(np.std(x_i)))

from torch.utils.data import DataLoader, TensorDataset

# Split into train/test datasets and loaders:
train_size = int(0.9*len(X))
X2 = X[:,None]
y2 = y[:,None]
train_dataset = TensorDataset(X2[:train_size], y2[:train_size])
test_dataset = TensorDataset(X2[train_size:], y2[train_size:])

train_loader = DataLoader(train_dataset, batch_size=1,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1,
                         shuffle=False, num_workers=2)

model = TrainableUNet(n_in_features=in_chan, n_out_features=out_chan, hidden=32)

trainer = pl.Trainer(
    gpus=1,
    max_steps=1_000,
    benchmark=True,  # torch.backends.cudnn.benchmark = True
    auto_lr_find=False,  # Automatically tune learning rate
    auto_scale_batch_size=False,  # Scale batch size to max allowable by GPU memory
    gradient_clip_val=1.0,  # Clip gradients to avoid parameters exploding
    log_every_n_steps=1,
)

trainer.fit(model, train_loader, test_loader)

