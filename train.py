import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
import glob
import matplotlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

import Network

# Load data

# Load data from /Data/80_160_0_80_0_80
files_d = glob.glob('./Data/80_160_0_80_0_80/density_one_*.h5')
files_x = glob.glob('./Data/80_160_0_80_0_80/xi_one_*.h5')
density = []
xi = []
files = []
for i,file in enumerate(files_d):
    try:
        f = ''.join(file[36:])

        d = h5py.File('./Data/80_160_0_80_0_80/density_one_'+f, 'r')['dataset_1'][:].astype(np.float32)
        x = h5py.File('./Data/80_160_0_80_0_80/xi_one_'+f, 'r')['dataset_1'][:].astype(np.float32)
        
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


# Load data from /Data/0_80_0_80_0_80
files_d = glob.glob('./Data/0_80_0_80_0_80/density_one_*.h5', )
files_x = glob.glob('./MultiRay/data_multiRay/xi_one_*.h5')
for i,file in enumerate(files_d):
  try: 
    f = ''.join(file[36:])

    d = h5py.File('./Data/0_80_0_80_0_80/density_one_'+f, 'r')['dataset_1'][:].astype(np.float32)
    x = np.swapaxes(h5py.File('./Data/0_80_0_80_0_80/xi_one_'+f, 'r')['dataset_1'][:].astype(np.float32), 0, 1)

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


# Save data in numpy array
density = np.array(density)
xi = np.array(xi)[:,20:-20,20:-20,20:-20] #output smaler shape

half = int(len(density[0,:])/2) #half box size

# Define training variables
in_chan = 1
out_chan = 1

# Standerdize data
X = torch.tensor(-(density-np.mean(density))/(np.std(density)))
y = torch.tensor((xi-np.mean(xi))/(np.std(xi)))

# Bring them in correct shape
X2 = X[:,None]
y2 = y[:,None]


# Split into train/test datasets and loaders:
train_size = int(0.9*len(X))

train_dataset = TensorDataset(X2[:train_size], y2[:train_size])
test_dataset = TensorDataset(X2[train_size:], y2[train_size:])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# Define model
model = Network.TrainableUNet(n_in_features=in_chan, n_out_features=out_chan, hidden=32)

# Define trainer
trainer = pl.Trainer(
    gpus=1,
    max_steps=1_000,
    benchmark=True,  # torch.backends.cudnn.benchmark = True
    auto_lr_find=False,  # Automatically tune learning rate
    auto_scale_batch_size=False,  # Scale batch size to max allowable by GPU memory
    gradient_clip_val=1.0,  # Clip gradients to avoid parameters exploding
    log_every_n_steps=1,
)

# Train model
trainer.fit(model, train_loader, test_loader)

# Plot a test example
sample = train_size+6
pos = 10
test = model(X2[sample:sample+1])

matplotlib.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(1,3, figsize=(12,4))

d_im = ax[0].imshow(X[sample, 20:-20, 20:-20,20+pos], cmap=plt.get_cmap("viridis_r"))

vmin = np.min(y[sample, :, :, pos].detach().numpy())
vmax = np.max(y[sample, :, :, pos].detach().numpy())

x_im = ax[1].imshow(y[sample, :, :, pos], vmin=vmin, vmax=vmax)

ax[2].imshow(test[0, 0, :, :, pos].detach().numpy(), vmin=vmin, vmax=vmax)

ax[0].get_yaxis().set_visible(False)
ax[0].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[1].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].get_xaxis().set_visible(False)

ax[0].set_title("density")
ax[1].set_title("ray tracing")
ax[2].set_title("ml")

plt.colorbar(d_im, cax = plt.axes([0.04, 0.15, 0.03, 0.7]))
plt.colorbar(x_im, cax = plt.axes([0.93, 0.15, 0.03, 0.7]))
plt.colorbar(x_im, cax = plt.axes([0.93, 0.15, 0.03, 0.7]))

plt.savefig("./Final.pdf", bbox_inches='tight')