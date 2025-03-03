import sys
sys.path.append('../../SevenNet')
import sevenn_df
import os.path

data_path = './data'
working_dir = os.getcwd() # save current path
assert os.path.exists(data_path) and os.path.exists(os.path.join(data_path, 'train'))

from sevenn_df.train.graph_dataset import SevenNetGraphDataset

dataset_prefix = os.path.join(data_path, 'train')
xyz_files = ['1200K.extxyz', '600K.extxyz']
dataset_files = [os.path.join(dataset_prefix, xyz) for xyz in xyz_files]

# Preprocess(build graphs) data before training. It will automatically saves processed graph to {root}/sevenn_data/train.pt, metadata + statistics as train.yaml
cutoff = 4.5  # cutoff radius for graph construction. You can change this value.
dataset = SevenNetGraphDataset(cutoff=cutoff, root=working_dir, files=dataset_files, processed_name='train.pt')

print(f'# graphs: {len(dataset)}')
print(f'# atoms (nodes): {dataset.natoms}')
print(dataset[0])

from torch_geometric.loader import DataLoader

# split the dataset into train & valid
num_dataset = len(dataset)
num_train = int(num_dataset * 0.95)
num_valid = num_dataset - num_train

dataset = dataset.shuffle()
train_dataset = dataset[:num_train]
valid_dataset = dataset[num_train:]

print(f'# graphs for training: {len(train_dataset)}')
print(f'# graphs for validation: {len(valid_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)

from copy import deepcopy

from sevenn_df._const import DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG
from sevenn_df.model_build import build_E3_equivariant_model
import sevenn_df.util as util

# copy default model configuration.
model_cfg = deepcopy(DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG)

# tune the channel and lmax parameters. You can experiment with different settings.
model_cfg.update({'channel': 16, 'lmax': 2})
# model_cfg.update({'cutoff': 5.0})

# tell models about element in universe
model_cfg.update(util.chemical_species_preprocess([], universal=True))

# tell model about statistics of dataset. kind of data standardization
train_shift = {'E': dataset.per_atom_energy_mean, 'F': dataset.force_mean, 'S': dataset.stress_mean}
train_scale = {'E': dataset.energy_rms, 'F': dataset.force_rms, 'S': dataset.stress_rms}
train_conv_denominator = dataset.avg_num_neigh
model_cfg.update({'shift': train_shift, 'scale': train_scale, 'conv_denominator': train_conv_denominator})
print(model_cfg)

model = build_E3_equivariant_model(model_cfg)
num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model) # model info
print(f'# model weights: {num_weights}')

from sevenn_df._const import DEFAULT_TRAINING_CONFIG
from sevenn_df.train.trainer import Trainer

# copy default training configuration
train_cfg = deepcopy(DEFAULT_TRAINING_CONFIG)

# set optimizer and scheduler for training.
train_cfg.update({
  'device': 'cuda',
  'optimizer': 'adam',
  'optim_param': {'lr': 0.05},
  # 'scheduler': 'linearlr',
  # 'scheduler_param': {'start_factor': 1.0, 'total_iters': 50, 'end_factor': 0.0001},
  'scheduler': 'exponentiallr',
  'scheduler_param': {'gamma': 0.98},
  'force_loss_weight': 0.2,
  'stress_loss_weight': 1e-4,
})

# Initialize trainer. It implements common rountines for training.
trainer = Trainer.from_config(model, train_cfg)
print(trainer.loss_functions)  # We have energy, force, stress loss function by defaults. With default 1.0, 0.1, and 1e-6 loss weight
print(trainer.optimizer)
print(trainer.scheduler)

from sevenn_df.error_recorder import ErrorRecorder

train_cfg.update({
  # List of tuple [Quantity name, metric name]
  # Supporting quantities: Energy, Force, Stress, Stress_GPa
  # Supporting metrics: RMSE, MAE, Loss
  # TotalLoss is special!
  'error_record': [
    ('Energy', 'RMSE'),
    ('Force', 'RMSE'),
    ('Stress', 'RMSE'),  # $We skip stress error cause it is too long to print, uncomment it if you want
    ('TotalLoss', 'None'),
  ]
})
train_recorder = ErrorRecorder.from_config(train_cfg)
valid_recorder = deepcopy(train_recorder)
for metric in train_recorder.metrics:
  print(metric)
  
from tqdm import tqdm

valid_best = float('inf')
total_epoch = 200    # you can increase this number for better performance.
pbar = tqdm(range(total_epoch))
config = model_cfg  # to save config used in this tutorial.
config.update(train_cfg)

# for epoch in pbar:
#   # trainer scans whole data from given loader, and updates error recorder with outputs.
#   trainer.run_one_epoch(train_loader, is_train=True, error_recorder=train_recorder)
#   trainer.run_one_epoch(valid_loader, is_train=False, error_recorder=valid_recorder)
#   trainer.scheduler_step(valid_best) 
#   train_err = train_recorder.epoch_forward()  # return averaged error over one epoch, then reset.
#   valid_err = valid_recorder.epoch_forward()

#   # for print. train_err is a dictionary of {metric name with unit: error}
#   err_str = 'Train: ' + '    '.join([f'{k}: {v:.3f}' for k, v in train_err.items()])
#   err_str += '// Valid: ' + '    '.join([f'{k}: {v:.3f}' for k, v in valid_err.items()])
#   pbar.set_description(err_str)

#   if valid_err['TotalLoss'] < valid_best:  # saves best checkpoint. by comparing validation set total loss
#     valid_best = valid_err['TotalLoss']
#     trainer.write_checkpoint(os.path.join(working_dir, 'checkpoint_best.pth'), config=config, epoch=epoch)

# load test model
import torch
import ase.io

from sevenn_df.sevennet_calculator import SevenNetCalculator

# Let's test our model by predicting DFT MD trajectory
# Instead of using other functions in SevenNet, we will use ASE calculator as an interface of our model
DFT_md_xyz = os.path.join(data_path, 'evaluation/test_md.extxyz')

# initialize calculator from checkpoint.
sevennet_calc = SevenNetCalculator(os.path.join(working_dir, 'experiment/checkpoint_best_base_full5.pth'))

# load DFT md trajectory
traj = ase.io.read(DFT_md_xyz, index=':')

import numpy as np
dft_energy = []
dft_forces = []
dft_stress = []

mlp_energy = []
mlp_forces = []
mlp_stress = []
to_kBar = 1602.1766208

for atoms in tqdm(traj):
  atoms.calc = sevennet_calc
  
  mlp_energy.append(atoms.get_potential_energy() / len(atoms) * 1000)  # as per atom energy
  mlp_forces.append(atoms.get_forces())
  mlp_stress.extend(-atoms.get_stress() * to_kBar)  # eV/Angstrom^3 to kBar unit

  dft_energy.append(atoms.info['DFT_energy'] / len(atoms) * 1000)
  dft_forces.append(atoms.arrays['DFT_forces'])
  dft_stress.append(-atoms.info['DFT_stress'] * to_kBar)

# flatten forces and stress for parity plot
mlp_forces = np.concatenate([mf.reshape(-1,) for mf in mlp_forces])
mlp_stress = np.concatenate([ms.reshape(-1,) for ms in mlp_stress])

dft_forces = np.concatenate([df.reshape(-1,) for df in dft_forces])
dft_stress = np.concatenate([ds.reshape(-1,) for ds in dft_stress])

import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

unit = {"energy": "meV/atom", "force": "eV/Å", "stress": "kbar"}

def density_colored_scatter_plot(dft_energy, nnp_energy, dft_force, nnp_force, dft_stress, nnp_stress):
    modes = ['energy', 'force', 'stress']
    # Calculate and print MAE for each quantity
    for mode, dft, nnp in zip(modes, [dft_energy, dft_force, dft_stress], [nnp_energy, nnp_force, nnp_stress]):
        mae = mean_absolute_error(dft, nnp)
        print(f'MAE for {mode}: {mae:.4f} {unit[mode]}', flush=True)

# Assuming you have the data variables like dft_energy, mlp_energy, etc.
density_colored_scatter_plot(dft_energy, mlp_energy, dft_forces, mlp_forces, dft_stress, mlp_stress)
