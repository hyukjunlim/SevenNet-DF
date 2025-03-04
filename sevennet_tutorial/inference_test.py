import sys
sys.path.append('../../SevenNet')
import sevenn_dfs
import os.path

data_path = './data'
working_dir = os.getcwd() # save current path
assert os.path.exists(data_path) and os.path.exists(os.path.join(data_path, 'train'))

from sevenn_dfs.train.graph_dataset import SevenNetGraphDataset

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

from sevenn_dfs._const import DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG
from sevenn_dfs.model_build import build_E3_equivariant_model
import sevenn_dfs.util as util

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

from sevenn_dfs._const import DEFAULT_TRAINING_CONFIG
from sevenn_dfs.train.trainer import Trainer

# copy default training configuration
train_cfg = deepcopy(DEFAULT_TRAINING_CONFIG)

# set optimizer and scheduler for training.
train_cfg.update({
  'device': 'cuda',
  'optimizer': 'adam',
  'optim_param': {'lr': 0.01},
  'scheduler': 'linearlr',
  'scheduler_param': {'start_factor': 1.0, 'total_iters': 100, 'end_factor': 0.0001},
  # 'scheduler': 'exponentiallr',
  # 'scheduler_param': {'gamma': 0.99},
  'force_loss_weight': 0.2,
  'stress_loss_weight': 1e-4,
})

# load test model
import torch
import ase.io
from tqdm import tqdm
import time

for mode in ['0', 'df']:
  if mode == 'df':
    from sevenn_dfs.sevennet_calculator import SevenNetCalculator
    sevennet_calc = SevenNetCalculator(os.path.join(working_dir, 'checkpoint_best.pth'))
  elif mode == '0':
    from sevenn.sevennet_calculator import SevenNetCalculator
    sevennet_calc = SevenNetCalculator(model='7net-0', device='cuda')
  
  DFT_md_xyz = os.path.join(data_path, 'evaluation/test_md.extxyz')
  traj = ase.io.read(DFT_md_xyz, index=':')


  start = time.time()
  for _ in range(10):
    for atoms in tqdm(traj):
      atoms.calc = sevennet_calc
      
      atoms.get_potential_energy()
      atoms.get_forces()
      atoms.get_stress()
  end = time.time()
      
  print(f'time cost for mode {mode}: {end - start}')