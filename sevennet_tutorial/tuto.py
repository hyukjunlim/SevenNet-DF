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


# CONCAT_DATASET

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

# set optimizer and scheduler for training.X``
train_cfg.update({
  'device': 'cuda',
  'optimizer': 'adam',
  'optim_param': {'lr': 0.005},
  'scheduler': 'linearlr',
  'scheduler_param': {'start_factor': 1.0, 'total_iters': 50, 'end_factor': 0.0001},
  # 'scheduler': 'exponentiallr',
  # 'scheduler_param': {'gamma': 0.99},
  'force_loss_weight': 0.2,
  'stress_loss_weight': 5e-6,
})

# Initialize trainer. It implements common rountines for training.
trainer = Trainer.from_config(model, train_cfg)
print(trainer.loss_functions)  # We have energy, force, stress loss function by defaults. With default 1.0, 0.1, and 1e-6 loss weight
print(trainer.optimizer)
print(trainer.scheduler)

from sevenn_dfs.error_recorder import ErrorRecorder

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
total_epoch = 50    # you can increase this number for better performance.
pbar = tqdm(range(total_epoch))
config = model_cfg  # to save config used in this tutorial.
config.update(train_cfg)

for epoch in pbar:
  # trainer scans whole data from given loader, and updates error recorder with outputs.
  trainer.run_one_epoch(train_loader, is_train=True, error_recorder=train_recorder)
  trainer.run_one_epoch(valid_loader, is_train=False, error_recorder=valid_recorder)
  trainer.scheduler_step(valid_best) 
  train_err = train_recorder.epoch_forward()  # return averaged error over one epoch, then reset.
  valid_err = valid_recorder.epoch_forward()

  # for print. train_err is a dictionary of {metric name with unit: error}
  err_str = 'Train: ' + '    '.join([f'{k}: {v:.3f}' for k, v in train_err.items()])
  err_str += '// Valid: ' + '    '.join([f'{k}: {v:.3f}' for k, v in valid_err.items()])
  pbar.set_description(err_str)

  if valid_err['TotalLoss'] < valid_best:  # saves best checkpoint. by comparing validation set total loss
    valid_best = valid_err['TotalLoss']
    trainer.write_checkpoint(os.path.join(working_dir, 'checkpoint_best.pth'), config=config, epoch=epoch)

# load test model
import torch
import ase.io

from sevenn_dfs.sevennet_calculator import SevenNetCalculator

# Let's test our model by predicting DFT MD trajectory
# Instead of using other functions in SevenNet, we will use ASE calculator as an interface of our model
DFT_md_xyz = os.path.join(data_path, 'evaluation/test_md.extxyz')

# initialize calculator from checkpoint.
sevennet_calc = SevenNetCalculator(os.path.join(working_dir, 'checkpoint_best.pth'))

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
  
  mlp_energy.append(atoms.get_potential_energy() / len(atoms))  # as per atom energy
  mlp_forces.append(atoms.get_forces())
  mlp_stress.extend(-atoms.get_stress() * to_kBar)  # eV/Angstrom^3 to kBar unit

  dft_energy.append(atoms.info['DFT_energy'] / len(atoms))
  dft_forces.append(atoms.arrays['DFT_forces'])
  dft_stress.append(-atoms.info['DFT_stress'] * to_kBar)

# flatten forces and stress for parity plot
mlp_forces = np.concatenate([mf.reshape(-1,) for mf in mlp_forces])
mlp_stress = np.concatenate([ms.reshape(-1,) for ms in mlp_stress])

dft_forces = np.concatenate([df.reshape(-1,) for df in dft_forces])
dft_stress = np.concatenate([ds.reshape(-1,) for ds in dft_stress])

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# draw a parity plot of energy / force / stress
unit = {"energy": "eV/atom", "force": r"eV/$\rm{\AA}$", "stress": "kbar"}
def density_colored_scatter_plot(dft_energy, nnp_energy, dft_force, nnp_force, dft_stress, nnp_stress):
    modes = ['energy', 'force', 'stress']
    plt.figure(figsize=(18/2.54, 6/2.54))
    for num, (x, y) in enumerate(zip([dft_energy, dft_force, dft_stress], [nnp_energy, nnp_force, nnp_stress])):
        mode = modes[num]
        idx = np.random.choice(len(x), 1000) if len(x) > 1000 else list(range(len(x)))
        xsam = [x[i] for i in idx]
        ysam = [y[i] for i in idx]
        xy = np.vstack([x, y])
        xysam = np.vstack([xsam, ysam])
        zsam = gaussian_kde(xysam)

        z = zsam.pdf(xy)
        idx = z.argsort()

        x = [x[i] for i in idx]
        y = [y[i] for i in idx]
        z = [z[i] for i in idx]
        
        ax = plt.subplot(int(f'13{num+1}'))
        plt.scatter(x, y, c=z, s=4, cmap='plasma')

        mini = min(min(x), min(y))
        maxi = max(max(x), max(y))
        ran = (maxi-mini) / 20
        plt.plot([mini-ran, maxi+ran], [mini-ran, maxi+ran], color='grey', linestyle='dashed')
        plt.xlim(mini-ran, maxi+ran)
        plt.ylim(mini-ran, maxi+ran)

        plt.xlabel(f'DFT {mode} ({unit[mode]})')
        plt.ylabel(f'MLP {mode} ({unit[mode]})')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('./results/parity_plot.png')

density_colored_scatter_plot(dft_energy, mlp_energy, dft_forces, mlp_forces, dft_stress, mlp_stress)



import warnings
warnings.filterwarnings("ignore")

from ase.constraints import UnitCellFilter
from ase.optimize import LBFGS
from ase.calculators.singlepoint import SinglePointCalculator

# codes for drawing EOS curve

def atom_oneshot(atoms, calc):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    calc_results = {"energy": energy, "forces": forces, "stress": stress}
    calculator = SinglePointCalculator(atoms, **calc_results)
    atoms = calculator.get_atoms()

    return atoms

def atom_cell_relax(atoms, calc, logfile="-"):
    atoms.calc = calc
    cf = UnitCellFilter(atoms, hydrostatic_strain=True)
    opt = LBFGS(cf, logfile=logfile)
    opt.run(fmax=0.05, steps=1000)

    return atoms

def make_eos_structures(relaxed):
    relaxed_cell = relaxed.get_cell()
    relaxed_lat = relaxed_cell.lengths()[0]
    
    eos_structures = []
    for strain in np.linspace(-0.05, 0.05, 11):
        strained_lat = relaxed_lat * (1+strain)
        relaxed.set_cell([strained_lat]*3, scale_atoms=True)
        eos_structures.append(deepcopy(relaxed))

    return eos_structures

def get_eos_and_volume(eos_list):
    en_list = []
    vol_list = []
    for atoms in eos_list:
        en_list.append(atoms.get_potential_energy())
        vol_list.append(atoms.get_volume())
        
    rel_en_list = np.array(en_list) - min(en_list)

    return rel_en_list, vol_list
  
  
# calculate EOS curve
from ase.io import read, write
from sevenn_dfs.sevennet_calculator import SevenNetCalculator

# get relaxed structure
os.makedirs('eos', exist_ok=True)
atoms_list = read(os.path.join(data_path, 'evaluation/eos.extxyz'), ':')  # most stable structure idx
most_stable_idx = np.argmin([at.get_potential_energy() for at in atoms_list])
print(f"(DFT) potential_energy (eV/atom): {atoms_list[most_stable_idx].get_potential_energy() / len(atoms_list[0])}")
atoms = atoms_list[most_stable_idx]

log_path = './eos/seven_relax_log.txt'
print("Relax with from-scratch potential...")
relaxed = atom_cell_relax(atoms, sevennet_calc, log_path)
print(f"(From scratch) potential_energy (eV/atom): {relaxed.get_potential_energy() / len(relaxed)}")

# make EOS structures and calculate EOS curve
eos_structures = make_eos_structures(relaxed)
eos_oneshot = []
for structure in eos_structures:
    eos_oneshot.append(atom_oneshot(structure, sevennet_calc))

write('./eos/eos.extxyz', eos_oneshot)

# draw EOS curve and compare with DFT
dft_eos, dft_vol = get_eos_and_volume(read(os.path.join(data_path, 'evaluation/eos.extxyz'), ':'))
mlp_eos, mlp_vol = get_eos_and_volume(read(os.path.join(working_dir, 'eos/eos.extxyz'), ':'))

plt.figure(figsize=(10/2.54, 8/2.54))
plt.plot(dft_vol, dft_eos, label='DFT')
plt.plot(mlp_vol, mlp_eos, label='From scratch')

plt.xlabel(r"Volume ($\rm{\AA}^3$)")
plt.ylabel("Relative energy (eV)")

plt.legend()
plt.tight_layout()
plt.savefig('./results/eos_curve.png')