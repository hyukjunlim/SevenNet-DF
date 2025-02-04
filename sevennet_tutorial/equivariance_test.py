import sys
sys.path.append('../../SevenNet')
import sevenn_df
import os.path
import numpy as np

data_path = './data'
working_dir = os.getcwd() # save current path
assert os.path.exists(data_path) and os.path.exists(os.path.join(data_path, 'train'))

from sevenn_df.train.graph_dataset import SevenNetGraphDataset
from scipy.spatial.transform import Rotation as R

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
train_shift = dataset.per_atom_energy_mean
train_scale = {'force': dataset.force_rms, 'stress': dataset.stress_rms}
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
  'optim_param': {'lr': 0.01},
  'scheduler': 'linearlr',
  'scheduler_param': {'start_factor': 1.0, 'total_iters': 100, 'end_factor': 0.0001},
  # 'scheduler': 'exponentiallr',
  # 'scheduler_param': {'gamma': 0.99},
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
import torch
import ase.io

# Let's test our model by predicting DFT MD trajectory
# Instead of using other functions in SevenNet, we will use ASE calculator as an interface of our model
DFT_md_xyz = os.path.join(data_path, 'evaluation/test_md.extxyz')

def voigt_to_full_stress(voigt_stress):
    """
    voigt_stress: array-like of length 6 in the order:
                   [xx, yy, zz, yz, xz, xy]

    returns a 3x3 numpy array (full stress tensor)
    """
    sxx, syy, szz, syz, sxz, sxy = voigt_stress
    stress_tensor = np.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxz, syz, szz]
    ])
    return stress_tensor

def full_stress_to_voigt(stress_tensor):
    """
    stress_tensor: 3x3 matrix

    returns a length-6 array in the ASE order [xx, yy, zz, yz, xz, xy]
    """
    sxx = stress_tensor[0,0]
    syy = stress_tensor[1,1]
    szz = stress_tensor[2,2]
    sxy = stress_tensor[0,1]
    syz = stress_tensor[1,2]
    sxz = stress_tensor[0,2]
    
    return np.array([sxx, syy, szz, syz, sxz, sxy])

def test_single_structure_rotation(atoms, calculator):
    """
    atoms      : ase.Atoms object
    calculator : SevenNetCalculator or any ASE-compatible calculator
    """
    # 1) Evaluate original system
    atoms.calc = calculator
    E_orig = atoms.get_potential_energy()             # scalar
    F_orig = atoms.get_forces()                       # (N,3)
    stress_orig_voigt = atoms.get_stress() * to_kBar            # (6,) in ASE Voigt format
    stress_orig_3x3 = voigt_to_full_stress(stress_orig_voigt)  # (3,3)

    print(f"MSE of energy: {(E_orig - ground_truth_E)**2:.6e}")
    print(f"MSE of forces: {np.mean((F_orig - ground_truth_F)**2):.6e}")
    print(f"MSE of stress: {np.mean((stress_orig_voigt - ground_truth_stress)**2):.6e}")
    # 2) Define a rotation R (e.g. 45° about z, 30° about y, 60° about x).
    #    Choose whatever angles you want, or random ones.
    rotation = R.from_euler('zyx', [45, 30, 60], degrees=True)
    R_mat = rotation.as_matrix()  # shape (3,3)

    # 3) Make a copy of the original atoms & rotate both cell and positions
    rotated_atoms = atoms.copy()

    # (a) rotate cell
    # ASE store cell vectors as row vectors [a, b, c]. So we do cell' = R * cell.
    cell_orig = atoms.get_cell()           # shape (3,3)
    cell_rotated = cell_orig @ R_mat.T
    rotated_atoms.set_cell(cell_rotated, scale_atoms=False)

    # (b) rotate positions
    # positions are Nx3 row vectors. For each row r_i, new_r_i = R_mat @ r_i
    # we can do it with dot as well.
    pos_orig = atoms.get_positions()       # (N,3)
    pos_rotated = pos_orig @ R_mat.T        # rotate each row
    rotated_atoms.set_positions(pos_rotated)

    # 4) Evaluate rotated system
    rotated_atoms.calc = calculator
    E_rot = rotated_atoms.get_potential_energy()
    F_rot = rotated_atoms.get_forces()
    stress_rot_voigt = rotated_atoms.get_stress() * to_kBar
    stress_rot_3x3 = voigt_to_full_stress(stress_rot_voigt)

    # 5) Check energy invariance
    print()
    print(f"Original energy: {E_orig:.6f} eV")
    print(f"Rotated  energy: {E_rot:.6f} eV")
    print(f"Energy difference: {(E_orig - E_rot):.6e} eV")

    # 6) Check force equivariance
    #    The correct transformation for forces under R is F' = R * F,
    #    if positions transform as r' = R * r.
    #    But be mindful that your row vs. column conventions might differ.
    F_orig_rotated_pred = (R_mat @ F_orig.T).T  # shape (N,3)
    force_diff = F_rot - F_orig_rotated_pred
    print()
    print(f"Max force difference: {np.abs(force_diff).max():.6e} eV/A")

    # 7) Check stress equivariance
    #    Stress is a rank-2 tensor, so under R it transforms as sigma' = R * sigma * R^T.
    #    We compare the predicted stress_rot_3x3 to the "rotated" original stress.
    stress_orig_rotated_3x3 = R_mat @ stress_orig_3x3 @ R_mat.T
    stress_diff = stress_rot_3x3 - stress_orig_rotated_3x3
    print()
    print(f"Max stress difference: {np.abs(stress_diff).max():.6e}")
    print("Stress (3x3) difference:\n", stress_diff)
    
    print("Stress original (3x3):\n", ground_truth_stress)
    print("Stress predicted (3x3):\n", stress_rot_voigt)
    print("Stress (3x3) rotated original:\n", stress_orig_rotated_3x3)
    print("Stress (3x3) rotated predicted:\n", stress_rot_3x3)
    
    # Return all computed values if needed for further inspection
    return {
        'E_orig': E_orig,
        'F_orig': F_orig,
        'stress_orig_3x3': stress_orig_3x3,
        'E_rot': E_rot,
        'F_rot': F_rot,
        'stress_rot_3x3': stress_rot_3x3,
    }


# load DFT md trajectory
traj = ase.io.read(DFT_md_xyz, index=':')
atoms = traj[-1]

atoms1 = atoms.copy()
atoms2 = atoms.copy()

to_kBar = 1602.1766208
ground_truth_E = atoms.info['DFT_energy']
ground_truth_F = atoms.arrays['DFT_forces']
ground_truth_stress = atoms.info['DFT_stress'] * to_kBar
print(f"\n[Label]")
print(f"Ground truth energy: {ground_truth_E:.6f} eV")
print(f"Ground truth forces (eV/A):\n", ground_truth_F)
print(f"Ground truth stress (kBar):\n", ground_truth_stress)

# test sevennet_df calculator
from sevenn_df.sevennet_calculator import SevenNetCalculator
dfcalc7 = SevenNetCalculator(os.path.join(working_dir, 'checkpoint_best.pth'))
print(f"\n[7net-DF]")
result = test_single_structure_rotation(atoms1, dfcalc7)

# test sevennet calculator
from sevenn.sevennet_calculator import SevenNetCalculator
calc7 = SevenNetCalculator(model='7net-0', device='cuda')
print(f"\n[7net-0]")
result = test_single_structure_rotation(atoms2, calc7)
