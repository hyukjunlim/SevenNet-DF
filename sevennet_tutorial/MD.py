import sys
sys.path.append('../../SevenNet')
import os
working_dir = os.getcwd() # save current path
working_dir = os.path.join(working_dir, 'experiment')

"""Demonstrates molecular dynamics with constant energy."""

from ase import units
from ase.io.trajectory import Trajectory
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
import ase.io

mode = 'df'
# mode = '7net-0'

data_path = './data'
DFT_md_xyz = os.path.join(data_path, 'evaluation/test_md.extxyz')
traj = ase.io.read(DFT_md_xyz, index=':')
atoms = traj[0]

# size = 3
# # Set up a crystal
# atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
#                           symbol="Cu",
#                           size=(size, size, size),
#                           pbc=True)

# Describe the interatomic interactions with the Effective Medium Theory
if mode == 'df':
    from sevenn_dfs.sevennet_calculator import SevenNetCalculator
    atoms.calc = SevenNetCalculator(os.path.join(working_dir, 'checkpoint_best_base_full5.pth'))
elif mode == '7net-0':
    from sevenn.sevennet_calculator import SevenNetCalculator
    atoms.calc = SevenNetCalculator(model='7net-0', device='cpu')

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# We want to run MD with constant energy using the VelocityVerlet algorithm.
dyn = VelocityVerlet(atoms, 2 * units.fs)  # 2 fs time step.
if mode == 'df':
    dyn.run(1000)
else:
    dyn.run(50)

#########################################################

atoms = traj[0]
if mode == 'df':
    from sevenn_dfs.sevennet_calculator import SevenNetCalculator
    atoms.calc = SevenNetCalculator(os.path.join(working_dir, 'checkpoint_best_base_full5.pth'))
elif mode == '7net-0':
    from sevenn.sevennet_calculator import SevenNetCalculator
    atoms.calc = SevenNetCalculator(model='7net-0', device='cpu')

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# We want to run MD with constant energy using the VelocityVerlet algorithm.
dyn = VelocityVerlet(atoms, 2 * units.fs)  # 2 fs time step.
    
Etot = []
def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    Etot.append(epot + ekin)
    # print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
        #   'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin), flush=True)


dyn = VelocityVerlet(atoms, 2 * units.fs)  # 2 fs time step.
# Now run the dynamics
dyn.attach(printenergy, interval=1)
# We also want to save the positions of all atoms after every 100th time step.
traj = Trajectory('moldyn3.traj', 'w', atoms)
dyn.attach(traj.write, interval=1)

# Now run the dynamics
printenergy()

import time
start = time.time()
dyn.run(1000)
print('Time:', time.time() - start, 's')


import matplotlib.pyplot as plt

# Function to plot the data
def plot_data(Etot):
    plt.figure(figsize=(10, 6))

    # Plotting Epot, Ekin, Etot against time
    plt.plot(Etot, label='Etot')

    # Adding labels and title
    plt.xlabel('Time (2fs per tick)')
    plt.ylabel('Energy per atom (eV)')
    plt.title('Fluctuation of Etot with Time')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('MD_plot_time.png')

plot_data(Etot)
