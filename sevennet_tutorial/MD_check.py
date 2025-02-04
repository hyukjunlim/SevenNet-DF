from ase.io import read
from ase.visualize import view
from ase.io import write

# Read the trajectory file
traj = read('moldyn3.traj', index=':')

write('movie.gif', traj, interval=10)
