#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition gpu2
#SBATCH -t 1-01:05:00
#SBATCH -J tutorial
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
# # SBATCH --gres=gpu:1

nvidia-smi
# python equivariance_test.py
python tuto.py
# python tuto_full.py
# python tuto_full_7net_0.py
# python tuto_7net_0.py
# python inference_test.py
# python MD.py
# python MAE_test.py






# #!/bin/bash 
# #SBATCH --partition=A1                # select partition (A1, A2, A3, A4, B1, B2, or B3) 
# #SBATCH --time=24:00:00               # set time limit in HH:MM:SS 
# #SBATCH --nodes=1                     # number of nodes 
# #SBATCH --ntasks-per-node=20          # number of processes per node (for MPI) 
# #SBATCH --cpus-per-task=1             # OMP_NUM_THREADS (for openMP) 
# #SBATCH --job-name=Pt                 # job name #SBATCH --output="error.%x"
#                                       # standard output and error are redirected to 
#                                       # <job name>_<job ID>.out # for OpenMP jobs export
#                                       # if srun -n not working, try mpirun -np
# OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} 
# # load modules if needed

# # quantum espresso executable 
# QE=/home/jmlim/program/qe-dev/bin 
# WANN=/home/jmlim/program/wannier90/wannier90.x 
# # NOTE # $SLURM_NTASKS : Total number of cores (N_nodes * N_tasks_per_node) 

# # run parallel 
# mpirun -np $SLURM_NTASKS $QE/pw.x -nk 2 -ndiag 1 -in scf.in > scf.out 
# mpirun -np $SLURM_NTASKS $QE/pw.x -nk 2 -ndiag 1 -in nscf.in > nscf.out 
# exit 0; 

# # mpirun -np $SLURM_NTASKS $QE/pw.x -nk 2 -ndiag 1 -in bands.in > bands.out 
# # ~/bin/my_qe_bands.py Pt temp 
# # cp temp/Pt.xml temp/Pt.save/data-file-schema.xml 

# # sed -i "/dis_froz_max/c\dis_froz_max = $EFrozMax" GeTe.bulk.win 

# mpirun -np 1 $WANN -pp Pt 
# mpirun -np $SLURM_NTASKS $QE/pw2wannier90.x -ndiag 1 -in pw2wan.in > pw2wan.out 
# mpirun -np $SLURM_NTASKS $WANN Pt 

# # job array example 
# #SBATCH --array=1-2 # job array 
# cp -r temp temp."$SLURM_ARRAY_TASK_ID" 
# filein=BiTeI.ph"$SLURM_ARRAY_TASK_ID".in 
# fileout=BiTeI.ph"$SLURM_ARRAY_TASK_ID".out 
# mpirun -np 32 $QE/ph.x -nk 8 -ndiag 1 -in $filein > $fileout