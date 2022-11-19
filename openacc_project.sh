#!/bin/bash

#SBATCH --job-name=openacc_project
#SBATCH --output=openacc_project-%J.out
#SBATCH --ntasks=1
#SBATCH --qos=gpu
#SBATCH --mem=3000            # memory per node in MB 
#SBATCH --requeue
#SBATCH --gres=gpu:volta:1
#SBATCH --time=01:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XYZ@crimson.ua.edu
source /opt/asn/etc/asn-bash-profiles-special/modules.sh
module load oneapi/2021.1.1_mkl
module load cuda/11.7.0
module load pgi/20.1
# module load openmpi/4.1.4-gcc11-cuda

export PGI_ACC_NOTIFY=2 # verbosity on each data transfer
export PGI_ACC_TIME=1 # timing summary for each kernel

# pgcc -c gameoflife_1d_acc.c -o gameoflife_1d_acc -ta=tesla:cc70 -Minfo=accel
# pgcc -fast -Minfo=accel -ta=tesla,time,cc70 -acc  -DDEBUG2 2dgol.c -o 2dgol
# pgcc -fast -Minfo=accel -ta=tesla,time,cc70 -acc  gol_project.c -o gol_project
# ./gol_project  10000 5000  /scratch/ualclsb0056

pgcc -fast -Minfo=accel -ta=tesla,time,cc70 -acc  openacc_project.c -o openacc_project
# ./openacc_project  1000 1000  /scratch/ualclsb0056
# ./openacc_project  1000 1000  /scratch/ualclsb0056
# ./openacc_project  1000 1000  /scratch/ualclsb0056
# ./openacc_project  1000 5000  /scratch/ualclsb0056
# ./openacc_project  1000 5000  /scratch/ualclsb0056
# ./openacc_project  1000 5000  /scratch/ualclsb0056
# ./openacc_project  5000 1000  /scratch/ualclsb0056
# ./openacc_project  5000 1000  /scratch/ualclsb0056
# ./openacc_project  5000 1000  /scratch/ualclsb0056
# ./openacc_project  5000 5000  /scratch/ualclsb0056
# ./openacc_project  5000 5000  /scratch/ualclsb0056
# ./openacc_project  5000 5000  /scratch/ualclsb0056
# ./openacc_project  10000 1000  /scratch/ualclsb0056
# ./openacc_project  10000 1000  /scratch/ualclsb0056
# ./openacc_project  10000 1000  /scratch/ualclsb0056
# ./openacc_project  10000 5000  /scratch/ualclsb0056
# ./openacc_project  10000 5000  /scratch/ualclsb0056
./openacc_project  10000 5000  /scratch/ualclsb0056



