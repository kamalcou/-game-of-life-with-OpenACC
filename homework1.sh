#!/bin/bash
#
#SBATCH --job-name=Homework1_Test
#SBATCH --output=Homework1-%J.out
#SBATCH --ntasks=1
##SBATCH --qos=gpu
##SBATCH --requeue
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=2000mb

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Name of the cluster on which the job is executing." $SLURM_CLUSTER_NAME
echo "Number of tasks to be initiated on each node." $SLURM_TASKS_PER_NODE
echo "Number of cpus requested per task." $SLURM_CPUS_PER_TASK
echo "Number of CPUS on the allocated node." $SLURM_CPUS_ON_NODE
echo "Total number of processes in the current job." $SLURM_NTASKS
echo "List of nodes allocated to the job" $SLURM_NODELIST
echo "Total number of nodes in the job's resource allocation." $SLURM_NNODES
echo "List of allocated GPUs." $CUDA_VISIBLE_DEVICES
echo "---------------------"
echo "---------------------"


#source /opt/asn/etc/asn-bash-profiles-special/modules.sh
#module load cuda/11.7.0
module load cuda11.4/blas/11.4.2
module load cuda11.4/fft/11.4.2
module load cuda11.4/nsight/11.4.2
module load cuda11.4/profiler/11.4.2
module load cuda11.4/toolkit/11.4.2




nvcc -DDEBUG0 -o HW1_base HW1_base.cu
./HW1_base  10000 5000
./HW1_base  10000 5000
./HW1_base  10000 5000
# ./run.out  $2 $3
# ./run.out  $2 $3


