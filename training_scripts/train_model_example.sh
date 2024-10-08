#!/bin/bash
#SBATCH --job-name=trans
#SBATCH --partition=gpu_test
#SBATCH --time=1:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --chdir=/n/home11/nswood/RS3L_2/
#SBATCH --output=slurm_monitoring/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=5323
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed

# cd Mixed_Curvature
source ~/.bashrc

source /n/holystore01/LABS/iaifi_lab/Users/nswood/mambaforge/etc/profile.d/conda.sh


# conda activate adenv
conda activate flat-samples


### the command to run 

python train_model.py --temperature 0.1 --nepochs 10 --batchsize 250 --dropout_rate 0.2 --lr 0.0001 --nparts 100 --nclasses 2 --feature_size 10 --trans --class_att --num_encoders 4 --num_attention_heads 4
