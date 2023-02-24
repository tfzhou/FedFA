#!/bin/bash

#SBATCH -n 4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:22g
#SBATCH --mem-per-cpu=10000
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=logs_pyfed/config.prostate_mri.fedsam_prostate_trial0.out

source ../../../../pytorch-1.11/bin/activate

module load gcc/8.2.0 python_gpu/3.10.4

#module unload eth_proxy

## copy data
rsync -aP /cluster/work/cvl/tiazhou/data/medical/FL/ProstateMRI.zip ${TMPDIR}/
unzip -q ${TMPDIR}/ProstateMRI.zip -d ${TMPDIR}/ProstateMRI

cd ../../../

trial=0
train_ratio=0.1

python main.py --config 'config.prostate_mri.fedsam' \
               --server 'euler' \
               --trial $trial \
               --train_ratio $train_ratio \
               --run_name 'fedsam_ratio_'$train_ratio'trial_'$trial \
               --run_notes 'trial '$trial': fedsam' \
               --exp_name 'fedsam'