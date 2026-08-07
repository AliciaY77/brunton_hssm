#!/bin/bash
#SBATCH --job-name=brunton_train
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/xyuan48/brunton_hssm/cluster/log/%x-%j.out
#SBATCH --error=/users/xyuan48/brunton_hssm/cluster/log/%x-%j.err

cd /users/xyuan48/brunton_hssm
module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate hssm
export PYTHONPATH=/users/xyuan48/brunton_hssm:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"
python -u src/train_lan_t750_smallB.py
