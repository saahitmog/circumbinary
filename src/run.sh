#!/bin/bash
#SBATCH --job-name=circumbinary_transits
#SBATCH --account=fc_searth
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=24:00:00

module load python
python3 task.py savio