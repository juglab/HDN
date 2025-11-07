#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/group/jug/aman/HDN_runs/%x_%j.log
#SBATCH --nodes=1
#SBATCH --mem=256GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=dgx
#SBATCH --gres=gpu:35gb:1
#SBATCH --job-name=HDN_convallaria
#SBATCH --time=36:00:00

cd ~/HDN
conda activate HDN
# Run inference
/scratch/aman.kukde/conda/envs/HDN/bin/python3.7 /home/aman.kukde/HDN/run_notebooks.py --notebook "/home/aman.kukde/HDN/examples/Pixel_Noise/Convallaria/2-Training.ipynb" --outputdir "/group/jug/aman/HDN_runs/notebooks/"