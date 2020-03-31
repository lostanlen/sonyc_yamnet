#!/bin/bash

#BATCH --job-name=01_sonycnode-b827eb4cc22e
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=1GB
#SBATCH --output=../slurm/01_sonycnode-b827eb4cc22e_%j.out

module purge

# The first argument is the path to the SONYC sensor.
# The second argument is the path to the output directory.
python ../../../src/01_run-yamnet_covid19_20-03-30.py /scratch/mc6591/covid_audio/sonycnode-b827eb4cc22e.sonyc /scratch/vl1019/c19_data