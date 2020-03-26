#PBS -S /bin/bash
#PBS -N viz_metrics
#PBS -P arkhn-3a
#PBS -l walltime=00:05:00
#PBS -l select=1:ngpus=1:mem=100mb
#PBS -q gpuq 

# Go to the current directory 
cd $PBS_O_WORKDIR

# Load the same modules as for compilation 
module load anaconda3/5.3.1
source activate pypa_env

# Run code
python ./src/utils/visualization.py --model_pathname 'test'
