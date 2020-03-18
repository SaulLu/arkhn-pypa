#PBS -S /bin/bash
#PBS -N test_pypa1
#PBS -P arkhn-3a
#PBS -l walltime=04:00:00
#PBS -l select=1:ngpus=1:mem=30gb
#PBS -M jean-baptiste.laval@student.ecp.fr
#PBS -m be

#PBS -N test_pypa3
#PBS -P arkhn-3a
#PBS -l walltime=04:00:00
#PBS -l select=1:ngpus=1:mem=30gb
#PBS -o output_pypa.txt
#PBS -e error_pypa.txt
#PBS -q gpuq 

# Go to the current directory 
cd $PBS_O_WORKDIR

# Load the same modules as for compilation 
module load anaconda3/5.3.1
source activate pypa_env

# Run code
python ./pypa.py

