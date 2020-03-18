#PBS -S /bin/bash
<<<<<<< HEAD
#PBS -N test_pypa1
#PBS -P arkhn-3a
#PBS -l walltime=01:00:00
#PBS -M jean-baptiste.laval@student.ecp.fr
#PBS -m be
#PBS -o output_pypa_1.txt
#PBS -e error_pypa_1.txt 
=======
#PBS -N test_pypa3
#PBS -P arkhn-3a
#PBS -l walltime=04:00:00
#PBS -l select=1:ngpus=1:mem=30gb
#PBS -o output_pypa.txt
#PBS -e error_pypa.txt
#PBS -q gpuq 
>>>>>>> 99f0f78b7597175725127de9d70cb4ed57afea1c

# Go to the current directory 
cd $PBS_O_WORKDIR

# Load the same modules as for compilation 
module load anaconda3/5.3.1
source activate pypa_env

# Run code
<<<<<<< HEAD
python ./pypa.py
=======
python ./pypa.py
>>>>>>> 99f0f78b7597175725127de9d70cb4ed57afea1c
