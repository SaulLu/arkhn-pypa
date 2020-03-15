#PBS -S /bin/bash
<<<<<<< HEAD
#PBS -N test_pypa3
#PBS -P arkhn-3a
#PBS -l walltime=04:00:00
#PBS -l select=1:ngpus=1:mem=30gb
#PBS -M jean-baptiste.laval@student.ecp.fr
#PBS -m be
#PBS -o output_pypa_1.txt
#PBS -e error_pypa_1.txt
#PBS -q gpuq 
=======
#PBS -N test_pypa1
#PBS -P arkhn-3a
#PBS -l walltime=01:00:00
#PBS -M jean-baptiste.laval@student.ecp.fr
#PBS -m be
#PBS -o output_pypa_1.txt
#PBS -e error_pypa_1.txt 
>>>>>>> 0cf9fd6963ae6f0877dfcaa536693c0680e15048

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
>>>>>>> 0cf9fd6963ae6f0877dfcaa536693c0680e15048
