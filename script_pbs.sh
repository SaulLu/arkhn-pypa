#PBS -S /bin/bash
#PBS -N test_pypa1
#PBS -P arkhn-3a
#PBS -l walltime=01:00:00
#PBS -M jean-baptiste.laval@student.ecp.fr
#PBS -m be
#PBS -o output_pypa_1.txt
#PBS -e error_pypa_1.txt 

# Go to the current directory 
cd $PBS_O_WORKDIR

# Load the same modules as for compilation 
module load anaconda3/5.3.1
source activate pypa_env

# Run code
python ./pypa.py