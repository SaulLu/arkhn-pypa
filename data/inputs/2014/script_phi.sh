#PBS -S /bin/bash
#PBS -N pypa_extract
#PBS -P arkhn-3a
#PBS -l walltime=00:15:00
#PBS -M name@student.ecp.fr
#PBS -m be
#PBS -o output_pypa_extract.txt
#PBS -e error_pypa_error.txt

# Go to the current directory
cd $PBS_O_WORKDIR

# Load the same modules as for compilation
module load anaconda3/5.3.1
source activate pypa_env

# Run code
python /workdir/NAME/pypa/src/parsers/2014/extract_job_phi.py /workdir/NAME/pypa/data/inputs/2014/training-PHI-Gold-Set1/
