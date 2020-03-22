#PBS -S /bin/bash
#PBS -N test_pypa_dir
#PBS -P arkhn-3a
#PBS -l walltime=04:00:00
#PBS -l select=1:ngpus=2:mem=30gb
#PBS -o output_pypa.txt
#PBS -e error_pypa.txt
#PBS -q gpuq 

# Go to the current directory 
cd $PBS_O_WORKDIR

# Load the same modules as for compilation 
module load anaconda3/5.3.1
source activate pypa_env

pip freeze | grep -v -f requirements.txt - | xargs pip uninstall -y
pip install -r requirements.txt --upgrade

# Run code
python ./pypa.py --n_epochs 11 --pretrained_model 'bert-base-cased'
