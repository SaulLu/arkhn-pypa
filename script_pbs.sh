#PBS -S /bin/bash
#PBS -N biobert_2009_fullfinetuning
#PBS -P arkhn-3a
#PBS -l walltime=01:00:00
#PBS -l select=1:ngpus=1:mem=20gb
#PBS -o output_pypa.txt
#PBS -e error_pypa.txt
#PBS -q gpuq 

# Go to the current directory 
cd $PBS_O_WORKDIR

# Load the same modules as for compilation 
module load anaconda3/5.3.1
source activate pypa_env

## pip freeze | grep -v -f requirements.txt - | xargs pip uninstall -y
## pip install -r requirements.txt --upgrade --no-warn-already-satisfied

# Run code
python ./pypa.py \
    --n_epochs 101 \
    --pretrained_model 'monologg/biobert_v1.1_pubmed' \
    --modified_model \
    