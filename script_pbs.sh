#PBS -S /bin/bash
#PBS -N biobert_2009_fullfinetuning
#PBS -P arkhn-3a
#PBS -l walltime=0:59:00
#PBS -l select=1:ngpus=1:mem=32gb
#PBS -q gpuq 

# Go to the current directory 
cd $PBS_O_WORKDIR

# Load the same modules as for compilation 
module load anaconda3/5.3.1
source activate pypa_env

## pip install flair

# # pip freeze | grep -v -f requirements.txt - | xargs pip uninstall -y
# # pip install -r requirements.txt --upgrade --no-warn-already-satisfied

# Run code
python ./pypa.py \
    --data_path  data/inputs/2009/dataframe_final_clean.csv \
    --n_epochs 1001 \
    --pretrained_model 'monologg/biobert_v1.1_pubmed' \
    --modified_model \
    --full_finetuning \
    --weighted_loss 'global' \
    --batch_size 100 \
    --l2_regularization 1e-1