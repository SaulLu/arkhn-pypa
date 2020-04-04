#PBS -S /bin/bash
#PBS -N biobert_dropout_0.1_finetunning_False
#PBS -P arkhn-3a
#PBS -l walltime=01:30:00
#PBS -l select=1:ngpus=2:mem=30gb
#PBS -o output_G_m.txt
#PBS -e error_G_m.txt
#PBS -q gpuq
#PBS -M jean-baptiste.laval@student.ecp.fr
#PBS -m be 

# Go to the current directory 
cd $PBS_O_WORKDIR

# Load the same modules as for compilation 
module load anaconda3/5.3.1
source activate pypa_env

pip freeze | grep -v -f requirements.txt - | xargs pip uninstall -y
pip install -r requirements.txt --upgrade --no-warn-already-satisfied

# Run code
python ./pypa.py --n_epochs 50 --pretrained_model 'monologg/biobert_v1.1_pubmed' --modified_model True --dropout 0.1 --full_finetuning False
