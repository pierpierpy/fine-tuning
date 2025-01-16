#!/bin/bash
#SBATCH --account=try23_ACN         
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=3         
#SBATCH --gres=gpu:4               
#SBATCH --time=01:30:00             
#SBATCH --error=logs/load_model/load_model.%j.err           
#SBATCH --output=logs/load_model/load_model.%j.out          
#SBATCH --partition=boost_usr_prod  
#SBATCH --qos=normal
#SBATCH --job-name=load_model
hostname
export HF_HUB_CACHE=$FAST/.cache
echo $HF_HUB_CACHE

python src/utils/load_model.py "" "<token>" "$FAST/.cache/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080" "llama-70"

# sometimes the model is too big to be saved in FAST directly from the login node, in this case
# downlaod the model from the login onto the cache (the cache is defaulted in the load_model.py as FAST/.cache),
# use this bash slurm script to save the cached snapshot model to the right folder  

# to check where is the snapshot use huggingface-cli scan-cache

# CREATE A load_model folder into logs#!/bin/bash
#SBATCH --account=try23_ACN         
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=3         
#SBATCH --gres=gpu:4               
#SBATCH --time=01:30:00             
#SBATCH --error=logs/load_model/load_model.%j.err           
#SBATCH --output=logs/load_model/load_model.%j.out          
#SBATCH --partition=boost_usr_prod  
#SBATCH --qos=normal
#SBATCH --job-name=load_model
hostname
export HF_HUB_CACHE=$FAST/.cache
echo $HF_HUB_CACHE

python src/utils/load_model.py "" "<token>" "$FAST/.cache/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080" "llama-70"

# sometimes the model is too big to be saved in FAST directly from the login node, in this case
# downlaod the model from the login onto the cache (the cache is defaulted in the load_model.py as FAST/.cache),
# use this bash slurm script to save the cached snapshot model to the right folder  

# to check where is the snapshot use huggingface-cli scan-cache

####  ATTENZIONE  ####
# CREATE A load_model folder into logs