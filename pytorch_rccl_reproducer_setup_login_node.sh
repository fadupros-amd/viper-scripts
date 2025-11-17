# Setup
# To Run on the login Node - Ouranos
# One-time-setup



# Setup
####################

export TARGET_DIR=/lustremi/users/$USER/ATOSA-47-reproducer
export HF_HOME=$TARGET_DIR/HF_HOME
export HF_DATASETS_CACHE=$TARGET_DIR/HF_DATA_CACHE
export PATH="/home_nfs/xduprosf/.pixi/bin:$PATH"
export PIXI_HOME=$TARGET_DIR/PXI_HOME
export PIXI_CACHE_DIR=$TARGET_DIR/PXI_CACHE
########
export HUGGINGFACE_HUB_TOKEN=
########

mkdir $TARGET_DIR
cd $TARGET_DIR



# Download initial repository
####################
wget https://github.com/fadupros-amd/viper-scripts/raw/refs/heads/main/RCCL-issue.tar.gz
tar -zxvf  RCCL-issue.tar.gz
cd RCCL-issue


# PIXI Setup
####################
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$PIXI_HOME/bin:$PATH"
which pixi
pixi info

pixi clean cache -y
pixi reinstall -e rocm
####################
sed -i '/^\[feature\.rocm\.activation\.env\]/,/^\[.*\]/ {
  s|^IMAGENET200_LABEL_MAPPING_PATH = .*|IMAGENET200_LABEL_MAPPING_PATH = "'"$TARGET_DIR"'/RCCL-issue/data/imagenet/meta/tinyin200_to_in1k_mapping.txt"|
  s|^IMAGENET_LABELS_PATH = .*|IMAGENET_LABELS_PATH = "'"$TARGET_DIR"'/RCCL-issue/data/imagenet_labels.json"|
  s|^DATA_PATH = .*|DATA_PATH = "'"$TARGET_DIR"'/RCCL-issue/data/imagenet"|
  s|^IMAGENET_TRAIN_SPLIT = .*|IMAGENET_TRAIN_SPLIT = "'"$TARGET_DIR"'/RCCL-issue/data/imagenet/tiny_imagenet_train.txt"|
  s|^IMAGENET_VAL_SPLIT = .*|IMAGENET_VAL_SPLIT = "'"$TARGET_DIR"'/RCCL-issue/data/imagenet/tiny_imagenet_val.txt"|
  s|^HUGGINGFACE_DATASETS_CACHE = .*|HUGGINGFACE_DATASETS_CACHE = "'"$TARGET_DIR"'/RCCL-issue/data/huggingface/datasets"|
}' pixi.toml


####################
pixi run -e rocm wandb disabled
pixi run -e rocm hf auth login --token TBD  --add-to-git-credential
####################
sed -i '1,11d' scripts/viper-gpu/google_vit/imagenet/vitc/google_vit_b_16_B2.sh
sed -i "1i\\
#!/bin/bash\\
#SBATCH -J RCCL_reproducer\\
#SBATCH -o $TARGET_DIR/RCCL_reproducer%j.out\\
#SBATCH -e $TARGET_DIR/RCCL_reproducer%j.err\\
#SBATCH --ntasks-per-node=2  # This needs to match Trainer(devices=...)\\
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)\\
#SBATCH --cpus-per-task=24\\
#SBATCH -p All_AMD_Node\\
#SBATCH -t 12:00:00\\
#SBATCH --exclusive\\
#SBATCH --no-requeue"  scripts/viper-gpu/google_vit/imagenet/vitc/google_vit_b_16_B2.sh

sed -i '/^#SBATCH --no-requeue/a\
\
export TARGET_DIR=/lustremi/users/$USER/ATOSA-47-reproducer\
export HF_HOME=$TARGET_DIR/HF_HOME\
export HF_DATASETS_CACHE=$TARGET_DIR/HF_DATA_CACHE\
export HF_DATASETS_DISABLE_MEMORY_MAPPING=1\
export HF_HUB_OFFLINE=1\
export TRANSFORMERS_OFFLINE=1\
export PATH="/home_nfs/xduprosf/.pixi/bin:$PATH"\
export PIXI_HOME=$TARGET_DIR/PXI_HOME\
export PIXI_CACHE_DIR=$TARGET_DIR/PXI_CACHE' scripts/viper-gpu/google_vit/imagenet/vitc/google_vit_b_16_B2.sh
sed -i 's|/u/gajdab/.pixi/bin/pixi|pixi|g' scripts/viper-gpu/google_vit/imagenet/vitc/google_vit_b_16_B2.sh



#Download +300GB from login node in $TARGET_DIR
####################
learning_rate=0.001
optimizer="adamw"
batch_size=256
epochs=90
num_workers=12
model="google_vit_b_16_B=2"
group="google-vit-imagenet-vit_b_16_B=2"
weight_decay=0.1

 
pixi run train_viper --experiment_name "$model" --num_workers $num_workers --group "$group" --epochs 90  --dataset "imagenet1k" --batch_size $batch_size --weight_decay $weight_decay --lr $learning_rate --optimizer $optimizer --compile_mode "default"

