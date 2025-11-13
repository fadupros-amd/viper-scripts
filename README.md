# viper-scripts  : script/data for both Viperand Ouranos
## Simple binding check
- **binding_simple_ouranos.sh** : report binding on ouranos (using default SLURM Setting for the pytorch/RCCL reproducer)
- **binding_simple_viper.sh** : report binding on viper-gpu (using default SLURM Setting for the pytorch/RCCL reproducer)

## One-time setup on Ouranos (login node) for the Pytorch_RCCL Reproducer + model download (+300 GB)
- **RCCL-issue.tar.gz** : reference archive for the reproducer
- **pytorch_rccl_reproducer_setup_login_node.sh** :
  * define $TARGET_DIR for the over-arching  root directory for this benchmark. 
  * setup pixi and modify default settings for the reproducer (fetch the archive, adjust paths, modify slurm script...)
  * prevent data fetching from the node in the slurm script, unlike the original/default setup on Viper - (export HF_HUB_OFFLINE=1 and export TRANSFORMERS_OFFLINE=1)
  * download +300 GB of input files (from HuggingFace).
  * NOTE : 1/This scripts needs further cleaning  2/Full validation needed w.r.t viper-gpu settings (internet on the compute node)

## Simple job submission
- `sbatch $TARGET_DIR/scripts/viper-gpu/google_vit/imagenet/vitc/google_vitc_b_16_B2.sh`


## Crontab 
- **crontab_submit.sh** : wrapper script for job management / submission
- Set the crontab with the ususal command : crontab -e (e.g. every hour, 0 * * * * crontab_submit.sh)

## Customizing the benchmark
- Training based on Pytorch, dataset is imagenet. 
- Normal completion is *CANCELLED.. DUE TO TIME LIMIT* after 12 hours.
- One can modify the benchmark to reduce overall elsapsed time
   - Adapt the epoch parameter (number of full pass on the images) from 90 to 1.
   - Runtime is now below 20 minutes and completion message is : *Training completed!* + some sanity checks
  
