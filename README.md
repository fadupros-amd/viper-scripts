# viper-scripts  : script/data for both Viper and Ouranos
## Simple binding check
- **binding_simple_ouranos.sh** : report binding on ouranos (using default SLURM Setting for the pytorch/RCCL reproducer)
  -*Thread 140692290500096 affinity mask: 24 25 26 27 28 29 30 31 32 33 34 35 72 73 74 75 76 77 78 79 80 81 82 83*
  -*Thread 139956932949504 affinity mask: 0 1 2 3 4 5 6 7 8 9 10 11 48 49 50 51 52 53 54 55 56 57 58 59*

- **binding_simple_viper.sh** : report binding on viper-gpu (using default SLURM Setting for the pytorch/RCCL reproducer)
  -*Thread 22798218901312 affinity mask: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23*
  -*Thread 23043020330816 affinity mask: 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47*
  
## One-time setup on Ouranos (login node) for the Pytorch_RCCL Reproducer + model download (+300 GB)
- **RCCL-issue.tar.gz** : reference archive for the reproducer
- **pytorch_rccl_reproducer_setup_login_node.sh** :
  * define $TARGET_DIR for the over-arching  root directory for this benchmark. 
  * setup pixi and modify default settings to target Ouranos (fetch the archive, adjust paths, modify slurm script...)
  * prevent data fetching from the node in the slurm script, unlike the original/default setup on Viper - (export HF_HUB_OFFLINE=1 and export TRANSFORMERS_OFFLINE=1)
  * download +300 GB of input files (from HuggingFace).
  * NOTE : 1/This scripts needs further cleaning  2/Full validation needed w.r.t viper-gpu settings.

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
  
