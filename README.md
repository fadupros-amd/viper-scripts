# viper-scripts
Script/Ddata for Viper/Ouranos
- **binding_simple_ouranos.sh** : report binding on ouranos (using default SLURM Setting for the pytorch/RCCL reproducer)
- **binding_simple_viper.sh** : report binding on viper-gpu (using default SLURM Setting for the pytorch/RCCL reproducer)

- **RCCL-issue.tar.gz** : reference archive for the reproducer
- **pytorch_rccl_reproducer_setup_login_node.sh** : On ouranos/login node (see TARGETDIR for the over-arching  target directory - ~350 GB needed) - One-time setup.
  * Setup pixi and modify default settings for the reproducer (fetch the archive, adjust paths, modify slurm script...)
  * Prevent data fetching from the node in the slurm script, unlike the original/default setup on Viper - (export HF_HUB_OFFLINE=1 and export TRANSFORMERS_OFFLINE=1)
  * Download +300 GB of input files (from HuggingFace).
  * NOTE : 1/This scripts needs further cleaning 2/Full validation needed w.r.t viper-gpu settings (internet on the compute node)

- **pytorch_rccl_reproducer_submission.sh** : Slurm script for the pytorch_rccl_reproducer
- 
