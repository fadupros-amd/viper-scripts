# viper-scripts
Script/Ddata for Viper/Ouranos
- **binding_simple_ouranos.sh** : report binding on ouranos (using default SLURM Setting for the pytorch/RCCL reproducer)
- **binding_simple_viper.sh** : report binding on viper-gpu (using default SLURM Setting for the pytorch/RCCL reproducer)
- **pytorch_rccl_reproducer_setup_login_node.sh** :
  - On ouranos/login node (see TARGETDIR for the over-arching  target directory - ~350 GB needed) - To be ran once.
    1/ setup pixi and modify default settings for the reproducer (paths, slurm submission script...) 
    2/ Download +300 GB of input files (from HuggingFace).
