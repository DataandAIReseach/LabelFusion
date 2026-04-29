# download and install miniforge
cd ~
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
CUSTOM_CONDA_ROOT="$HOME/conda"
# comment in the following line if you want to install conda in the project-specific data store instead of your home directory
# CUSTOM_CONDA_ROOT="$HOME/.project/dir.project/conda_$USER" 
MAMBA_ROOT_PREFIX="$CUSTOM_CONDA_ROOT" CONDA_ROOT_PREFIX="$CUSTOM_CONDA_ROOT" bash Miniforge3.sh -b -p "$CUSTOM_CONDA_ROOT"
rm Miniforge3.sh

# create .condarc file for some basic configurations
cat > "$HOME/.condarc" <<EOF
channels:
  - conda-forge
channel_priority: flexible
EOF

# initialize conda in shell
source "$CUSTOM_CONDA_ROOT/etc/profile.d/conda.sh"
conda init
