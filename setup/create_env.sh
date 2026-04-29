ENV_PATH='/scratch1/users/u19147/envs/env_labelFusion_py312'

eval "$(mamba shell hook --shell bash)"
mamba env remove -p $ENV_PATH -y
mamba create -p $ENV_PATH python=3.12.3 -y
mamba activate $ENV_PATH

mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install nvidia/label/cuda-11.8.0::cuda-toolkit -y
mamba install mkl=2024.0.0 -y
pip install numpy==1.26.4
pip install jupyter
conda install anaconda::scikit-learn -y
