# Prelminaries
* This repository should be cloned into a folder named "repos" in your home directory.
* It is further assumed that you use a linux shell, e.g. ``bash``. You can choose which shell is used on the SCC/NHR on your AcademicCloud profile under ``OTHER``.
* The shell commands provided in this guide assume that you have a shell opened with the root directory of this repository as the working directory.

# 1. Making shell prompt nicer
* While not absolutely necessary, it is much nicer to have a colorful shell prompt to make it easily distinguishable from the commands you run.
* This and [many other](https://unix.stackexchange.com/questions/129143/what-is-the-purpose-of-bashrc-and-how-does-it-work) shell-related things can be modified by adapting your ``.bashrc`` file in your home directory
* Run the following shell script for some basic visual enhancements of your shell prompt:
```
bash 1_setup/modify_bashrc.sh
```
* Open a new shell session now. You should see that the visual appearance of your shell prompt changed (to the better).

# 2. Installing miniforge
* miniforge is a minimal distribution of conda that allows to structure packages, for example python packages, into environments.
* In 2025, the disc space in the home directory was increased to 60 GB, which is why it is now possible to install miniforge there without running into memory issues. Nevertheless, you should always monitor the disc space (run ``show-quota`` in command line) since it is easily possible to exceed the quota by accidentally saving large files there. There are various other data stores for these purposes (more on that later)
* Running the following shell script installs miniforge in your home directory:
```
bash 1_setup/install_miniforge.sh
```
* The installation of miniforge might take a while. After it finished, open a new shell session. Conda should now be available in this new session.

# 3. Creating conda environment with Pytorch
* Once miniforge has been installed and initialized, you can create your own conda environments. The examples in this guide only require pytorch to be installed. If conda is activated, you can install an environment containing pytorch by running the following command:
```
source 1_setup/create_env.sh
```
* The creation of the environment might take a while since Pytorch is quite a big package.
