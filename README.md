### About
This repo focuses on 3 things:
1. Install Anaconda, Pytorch, Jupyer Notebook, etc. on a brand-new Jeston Nano 2G.
2. Setup, run, and benchmark Videopose3D
3. Optimize Videopose3D models with TensorRT and benchmark the inference frame rate again.

### Install Depencenties

Jeston Nano uses the aarch64 or arm64 architecture, rather than the widely-adopted x86 architecture. Softwares like Anaconda, Pytorch, Jupyter Notebook, etc. are found to have installation issues following the official instructions. Working installation steps were scattered in forums and blogs around the web, and this document is here to condense them in one place.

TL;DR: How to install Anaconda, Pytorch, Jupyter Notebook, etc. on Jeston Nano 2G.
 
### Instructions on how to setup and run VideoPose3D on Jeston Nano 2G

(adpated from https://sahilramani.com/2020/10/how-to-setup-python3-and-jupyter-notebook-on-jetson-nano/)

 
```bash
# update source.list
sudo apt update
sudo apt upgrade

# install miniforge3 (the official anaconda did not work in my case)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
chmod a+x Miniforge3-Linux-aarch64.sh
./Miniforge3-Linux-aarch64.sh
rm ./Miniforge3-Linux-aarch64.sh
bash  # source from .bashrc

# create videopose3d env
conda config --set auto_activate_base false  # not auto-activate the base env
sudo apt install python3-h5py libhdf5-serial-dev hdf5-tools python3-matplotlib
conda create -n videopose3d python=3.6

# install necessary packages to the videopose3d env
conda activate videopose3d
conda install matplotlib numpy tqdm h5py jupyter ipywidgets
# install torch 1.6 
wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
sudo apt install python3-pip libopenblas-base libopenmpi-dev
pip install torch-1.6.0-cp36-cp36m-linux_aarch64.whl
# install ffmpeg
git clone https://github.com/jocover/jetson-ffmpeg.git
cd jetson-ffmpeg
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig
cd ../..
git clone git://source.ffmpeg.org/ffmpeg.git -b release/4.2 --depth=1
cd ffmpeg
wget https://github.com/jocover/jetson-ffmpeg/raw/master/ffmpeg_nvmpi.patch
git apply ffmpeg_nvmpi.patch
./configure --enable-nvmpi
make
cd ..
# install torchvision
sudo apt install libjpeg-dev zlib1g-dev
git clone --branch v0.7.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.7.0  
python setup.py install
cd ../
pip install 'pillow<7'

# setup the jupyter server
jupyter notebook --generate-config
vim /home/huanx/.jupyter/jupyter_notebook_config.py
# let jupyter listen to the wildcard address, exposing port 8888
# c.NotebookApp.open_browser = False
# c.NotebookApp.ip = '*'
jupyter notebook password  # set password for the jupyter server
python -m ipykernel install --user
```

### Setup Videopose3D
```bash
git clone https://github.com/facebookresearch/VideoPose3D.git
cd VideoPose3D
# dataset setup
# TODO
# download pretrained models
# TODO
# run inference
# benchmark inference on Jupyter Notebook
```

### Optimize models with TensorRT
