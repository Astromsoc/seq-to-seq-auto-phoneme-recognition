# create a conda virtual env for this project
conda create -n dl3p2 python=3.9
conda activate dl3p2
# recommend running the 2 lines above seperately to avoid problems

# packages
pip install python-Levenshtein torchsummaryX torchaudio numpy pandas clang matplotlib kaggle wandb pyyaml wget
conda install pytorch torchvision torchtext cudatoolkit -c pytorch
# to ensure that ctcencode could be correctly installed
# make sure to install the following c/c++ libraries
conda install gcc_linux-64 gxx_linux-64


# vital: ctc-decoder
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
cd ..

# download & unzip data
kaggle competitions download -c 11-785-f22-hw3p2
unzip -q 11-785-f22-hw3p2