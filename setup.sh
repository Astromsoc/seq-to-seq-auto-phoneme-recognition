# create a conda virtual env for this project
conda create -n dl3p2 python=3.9
conda activate dl3p2
# recommend running the 2 lines above seperately to avoid problems

# packages
pip install python-Levenshtein torchsummaryX torchaudio numpy clang
conda install pytorch torchvision torchtext -c pytorch

# trivial but necessary for data downloads & exp monitoring
pip install kaggle wandb pyyaml wget

# vital: ctc-decoder
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
cd ..

# download & unzip data
kaggle competitions download -c 11-785-f22-hw3p2
unzip -q 11-785-f22-hw3p2