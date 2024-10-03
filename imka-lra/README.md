# In-Memory Kernel Approximation: Code for the experiments relating to LRA

## Getting started 🚀

First of all, create a conda environment with `python>=3.10`. You can use the following command to create an environment containing all the dependencies of the project.

```bash
conda create --name kernel python=3.10 openblas -y
conda activate kernel
```
In the follwoing steps we will install the libraries necessaries to run our experimental code.
We recommend to create a separate folder where all these external repositories can be stored (e.g. a `dependency` folder inside the root).

### (1) Installing `fairseq`
Clone the fairseq repository from Facebook Research, checkout the right branch, and apply our patches to add our custom model definitions (the patch file is included in the folder `./patches`).
```bash
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout 3f6ba43f07a6e9e2acf957fc24e57251a7a3f55c
git apply fairseq.patch
pip install -e .
```
Problems with conflicting packages versions might be caused by newer versions of `pip`. If you experience such problems, consider downgrading your `pip` with
```bash
pip install pip==24.0
```

### (2) Installing `xformers`
If you're installing `xformers` on MacOS, you might run into troubles with `llvm`. If this is the case, install it using 
```bash
brew install llvm libomp
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

Finally, to install the `xformers` package, clone the xformers repository from Facebook Research, checkout the right branch, and apply our patches to add our custom model definitions (the patch file is included in the folder `./patches`).
```bash
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git checkout efcd7894e3f54f8a13abbb8058d7fdef1904d2d2
git apply xformers.patch
pip install -r requirements.txt
pip install ninja
git submodule update --init --recursive
pip install -e .
```

### (3) Installing `aihwkit`
For **GPU installation**, follow these steps:
```bash
git clone https://github.com/IBM/aihwkit.git; cd aihwkit
pip install -r requirements.txt
conda install mkl mkl-include -y
cd ~ ; mkdir aihwkit_installation ; cd aihwkit_installation ; nano install_aiwhkit.sh
```
Paste the following and make sure to change the CUDA paths accordingly. Also,
make sure the CUDA version matches the one you installed PyTorch for.
```bash
export CXX=/usr/bin/g++
export CC=/usr/bin/gcc
export MKLROOT=$CONDA_PREFIX
export CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib
export CUDA_VERSION=12.1
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}
export CUDA_LIB_PATH=${CUDA_HOME}/lib
export CUDA_INCLUDE_DIRS=${CUDA_HOME}/include
export PATH=${CUDA_HOME}/bin:$COMPILER/bin:${PATH}
cd ~/aihwkit
make build_inplace_cuda
```

Change the permissions of the file:
```bash
chmod +x install_aiwhkit.sh
```

Execute the script
```bash
. install_aiwhkit.sh
```

In your `~/.bashrc` add the following at the bottom:
```bash
export PYTHONPATH=$HOME/aihwkit/src/
```

For the **CPU installation**, execute
```bash
git clone https://github.com/IBM/aihwkit.git; cd aihwkit
pip install -r requirements.txt
conda install openblas -y
pip install mypy
pip install -e .
export PYTHONPATH=[path to aihwkit folder]:$PYTHONPATH
```

### (4) Installing `imka-lra`
Move to the main `imka-lra` folder and run the following commands.
```
pip install -r requirements.txt
pip install pandas
pip install -e .
```

## Models and Datasets
To download the model checkpoints and dataset files, follow these steps
```bash
wget https://ibm.box.com/shared/static/5i6u7hpeobunhazlbikxl84zmc3jjpdi.zip
unzip 5i6u7hpeobunhazlbikxl84zmc3jjpdi.zip; rm 5i6u7hpeobunhazlbikxl84zmc3jjpdi.zip
cp -r imka-lra-data/LRA ~; rm -r imka-lra-data/LRA
cp -r imka-lra-data/models [path to imka-lra]/resources; rm -r imka-lra-data
```

Make sure to move every component in the right file system position, otherwise problems with paths might arise.


## Running
**Note:** Since we worked with a slightly different private version of AIHWKIT and
some randomness-inducing code was omitted, results differ slightly. However,
the deviations are minor and don't change the claims of the paper.

### Attention-only
Obtaining the results from the paper Table 1, row 1.
First, apply the attention-only patch to fairseq
```bash
cd fairseq
# revert the other patch, if needed
git apply -R [path to fairseq]/fairseq.patch
git apply [path to fairseq]/fairseq-attn-only.patch
cd [path to imka-lra]
```
then run
```bash
python experiments/lra/evaluation.py --task all --mode fp_attn_only --seed all
```
| Task         | Mean     | Std      |
|--------------|----------|----------|
| aan          | 77.8550% | 0.8929%  |
| cifar10      | 46.1100% | 0.7499%  |
| imdb         | 66.2100% | 1.0521%  |
| pf32         | 70.7350% | 1.2793%  |
| listops      | 37.5000% | 0.0000%* |
*Note: We don't re-draw projections in this code for the listops task.

### Full on-chip model
Go to `fairseq` and checkout `main`.

For evaluating the pre-trained HW-aware models on the subset that was used
for running the full model on-chip, run

```bash
# revert the attention-only patch, if needed
git apply -R [path to fairseq]/fairseq-attn-only.patch
git apply [path to fairseq]/fairseq.patch
python experiments/lra/evaluation.py --task imdb --mode hw_aware --seed 5
python experiments/lra/evaluation.py --task cifar10 --mode hw_aware --seed 8
python experiments/lra/evaluation.py --task pf32 --mode hw_aware --seed 7
python experiments/lra/evaluation.py --task aan --mode hw_aware --seed 0

# for running the ReLU based model from the discussion
python experiments/lra/evaluation.py --task cifar10_relu --mode hw_aware --seed 7
```

For running the pre-trained models on all of the samples, pass `--samples -1`.

Obtaining the results from the paper Table 1, row 3, run
```bash
python experiments/lra/evaluation.py --task all --mode hw_aware --seed all
```

You should get
| Task         | Mean     | Std      |
|--------------|----------|----------|
| aan          | 78.3400% | 0.6111%  |
| cifar10      | 46.0350% | 0.6633%  |
| cifar10_relu | 48.6900% | 0.7908%  |
| imdb         | 65.5750% | 0.9943%  |
| pf32         | 72.1850% | 0.8983%  |
| listops      | 38.8500% | 0.0000%* |
*Note: We don't re-draw projections in this code for the listops task.

## Training
### FP-32
Training the FP-32-model locally can be done using:
```bash
python experiments/lra/train_fp.py --task <TASK (e.g. cifar10)> --mode local --data_dir /path/to/LRA --base_dir /path/where/to/store --config_name config/lra/fp/<TASK>.yaml
```

**Note:** For the attention-only FP-32 models checkout the `attn-only` branch of fairseq and pass `--config_name config/lra/fp_attn_only/<TASK>.yaml`

### HW-aware training
For a specific task, go to `experiments/lra/aihwkit_training.py` and uncomment the specific lines of code for that task. Then,
execute 
```bash
python experiments/lra/aihwkit_training.py --config_name config/lra/fp/<TASK>.yaml
```
**Note:** All models can be trained on one V100, except the AAN task, which requires more DRAM.

**Note:** For the attention-only models, you need to checkout the `attn-only` branch of fairseq. Also, for HW-aware training,
follow the notes in `experiments/lra/aihwkit_training.py`.
And execute
```bash
python experiments/lra/aihwkit_training.py --config_name config/lra/fp_attn_only/<TASK>.yaml
```