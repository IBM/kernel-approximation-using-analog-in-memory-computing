# Kernel Approximation for In-Memory Computing
Implementation of kernel-approximation techniques for in-memory computing devices.
Supported approximation techniques:

- Random Fourier Features [1]
- Orthogonal Random Features [2]
- Structured Orthogonal Random Features [2]
- FAVOR+ [4]


## Table of Content 🗂️
- [Kernel Approximation for In-Memory Computing](#kernel-approximation-for-in-memory-computing)
  - [Table of Content 🗂️](#table-of-content-️)
  - [Get started 🚀](#get-started-)
    - [Create conda environment](#create-conda-environment)
    - [Fetch experiment data](#fetch-experiment-data)
  - [Experiment replication 🧪](#experiment-replication-)
      - [Figure 2 and 3](#figure-2-and-3)
      - [Supplementary Figures 10 and 11](#supplementary-figures-10-and-11)
      - [Supplementary Figures 1-6](#supplementary-figures-1-6)
  - [References 📖](#references-)



## Get started 🚀
In this section, we include a detail guide on how to install all the dependencies of the project in order to get started with our project.

### Create conda environment
First of all, create a conda environment with `python>=3.10`. You can use the following command to create an environment containing all the dependencies of the project.

```bash
conda create --name kernel python=3.10.9 openblas -y
conda activate kernel
```

Install all the required dependencies using the `requirements.txt` file inside the main project folder.
```bash
cd [your-main-folder-location]/Kernel-Approximation
pip install -r requirements.txt
```

Finally, you should be ready to install the project library itself.
```bash
python setup.py develop
```


### Fetch experiment data
The supported datasets are:

- ijcnn01 [link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)
- magic04 [link](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)
- letter [link](https://archive.ics.uci.edu/dataset/59/letter+recognition)
- skin [link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)
- EEG [link](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)
- cod-rna [link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)
- covtype [link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)
- Syntetic Attention Dataset (generated from Performer model trained on IMBD)

You can download the data by executing in the main folder the following commands
```bash
wget https://ibm.box.com/shared/static/c3brah3t04o8ruixb4m7z94e2rylsz8o.zip
unzip c3brah3t04o8ruixb4m7z94e2rylsz8o.zip; rm c3brah3t04o8ruixb4m7z94e2rylsz8o.zip
```


## Experiment replication 🧪
To replicate our experiments, the following scripts are available. All the results will be stored in the `resources` folder, under the corresponding experiment folders.

#### Figure 2 and 3
The code to replicate Figure 2 and 3, use
```bash
python plots.py
```

To recompute the experimental results of these two figures, please re-run all the experiments as described in the instructions below to replicate Supplementary Figures 1-6.
Note that this will only re-compute the FP32 baseline -- re-computing the hardware results would require physical access to the chip.
The recomputation of the results will lead to small deviations from the values reported in our manuscript - this is because the results are influenced by the random seeds (used to draw the random features). Since the hardware result computation is missing, this will result in different random seeds being drawn at inference time. The deviation of results is negligible.

#### Supplementary Figures 10 and 11
- Liu et al [1] replication, Supplementary Figure 10 in our manuscript: `python experiments/replications/liu.py`

- Choromanski et al [2] replication, Supplementary Figure 10 in our manuscript: `python experiments/replications/choromanski.py`

#### Supplementary Figures 1-6
To run all the experiment together, use
```bash
./experiments/hardware/run.sh
```
To run every experiments individually, use
- `python experiments/hardware/hardware_general.py --config experiments/hardware/config/[config_name].yml` for experiments on RBF and ArcCos kernels
- `python experiments/hardware/hardware_attn.py` for experiments on Softmax kernel

Note that these experiments will only allow to replicate the FP32 baselines reported in the paper. Hardware replication would require the physical access to the Hermes Project chip and the disclosure of the software stack built on top of it. All the experimental evidences gathered in our hardware experiments are however stored in pickle files, to allow the re-computation of some of the figures in the manuscript.


## References 📖

[1] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In Advances in neural information processing systems, pages 1177–1184, 2007.

[2] F. Yu, A. Suresh, K. Choromanski, D. Holtmann-Rice, and S. Kumar. Orthogonal random features. In NIPS, 2016.

[3] F. Liu, X. Huang, Y. Chen and J. A. K. Suykens, "Random Features for Kernel Approximation: A Survey on Algorithms, Theory, and Beyond," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 7128-7148, 1 Oct. 2022, doi: 10.1109/TPAMI.2021.3097011.

[4] Krzysztof Marcin Choromanski and Valerii Likhosherstov and David Dohan and Xingyou Song and Andreea Gane and Tamas Sarlos and Peter Hawkins and Jared Quincy Davis and Afroz Mohiuddin and Lukasz Kaiser and David Benjamin Belanger and Lucy J Colwell and Adrian Weller, "Rethinking Attention with Performers",
in International Conference on Learning Representations, 2021.
