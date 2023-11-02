# Learning Space-Time Continuous Neural PDEs from Partially Observed States

Official implementation of [Learning Space-Time Continuous Neural PDEs from Partially Observed States](https://arxiv.org/abs/2307.04110).

## Installation

To make the code work please run the following commands in that order.
```bash
git clone https://github.com/yakovlev31/LatentNeuralPDEs.git; 
cd LatentNeuralPDEs; 
conda create -n lnpde_env python=3.10; 
conda activate lnpde_env; 
conda install -c conda-forge fenics; 
pip install torch==2.1.0; 
pip install wandb==0.15.12; 
pip install tqdm==4.66.1; 
pip install matplotlib; 
pip install scipy; 
pip install scikit-learn; 
pip install einops; 
pip install git+https://github.com/rtqichen/torchdiffeq; 
pip install seaborn; 
pip install -e .; 
```

## Getting Started

### Data

Archive with datasets can be downloaded [here](https://drive.google.com/file/d/1uFxs-A4MhvsZdBCsudPuvBca69zWDmUq/view?usp=sharing). The datasets should be extracted to `./experiments/data/`.

If you want to use your own dataset, follow the scripts in `./lnpde/utils/`.

### Training and testing

```bash
python experiments/{_shallow_water,_navier_stokes,_scalar_flow}/train.py --name mymodel --device cuda --visualize 1

python experiments/{_shallow_water,_navier_stokes,_scalar_flow}/test.py --name mymodel --device cuda
```

See `./msvi/utils/{_shallow_water,_navier_stokes,_scalar_flow}.py` for all command line arguments.
