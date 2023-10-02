# Learning Space-Time Continuous Neural PDEs from Partially Observed States

Official implementation of [Learning Space-Time Continuous Neural PDEs from Partially Observed States](https://arxiv.org/abs/2307.04110).

## Installation

```bash
git clone https://github.com/yakovlev31/LatentNeuralPDEs.git
cd LatentNeuralPDEs
pip install -r requirements.txt
pip install -e .
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