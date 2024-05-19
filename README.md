# BOND: Bernstein Over-approximated Neural Dynamics
This is the official code for L4DC 2024 paper "Real-Time Safe Control of Neural Network Dynamic Models with Sound Approximation" [PDF](https://arxiv.org/abs/2404.13456) can be found here.

## Preparation
This repo is based on [Julia](https://julialang.org/) and is tested with Julia v1.8.0. Check [here](https://julialang.org/downloads/oldreleases/) to install Julia environment. Install `NeuralVerficiaton.jl` from [this repo](https://github.com/intelligent-control-lab/NeuralVerification.jl) under branch `nn-safe-control` using the Julia package manager as follows,

```julia
import Pkg
Pkg.add(url="https://github.com/intelligent-control-lab/NeuralVerification.jl.git", rev="nn-safe-control")
```

## Models
Download pre-trained models from [here](https://github.com/intelligent-control-lab/NNDM-safe-control/tree/master/nnet) and place all the model folders under root path. The training script can be found [here](https://github.com/intelligent-control-lab/NNDM-safe-control/blob/master/src/train_nn_dynamics.py).

## Comparison with baseline
Run the following scripts for both baseline and ours. Set `SIS=true` for safety index synthesis. Set `STATS=true` to find quantitative results by sampling multiple trajectories. Set `VISUALIZE=true` to visualize and debug. More options can be set for ablation studies, e.g. `USE_IA_FLAG,BPO_DEGREE,BPO_SOLVER,P_NORM`.

For collision avoidance, run
```julia
include("test_collision_original.jl") # baseline
include("test_collision_BPO.jl") # ours
```
For safe following, run
```julia
include("test_following_original.jl") # baseline
include("test_following_BPO.jl") # ours
```
## Reference 
- [NNDM-safe-control](https://github.com/intelligent-control-lab/NNDM-safe-control)
- [NeuralVerification.jl](https://github.com/intelligent-control-lab/NeuralVerification.jl)

## Citation 
If you find the repo useful, please cite:

H. Hu, J. Lan and C. Liu
"[Real-Time Safe Control of Neural Network Dynamic Models with Sound Approximation](https://arxiv.org/abs/2404.13456)", Learning for Dynamics and Control Conference (L4DC). PMLR, 2024
```
@article{hu2024real,
  title={Real-Time Safe Control of Neural Network Dynamic Models with Sound Approximation},
  author={Hu, Hanjiang and Lan, Jianglin and Liu, Changliu},
  journal={arXiv preprint arXiv:2404.13456},
  year={2024}
}
```