# Generative Adversarial Imitation Learning on NGSIM I-80 Dataset

[![Maintainability](https://api.codeclimate.com/v1/badges/a1f4c9260c5298bc8c40/maintainability)](https://codeclimate.com/github/zbzhu99/NGSIM_Imitation/maintainability)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)

This repository contains three components: an improved version of ILSwiss, which is an RL training suite containing the implementation of Generative Adversarial Imitation Learning (GAIL), and two environments. The two environments both are driving simulators that use [NGSIM I-80]((https://www.fhwa.dot.gov/publications/research/operations/06137/)) dataset as background traffic. One of them is PPUU, the other is implemented as an imitation learning scenario based on [SMARTS](https://github.com/huawei-noah/SMARTS.git) platform.

## How to setup:

0. Clone this repository

1. Install python requirements with

```bash
pip install -r requirements.txt
```

### Setup SMARTS

2. Download the SMARTS platform as a submodule with

   ```bash
   git submodule init
   git submodule update
   ```

3. Install SMARTS simulation platform in the `./SMARTS` folder following its instructions

   ```bash
   cd ./SMARTS
   bash ./utils/setup/install_deps.sh
   echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc
   pip install -e .[camera-obs]
   ```

4. Download the NGSIM dataset from: https://drive.google.com/file/d/1SILBfK2Z1LiTJ9cc4qGErq547Ax5nX8M/view?usp=share_link, and place it under `smarts-imitation/`

5. Install the ngsim wrapper for smarts with

   ```bash
   pip install -e ./smarts-imitation
   # You may encounter the ModuleNotFoundError when running this command,
   # which is okay. Please just wait until it terminates.
   scl scenario build --clean ./smarts-imitation/ngsim
   ```

6. Generate structured expert demonstrations with

   ```bash
   cd ILSwiss
   python run_experiment.py -e exp_specs/gen_expert/smarts.yaml
   ```

7. Run GAIL with

   ```bash
   python run_experiment.py -e exp_specs/gail/gail_smarts.yaml
   ```

### Setup PPUU

8. Make sure you have downloaded the NGSIM dataset and placed it under `smarts-imitation`. Create a link to the dataset under PPUU folder:

   ```bash
   mkdir pytorch-PPUU/traffic-data
   ln -s $(pwd)/smarts-imitation/xy-trajectories pytorch-PPUU/traffic-data/
   ```

   Process the traffic data to dump the "state, action, cost" triple:

   ```bash
   cd pytorch-PPUU
   for t in 0 1 2; do python generate_trajectories.py -map i80 -time_slot $t; done
   ```

9. Generate structured expert demonstrations with

   ```bash
   python run_experiment.py -e exp_specs/gen_expert/ppuu.yaml
   ```

10. Run GAIL with

   ```bash
   python run_experiment.py -e exp_specs/gail/gail_pppu.yaml
   ```
