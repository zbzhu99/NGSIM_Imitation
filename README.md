# Generative Adversarial Imitation Learning on NGSIM I-80 Dataset

[![Total alerts](https://img.shields.io/lgtm/alerts/g/zbzhu99/NGSIM_Imitation.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/zbzhu99/NGSIM_Imitation/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/zbzhu99/NGSIM_Imitation.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/zbzhu99/NGSIM_Imitation/context:python)
[![Maintainability](https://api.codeclimate.com/v1/badges/a1f4c9260c5298bc8c40/maintainability)](https://codeclimate.com/github/zbzhu99/NGSIM_Imitation/maintainability)

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

4. Download the NGSIM I-80 dataset from: https://drive.google.com/file/d/14j5S8lGir9J5QAl8AHMamDu3WTDQAQc3/view?usp=sharing, and place it under `smarts-imitation/ngsim`

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

8. Download data file for the PPUU simulator from: https://drive.google.com/file/d/1oMzxTWK-wzpFufi2SryyD2Pa_l1tTEW0/view?usp=sharing, and place it under `pytorch-PPUU/`

9. Generate structured expert demonstrations with

   ```bash
   python run_experiment.py -e exp_specs/gen_expert/ppuu.yaml
   ```

10. Run GAIL with

   ```bash
   python run_experiment.py -e exp_specs/gail/gail_pppu.yaml
   ```
