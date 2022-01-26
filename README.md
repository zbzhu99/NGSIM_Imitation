# Generative Adversarial Imitation Learning on NGSIM I-80 Dataset

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

4. Download the NGSIM I-80 dataset from: https://drive.google.com/file/d/14j5S8lGir9J5QAl8AHMamDu3WTDQAQc3/view?usp=sharing, and place it under `smarts-imitation/ngsim`

5. Install the ngsim wrapper for smarts with

   ```bash
   pip install -e smarts_imitation
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

8. Download the PPUU simulator from: 

9. Generate structured expert demonstrations with

   ```bash
   python run_experiment.py -e exp_specs/gen_expert/ppuu.yaml
   ```

10. Run GAIL with

   ```bash
   python run_experiment.py -e exp_specs/gail/gail_pppu.yaml
   ```
