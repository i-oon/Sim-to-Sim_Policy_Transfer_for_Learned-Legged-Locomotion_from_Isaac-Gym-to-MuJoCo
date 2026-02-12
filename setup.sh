#!/bin/bash

# Sim-to-Sim Policy Transfer Setup Script (VISTEC Internship Exam Edition)
# Author: Disthorn Suttawet
# Target OS: Ubuntu 22.04 LTS

set -e  # Exit immediately if a command exits with a non-zero status

# --- Color Definitions for Output ---
NC='\033[0m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   Sim-to-Sim Legged Locomotion Setup - VISTEC Exam   ${NC}"
echo -e "${BLUE}======================================================${NC}"

# STEP 2 Verification: Check if Isaac Gym is placed correctly (As per README Step 2)
ISAACGYM_PATH="./unitree_rl_gym/isaacgym"
if [ ! -d "$ISAACGYM_PATH" ]; then
    echo -e "${RED}Error: Isaac Gym not found at $ISAACGYM_PATH${NC}"
    echo -e "${YELLOW}Please follow Step 2 in README: Place 'isaacgym' folder inside 'unitree_rl_gym'${NC}"
    exit 1
fi

# STEP 3: Setup Conda Environment
echo -e "${GREEN}[1/3] Setting up Conda environment 'unitree_rl'...${NC}"

# Source conda to enable 'conda activate' within the script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda info --envs | grep -q "unitree_rl"; then
    echo -e "${YELLOW}Environment 'unitree_rl' already exists. Activating...${NC}"
else
    conda create -n unitree_rl python=3.8 -y
fi
conda activate unitree_rl

# STEP 4: Install Dependencies (PyTorch, MuJoCo, and RL Science Stack)
echo -e "${GREEN}[2/3] Installing Core Dependencies and PyTorch...${NC}"

# Install Isaac Gym Python interface first
pip install -e "$ISAACGYM_PATH/python"

# Detect CUDA version to install the correct PyTorch wheel
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}NVIDIA Driver not found. Please install drivers before proceeding.${NC}"
    exit 1
fi

CUDA_VER=$(nvidia-smi | grep -Po 'CUDA Version: \K(\d+\.\d+)' | head -n1)
echo -e "${BLUE}Detected CUDA Version: $CUDA_VER${NC}"

# Logical check for PyTorch version based on CUDA 12.1 vs 11.8
if (( $(echo "$CUDA_VER >= 12.1" | bc -l) )); then
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
fi

# Install MuJoCo, Gymnasium, and standard data science libraries
pip install mujoco mujoco-viewer gymnasium tensorboard
pip install numpy scipy pandas scikit-learn matplotlib seaborn pillow typeguard pyyaml tqdm

# STEP 5: Install Local Project Packages
echo -e "${GREEN}[3/3] Installing local project packages...${NC}"
pip install -e unitree_rl_gym/
pip install -r actuator_net/requirements.txt

# --- VERIFICATION (As per README Step 4) ---
echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}Installation Complete! Running Sanity Checks...${NC}"
echo -e "${BLUE}======================================================${NC}"

python -c "import torch; print(f'✓ PyTorch OK (CUDA: {torch.cuda.is_available()})')"
python -c "from isaacgym import gymapi; print('✓ Isaac Gym OK')"
python -c "import mujoco; print('✓ MuJoCo OK')"

echo -e "${YELLOW}Usage: Run 'conda activate unitree_rl' to start.${NC}"
echo -e "${BLUE}======================================================${NC}"