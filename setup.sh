#!/bin/bash
# Sim-to-Sim Policy Transfer Setup Script
# Target: Ubuntu 22.04 LTS (tested on 22.04, may have issues on 24.04)
set -e

# Color definitions
NC='\033[0m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'

echo -e "${BLUE}========================================================${NC}"
echo -e "${BLUE}  Sim-to-Sim Legged Locomotion Setup - VISTEC Exam${NC}"
echo -e "${BLUE}========================================================${NC}"

# Check Ubuntu version
UBUNTU_VER=$(lsb_release -rs)
if [[ "$UBUNTU_VER" != "22.04" ]]; then
    echo -e "${YELLOW}Warning: Tested on Ubuntu 22.04. You are running $UBUNTU_VER${NC}"
    echo -e "${YELLOW}Isaac Gym may have compatibility issues.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verify Isaac Gym
ISAACGYM_PATH="./unitree_rl_gym/isaacgym"
if [ ! -d "$ISAACGYM_PATH" ]; then
    echo -e "${RED}Error: Isaac Gym not found at $ISAACGYM_PATH${NC}"
    echo -e "${YELLOW}Download from: https://developer.nvidia.com/isaac-gym${NC}"
    exit 1
fi

# Setup Conda
echo -e "${GREEN}[1/4] Setting up Conda environment...${NC}"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda info --envs | grep -q "unitree_rl"; then
    echo -e "${YELLOW}Removing existing 'unitree_rl' environment...${NC}"
    conda env remove -n unitree_rl -y
fi

conda create -n unitree_rl python=3.8 -y
conda activate unitree_rl

# Install system-level Python 3.8 lib (for Isaac Gym)
echo -e "${GREEN}[2/4] Installing system dependencies for Isaac Gym...${NC}"
if [ ! -f "/usr/lib/x86_64-linux-gnu/libpython3.8.so.1.0" ]; then
    echo -e "${YELLOW}Installing libpython3.8...${NC}"
    sudo apt-get update
    sudo apt-get install -y libpython3.8 libpython3.8-dev
fi

# Install Isaac Gym first
echo -e "${GREEN}[3/4] Installing Isaac Gym...${NC}"
pip install -e "$ISAACGYM_PATH/python"

# Detect CUDA and install compatible PyTorch
echo -e "${GREEN}[4/4] Installing PyTorch and dependencies...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}NVIDIA driver not found!${NC}"
    exit 1
fi

CUDA_VER=$(nvidia-smi | grep -Po 'CUDA Version: \K(\d+\.\d+)' | head -n1)
CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
echo -e "${BLUE}Detected CUDA $CUDA_VER${NC}"

# PyTorch compatibility matrix
if [ "$CUDA_MAJOR" -ge 13 ]; then
    echo -e "${YELLOW}CUDA $CUDA_VER detected - using latest PyTorch (2.4.1)${NC}"
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
elif (( $(echo "$CUDA_VER >= 12.1" | bc -l) )); then
    echo -e "${BLUE}Using PyTorch 2.1.0 for CUDA 12.1+${NC}"
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
elif (( $(echo "$CUDA_VER >= 11.8" | bc -l) )); then
    echo -e "${BLUE}Using PyTorch 2.0.1 for CUDA 11.8${NC}"
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${RED}CUDA version $CUDA_VER too old. Need >= 11.8${NC}"
    exit 1
fi

# Install remaining dependencies
pip install mujoco mujoco-viewer gymnasium tensorboard
pip install numpy scipy pandas scikit-learn matplotlib seaborn pillow typeguard pyyaml tqdm

# Install project packages
pip install -e unitree_rl_gym/

# Install ActuatorNet (check if requirements.txt exists)
if [ -f "actuator_net/requirements.txt" ]; then
    pip install -r actuator_net/requirements.txt
else
    echo -e "${YELLOW}actuator_net/requirements.txt not found - skipping${NC}"
fi

# Verification
echo -e "${BLUE}========================================================${NC}"
echo -e "${GREEN}Running sanity checks...${NC}"
echo -e "${BLUE}========================================================${NC}"

python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" || echo -e "${RED}✗ PyTorch FAILED${NC}"
python -c "from isaacgym import gymapi; print('✓ Isaac Gym OK')" || echo -e "${RED}✗ Isaac Gym FAILED${NC}"
python -c "import mujoco; print(f'✓ MuJoCo {mujoco.__version__} OK')" || echo -e "${RED}✗ MuJoCo FAILED${NC}"

echo -e "${BLUE}========================================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}Activate with: conda activate unitree_rl${NC}"
echo -e "${BLUE}========================================================${NC}"