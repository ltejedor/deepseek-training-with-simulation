# DeepSeek Robot Design Trainer

This project implements a training pipeline for the DeepSeek language model to generate robot designs using the Gazebo simulator for validation. The system uses Gazebo to evaluate generated robot designs in SDF (Simulation Description Format) and provides feedback for reinforcement learning.

Utils are from Strong Compute, project is built to run on Strong Compute infrastructure. See https://github.com/StrongResearch/isc-demos for more details.


### Required Software

1. **Gazebo Simulator**
Follow installation instructions at https://gazebosim.org/docs/latest/getstarted/

2. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deepseek-robot-trainer
   cd deepseek-robot-trainer
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify Gazebo installation:
   ```bash
   gz sim shapes.sdf
   ```
   You should see the Gazebo simulator launch with some basic shapes.

## Usage
   ```bash
   python train.py --dataset_id /path/to/model --chk_path /path/to/checkpoints
   ```

## Training Pipeline

1. Loads the DeepSeek model with LoRA adapters
2. Generates robot designs in SDF format
3. Validates designs using Gazebo simulator
4. Uses simulation success/failure as reward signals
5. Updates the model using GRPO (Generative Reinforcement Policy Optimization)

## Reward Function

The reward function evaluates generated SDF code by:
1. Extracting code between <answer> tags
2. Writing to a temporary SDF file
3. Running Gazebo simulation
4. Checking for simulation errors
5. Returning +1 for successful simulations, -1 for failures

