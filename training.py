#!/usr/bin/env python
import os
import argparse
import logging
import warnings
import subprocess
import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig
from datasets import Dataset  # using an in-memory dataset for robot designs
from trl import GRPOConfig, GRPOTrainer
from fsdp_utils import get_args_parser
from fsdp_utils import AppState
from torch.utils.data import DataLoader

# Import the atomic saving utilities from your helper file.
from cycling_utils import atomic_torch_save, AtomicDirectory

# Suppress warnings.
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    args = get_args_parser().parse_args()
    
    # Set up a single device (GPU if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("/data", args.dataset_id)
    
    # Load tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=False,
        torch_dtype=torch.bfloat16
    ).to(device)
    print(f"Loaded model on device: {device}")
    
    # Inject LoRA modules.
    ADAPTER_NAME = "ExampleLora"
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0  # zero dropout for deterministic reward comparisons
    )
    model = LoraModel(model, lora_config, ADAPTER_NAME)
    
    # Define the conversation meta-template.
    conversation_template = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think><answer> answer here </answer>. "
        "User: {} Assistant: <think>"
    )
    
    # Original robot design prompts.
    original_prompts = [
        "Write SDF code for the Gazebo simulator for a humanoid robot with two arms, and two legs. The answer should include only code for a full SDF (Simulation Description Format) file.",
        "Write SDF code for the Gazebo simulator for a quadruped robot for rough terrain navigation. The answer should include only code for a full SDF (Simulation Description Format) file.",
        "Write SDF code for the Gazebo simulator for a flying drone robot with stabilizers for outdoor exploration. The answer should include only code for a full SDF (Simulation Description Format) file.",
        "Write SDF code for the Gazebo simulator for an underwater robot with articulated arms for deep-sea research. The answer should include only code for a full SDF (Simulation Description Format) file."
    ]
    
    # Wrap the original prompts in the conversation meta-template.
    robot_designs = {
        "prompt": [conversation_template.format(p) for p in original_prompts]
    }
    train_dataset = Dataset.from_dict(robot_designs)
    
    def extract_answer(text):
        """Extracts content between <answer> and </answer> tags."""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return match.group(1).strip() if match else ""


    def reward_gazebo(completions, **kwargs):
        rewards = []
        for code in completions:
            extracted_code = extract_answer(code)
            # Save the generated code to shapes.sdf
            sdf_file = "sample.sdf"
            with open(sdf_file, "w") as f:
                f.write(extracted_code)
                print("code:")
                print(extracted_code)
            try:
                # Run the Gazebo simulation using the provided bash script command.
                result = subprocess.run(
                    ["gz", "sim", "-s", sdf_file, "-v", "4"],
                    capture_output=True,
                    text=True,
                    timeout=30  # adjust timeout as needed
                )

                print("Gazebo output:")
                print(result)

                # Check for errors in stderr even if returncode is 0
                error_keywords = ["Error parsing XML", "Unable to read file", "Error Code"]
                stderr_lower = result.stderr.lower()

                if result.returncode == 0 and not any(keyword.lower() in stderr_lower for keyword in error_keywords):
                    rewards.append(1)
                    print("Gazebo simulation succeeded.")
                else:
                    rewards.append(-1)
                    print("Gazebo simulation failed:", result.stderr.strip())
            except Exception as e:
                rewards.append(-1)
                print("Exception during simulation:", str(e))
        return rewards

    
    # Configure GRPO training.
    training_args = GRPOConfig(
        output_dir=args.chk_path,         # where to save checkpoints and logs
        logging_steps=10,
        num_train_epochs=200,
        per_device_train_batch_size=4,
        num_generations=2,
        max_completion_length=1024
    )
    
    # Set up the saving utility.
    saver = AtomicDirectory(output_directory=training_args.output_dir, is_master=True)
    
    # Subclass GRPOTrainer to integrate saving logic.
    class SaveGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, save_every=training_args.logging_steps, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_every = save_every
            self.step = 0
            self.saver = saver
            # Optionally create a DataLoader (if needed for other parts of your training).
            self.dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True
            )

        def training_step(self, *args, **kwargs):
            # Run the standard training step.
            loss = super().training_step(*args, **kwargs)
            self.step += 1
            if self.step % self.save_every == 0:
                checkpoint_directory = self.saver.prepare_checkpoint_directory()
                # Save model state_dict using atomic saving.
                atomic_torch_save(model.state_dict(), os.path.join(checkpoint_directory, "model.pt"))
                self.saver.atomic_symlink(checkpoint_directory)
                print("Saved checkpoint at step", self.step)
            return loss

    # Initialize our custom trainer with the new reward function.
    trainer = SaveGRPOTrainer(
        model=model,                  # model on a single GPU
        reward_funcs=reward_gazebo,   # new custom reward function based on Gazebo logs
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer    # tokenizer for dataset processing
    )
    
    # Begin training.
    trainer.train()


if __name__ == "__main__":
    main()
