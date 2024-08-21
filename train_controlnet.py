import os
import time
import itertools
import torch
import torch.nn.functional as F
import json

from pathlib import Path
from tqdm import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from octa_datasets import OCTA_Dataset

class ControlNet_Trainer:
    def __init__(self, config_file="config_boundary_unified_3M_FULL.json"):
        self.image_root_path = "datasets/OCTA-500"
        self.image_root_path = "datasets/ROSE"
        conds = "_".join(config_file[:-5].split("_")[1:])
        self.output_dir = "outputs/controlnet_octa500_{}".format(conds)
        self.config_file = "prompts/{}".format(config_file)
        
        self.pretrained_model_name_or_path = "./models/stable_diffusion_v1_5"
        self.controlnet_model_name_or_path = "./models/lllyasviel/control_v11f1p_sd15_depth"

        self.total_training_steps = 3000
        self.batch_size = 8
        self.save_every_steps = 200

        logging_dir = Path(self.output_dir, "logs")
        
        accelerator_project_config = ProjectConfiguration(project_dir=self.output_dir, logging_dir=logging_dir)

        self.accelerator = Accelerator(
            mixed_precision=None,
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )

        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)

        self.load_models()
        self.training_prepare()


        self.control_prompts = []
        self.text_prompts = []

        with open("{}/{}".format(self.image_root_path, self.config_file), 'r') as file:
            for item in json.load(file)[:5]:
                control_prompt = Image.open("{}/{}".format(self.image_root_path, item["source"])).resize((512, 512))
                self.control_prompts.append(control_prompt)
                self.text_prompts.append(item["prompt"])
    
    def load_models(self):
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="unet")
        self.controlnet = ControlNetModel.from_pretrained(self.controlnet_model_name_or_path)
        self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder")

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.train()
    
    
    def training_prepare(self):
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.controlnet.to(self.accelerator.device, dtype=self.weight_dtype)

        self.optimizer = torch.optim.AdamW(self.controlnet.parameters(), lr=5e-5, weight_decay=1e-2)
        # self.optimizer = Prodigy(self.controlnet.parameters(), lr=1)

        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.optimizer,
            num_warmup_steps=500 * self.accelerator.num_processes,
            num_training_steps= 100000 * self.accelerator.num_processes,
            num_cycles=1,
            power=1.0,
        )

        train_dataset = OCTA_Dataset(self.config_file, size=512, image_root_path=self.image_root_path)

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )

        key_modules = self.unet, self.controlnet, self.lr_scheduler, self.optimizer, self.train_dataloader
        self.unet, self.controlnet, self.lr_scheduler, self.optimizer, self.train_dataloader = self.accelerator.prepare(key_modules)
    
    def collate_fn(self, data):
        images = torch.stack([example["image"] for example in data])
        def get_text_input_ids(text):
            text_input_ids = self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            return text_input_ids

        text_input_ids = torch.stack([get_text_input_ids(example["prompt"]) for example in data])
        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in data])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

        clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
        drop_image_embeds = [example["drop_image_embed"] for example in data]

        return {
            "images": images,
            "clip_images": clip_images,
            "conditioning_pixel_values": conditioning_pixel_values,
            "input_ids": text_input_ids,
            "drop_image_embeds": drop_image_embeds
        }
    
    def image_grid(self, imgs, rows, cols):
        assert len(imgs) == rows*cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size
        
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid
    
    def evaluate(self, step=0):
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            controlnet=self.controlnet,
            torch_dtype=self.weight_dtype,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            feature_extractor=None,
            safety_checker=None
        ).to(self.accelerator.device)
        
        generator = torch.manual_seed(42)

        grids = []
        for control, text in zip(self.control_prompts, self.text_prompts):
            images = pipe(
                text, num_inference_steps=50, generator=generator, image=control, num_samples=4, seed=42
            ).images
            
            grids.append(self.image_grid([control] + images, len(images) + 1, 1))
        
        grids = self.image_grid(grids, 1, len(grids))
        grids.save("{}/samples-step-{}.png".format(self.output_dir, step))

    def train(self):
        training_steps = 0
        progress_bar = tqdm(
            range(self.total_training_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        epoch = 0
        while True:
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.controlnet):
                    # Convert images to latent space
                    with torch.no_grad():
                        latents = self.vae.encode(batch["images"].to(self.accelerator.device, dtype=self.weight_dtype)).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(self.accelerator.device)
            

                    controlnet_image = batch["conditioning_pixel_values"].to(self.accelerator.device, dtype=self.weight_dtype)
                    encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.accelerator.device, dtype=torch.long), return_dict=False)[0]


                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(self.accelerator.device, dtype=self.weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(self.accelerator.device, dtype=self.weight_dtype),
                        return_dict=False,
                    )[0]

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = self.controlnet.parameters()
                        self.accelerator.clip_grad_norm_(params_to_clip, 1)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    training_steps += 1

                    logs = {
                        "epoch" : epoch,
                        "step_loss": loss.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[0]
                    }

                    self.accelerator.log(logs)
                    progress_bar.set_postfix(**logs)
            
                if training_steps % self.save_every_steps == 0:
                    save_path = os.path.join(self.output_dir, f"checkpoint-step-{training_steps}")
                    self.evaluate(training_steps)
                    self.controlnet.save_pretrained(save_path)
                
            if training_steps > self.total_training_steps: break
                    
            epoch += 1
            

if __name__=="__main__":
    trainer = ControlNet_Trainer()
    trainer.train()