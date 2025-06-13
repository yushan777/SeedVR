# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import os
import torch
import mediapy
from einops import rearrange
from omegaconf import OmegaConf
print(os.getcwd())
import datetime
from tqdm import tqdm
from models.dit import na
import gc

from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
if os.path.exists("./projects/video_diffusion_sr/color_fix.py"):
    from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
    use_colorfix=True
else:
    use_colorfix = False
    print('Note!!!!!! Color fix is not available!')
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.io.video import read_video
import argparse

# Import single GPU optimized functions
from common.distributed.single_gpu import (
    get_device,
    init_torch,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_sequence_parallel,
    sync_data,
)

from projects.video_diffusion_sr.infer import VideoDiffusionInfer
from common.config import load_config
from common.seed import set_seed


def configure_sequence_parallel(sp_size):
    """Single GPU version - no sequence parallel needed."""
    pass


def configure_runner():
    """Configure runner for single GPU inference."""
    config_path = os.path.join('./configs_7b', 'main_single_gpu.yaml')
    config = load_config(config_path)
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)
    
    # Single GPU initialization - no distributed setup
    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    
    # Load model directly to GPU
    runner.configure_dit_model(device="cuda", checkpoint='./ckpts/seedvr_ema_7b.pth')
    runner.configure_vae_model()
    
    # Set memory limit
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    
    return runner


def generation_step(runner, text_embeds_dict, cond_latents):
    """Single GPU generation step - simplified without distributed operations."""
    
    # Generate noise tensors
    noises = [torch.randn_like(latent) for latent in cond_latents]
    aug_noises = [torch.randn_like(latent) for latent in cond_latents]
    print(f"Generating with noise shape: {noises[0].size()}.")
    
    # Move to GPU (no sync needed for single GPU)
    device = get_device()
    noises = [noise.to(device) for noise in noises]
    aug_noises = [aug_noise.to(device) for aug_noise in aug_noises]
    cond_latents = [latent.to(device) for latent in cond_latents]
    
    cond_noise_scale = 0.1

    def _add_noise(x, aug_noise):
        t = torch.tensor([1000.0], device=device) * cond_noise_scale
        shape = torch.tensor(x.shape[1:], device=device)[None]
        t = runner.timestep_transform(t, shape)
        print(f"Timestep shifting from {1000.0 * cond_noise_scale} to {t}.")
        x = runner.schedule.forward(x, aug_noise, t)
        return x

    conditions = [
        runner.get_condition(
            noise,
            task="sr",
            latent_blur=_add_noise(latent_blur, aug_noise),
        )
        for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
    ]

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        video_tensors = runner.inference(
            noises=noises,
            conditions=conditions,
            dit_offload=True,
            **text_embeds_dict,
        )

    samples = [
        (
            rearrange(video[:, None], "c t h w -> t c h w")
            if video.ndim == 3
            else rearrange(video, "c t h w -> t c h w")
        )
        for video in video_tensors
    ]
    del video_tensors

    return samples


def generation_loop(runner, video_path='./test_videos', output_dir='./results', 
                   batch_size=1, cfg_scale=6.5, cfg_rescale=0.0, sample_steps=50, 
                   seed=666, res_h=1280, res_w=720):
    """Single GPU generation loop - simplified without distributed partitioning."""

    def _build_pos_and_neg_prompt():
        positive_text = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, \
        hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, \
        skin pore detailing, hyper sharpness, perfect without deformations."
        negative_text = "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, \
        CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, \
        signature, jpeg artifacts, deformed, lowres, over-smooth"
        return positive_text, negative_text

    def _build_test_prompts(video_path):
        positive_text, negative_text = _build_pos_and_neg_prompt()
        original_videos = []
        prompts = {}
        video_list = os.listdir(video_path)
        for f in video_list:
            if f.endswith(".mp4"):
                original_videos.append(f)
                prompts[f] = positive_text
        print(f"Total prompts to be generated: {len(original_videos)}")
        return original_videos, prompts, negative_text

    def _extract_text_embeds(original_videos_local):
        """Extract text embeddings for single GPU."""
        positive_prompts_embeds = []
        for videos_batch in tqdm(original_videos_local):
            text_pos_embeds = torch.load('pos_emb.pt')
            text_neg_embeds = torch.load('neg_emb.pt')

            positive_prompts_embeds.append(
                {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
            )
        gc.collect()
        torch.cuda.empty_cache()
        return positive_prompts_embeds

    def cut_videos_single_gpu(videos):
        """Single GPU version - no sequence parallel cutting needed."""
        return videos

    def partition_videos_for_batch(videos, batch_size):
        """Partition videos into batches for single GPU processing."""
        batches = []
        for i in range(0, len(videos), batch_size):
            batches.append(videos[i:i + batch_size])
        return batches

    # Configure diffusion parameters
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = cfg_rescale
    runner.config.diffusion.timesteps.sampling.steps = sample_steps
    runner.configure_diffusion()

    # Set random seed
    set_seed(seed, same_across_ranks=True)
    os.makedirs(output_dir, exist_ok=True)

    # Get test prompts
    original_videos, _, _ = _build_test_prompts(video_path)
    
    # Partition videos into batches (no distributed partitioning needed)
    original_videos_local = partition_videos_for_batch(original_videos, batch_size)

    # Pre-extract text embeddings
    positive_prompts_embeds = _extract_text_embeds(original_videos_local)

    video_transform = Compose([
        NaResize(
            resolution=(res_h * res_w) ** 0.5,
            mode="area",
            downsample_only=False,
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])

    # Generation loop
    for videos, text_embeds in tqdm(zip(original_videos_local, positive_prompts_embeds)):
        # Read condition latents
        cond_latents = []
        for video in videos:
            video = (
                read_video(
                   os.path.join(video_path, video), output_format="TCHW"
                )[0]
                / 255.0
            )
            print(f"Read video size: {video.size()}")
            cond_latents.append(video_transform(video.to(get_device())))

        ori_lengths = [video.size(1) for video in cond_latents]
        input_videos = cond_latents
        
        # No sequence parallel cutting needed for single GPU
        cond_latents = [cut_videos_single_gpu(video) for video in cond_latents]

        # Encode videos with VAE
        runner.dit.to("cpu")
        print(f"Encoding videos: {list(map(lambda x: x.size(), cond_latents))}")
        runner.vae.to(get_device())
        cond_latents = runner.vae_encode(cond_latents)
        runner.vae.to("cpu")
        runner.dit.to(get_device())

        # Move text embeddings to GPU
        for i, emb in enumerate(text_embeds["texts_pos"]):
            text_embeds["texts_pos"][i] = emb.to(get_device())
        for i, emb in enumerate(text_embeds["texts_neg"]):
            text_embeds["texts_neg"][i] = emb.to(get_device())

        # Generate samples
        samples = generation_step(runner, text_embeds, cond_latents=cond_latents)
        runner.dit.to("cpu")
        del cond_latents

        # Save results (no sequence parallel rank check needed)
        for path, input, sample, ori_length in zip(videos, input_videos, samples, ori_lengths):
            if ori_length < sample.shape[0]:
                sample = sample[:ori_length]
            filename = os.path.join(output_dir, os.path.basename(path))
            
            # Color fix
            input = (
                rearrange(input[:, None], "c t h w -> t c h w")
                if input.ndim == 3
                else rearrange(input, "c t h w -> t c h w")
            )
            if use_colorfix:
                sample = wavelet_reconstruction(
                    sample.to("cpu"), input[: sample.size(0)].to("cpu")
                )
            else:
                sample = sample.to("cpu")
            
            sample = (
                rearrange(sample[:, None], "t c h w -> t h w c")
                if sample.ndim == 3
                else rearrange(sample, "t c h w -> t h w c")
            )
            sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
            sample = sample.to(torch.uint8).numpy()

            if sample.shape[0] == 1:
                mediapy.write_image(filename, sample.squeeze(0))
            else:
                mediapy.write_video(filename, sample, fps=24)
        
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--video_path", type=str, default="./test_videos")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--cfg_scale", type=float, default=6.5)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    print("Starting single GPU inference for SeedVR 7B...")
    runner = configure_runner()
    generation_loop(runner, **vars(args))
    print("Single GPU inference completed!")
