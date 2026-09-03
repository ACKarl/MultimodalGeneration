"""
A2 Multimodal Generative AI Workbench
======================================

This single-file project replaces the original MIDI/LSTM pipeline with an
image-first generative AI system:

Core image workflow
-------------------
1. Train a pixel-space diffusion model from a directory of images.
2. Save a portable checkpoint containing the model, diffusion configuration,
   EMA weights, optimiser state, metrics, and reproducibility metadata.
3. Generate images from an integer random seed.
4. Edit a seed image with SDEdit-style image-to-image diffusion and optional
   inpainting masks.

Extended workflows
------------------
* Temporally coherent image-to-video generation and frame-preserving video edit.
* Multi-GPU DDP, AMP, gradient accumulation, EMA, checkpoint resume, and
  gradient checkpointing for larger training runs.
* Teacher/student diffusion distillation.
* Multimodal manifest production, quality filtering, duplicate detection,
  preference-pair production, and image/multimodal evaluation.
* Lightweight VLM post-training: supervised fine-tuning (SFT), pairwise reward
  modelling (RM), and KL-regularised policy-gradient reinforcement learning.

Minimal examples
----------------
Train:
    python A2_multimodal_generation.py --mode train \
        --data-dir training_images --model-file image_diffusion_model.pth

Generate from a deterministic seed:
    python A2_multimodal_generation.py --mode generate \
        --model-file image_diffusion_model.pth --seed 42 \
        --output-file generated_image.png

Edit or inpaint:
    python A2_multimodal_generation.py --mode edit \
        --model-file image_diffusion_model.pth --seed-image Seed.png \
        --mask-file Mask.png --strength 0.65 --output-file edited_image.png

Large-scale training:
    torchrun --standalone --nproc_per_node=4 A2_multimodal_generation.py \
        --mode train --data-dir training_images --mixed-precision \
        --gradient-checkpointing --gradient-accumulation 4

Run ``python A2_multimodal_generation.py --help`` for all workflows.

Required packages: torch, Pillow, numpy.
Optional packages: imageio/imageio-ffmpeg for MP4, torchvision for Inception
evaluation, and transformers for CLIPScore.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint as activation_checkpoint


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

DEFAULT_DATA_DIR = "training_images"
DEFAULT_MODEL_FILE = "image_diffusion_model.pth"
DEFAULT_OUTPUT_FILE = "generated_image.png"
DEFAULT_SEED_IMAGE = "Seed.png"
FORMAT_VERSION = 2

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_BICUBIC = getattr(Image, "BICUBIC")
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS")


# -----------------------------------------------------------------------------
# Runtime, reproducibility, and I/O
# -----------------------------------------------------------------------------
@dataclass
class Runtime:
    device: torch.device
    distributed: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_runtime(device_name: str) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("This DDP configuration requires CUDA.")
        torch.cuda.set_device(local_rank)
        backend = (
            "nccl"
            if os.name != "nt" and dist.is_nccl_available()
            else "gloo"
        )
        dist.init_process_group(backend=backend, init_method="env://")
        device = torch.device("cuda", local_rank)
    elif device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return Runtime(device, distributed, rank, local_rank, world_size)


def cleanup_runtime(runtime: Runtime) -> None:
    if runtime.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank_print(runtime: Runtime, *values: Any) -> None:
    if runtime.is_main:
        print(*values, flush=True)


def seed_everything(seed: int, rank: int = 0) -> None:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed % (2**32 - 1))
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def atomic_torch_save(payload: Dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_torch_checkpoint(path: str | Path, map_location: Any) -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint '{path}' was not found.")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def write_json(path: str | Path, value: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"JSONL file '{path}' was not found.")
    records: List[Dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of '{path}': {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Line {line_number} of '{path}' must contain a JSON object."
                )
            records.append(value)
    return records


def find_image_files(data_dir: str | Path, max_files: int = 0) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory '{data_dir}' was not found.")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            f"No supported images were found inside '{data_dir}'."
        )
    return files


# -----------------------------------------------------------------------------
# Image preprocessing and datasets
# -----------------------------------------------------------------------------
def crop_and_resize(
    image: Image.Image,
    image_size: int,
    random_crop: bool,
    horizontal_flip: bool,
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = image_size / min(width, height)
    resized_width = max(image_size, int(round(width * scale)))
    resized_height = max(image_size, int(round(height * scale)))
    image = image.resize((resized_width, resized_height), RESAMPLE_BICUBIC)

    if random_crop:
        left = random.randint(0, max(0, resized_width - image_size))
        top = random.randint(0, max(0, resized_height - image_size))
    else:
        left = max(0, (resized_width - image_size) // 2)
        top = max(0, (resized_height - image_size) // 2)

    image = image.crop((left, top, left + image_size, top + image_size))
    if horizontal_flip and random.random() < 0.5:
        image = ImageOps.mirror(image)
    return image


def pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = crop_and_resize(
        image,
        image_size=image_size,
        random_crop=False,
        horizontal_flip=False,
    )
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def training_pil_to_tensor(
    image: Image.Image,
    image_size: int,
    random_crop: bool,
    horizontal_flip: bool,
) -> torch.Tensor:
    image = crop_and_resize(image, image_size, random_crop, horizontal_flip)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    array = (
        ((tensor + 1.0) * 127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def save_image_batch(images: torch.Tensor, output_file: str | Path) -> List[Path]:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_list = [tensor_to_pil(image) for image in images]
    saved_paths: List[Path] = []

    if len(image_list) == 1:
        image_list[0].save(output_path)
        return [output_path]

    columns = int(math.ceil(math.sqrt(len(image_list))))
    rows = int(math.ceil(len(image_list) / columns))
    width, height = image_list[0].size
    grid = Image.new("RGB", (columns * width, rows * height), color=(0, 0, 0))
    for index, image in enumerate(image_list):
        grid.paste(image, ((index % columns) * width, (index // columns) * height))
        individual = output_path.with_name(
            f"{output_path.stem}_{index:03d}{output_path.suffix}"
        )
        image.save(individual)
        saved_paths.append(individual)
    grid.save(output_path)
    saved_paths.insert(0, output_path)
    return saved_paths


class ImageListDataset(Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        image_size: int,
        training: bool,
        horizontal_flip: bool = True,
    ) -> None:
        self.files = list(files)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.horizontal_flip = bool(horizontal_flip)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        try:
            with Image.open(path) as image:
                return training_pil_to_tensor(
                    image,
                    self.image_size,
                    random_crop=self.training,
                    horizontal_flip=self.training and self.horizontal_flip,
                )
        except Exception as error:
            raise RuntimeError(f"Could not load image '{path}': {error}") from error


def split_image_files(
    files: Sequence[Path],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2 or validation_fraction <= 0:
        return shuffled, []
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def make_loader(
    dataset: Dataset,
    runtime: Runtime,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    sampler: Optional[DistributedSampler] = None
    if runtime.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        shuffle = False

    generator = torch.Generator()
    generator.manual_seed(seed + runtime.rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=runtime.device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    return loader, sampler


# -----------------------------------------------------------------------------
# Diffusion U-Net
# -----------------------------------------------------------------------------
def normalisation_groups(channels: int) -> int:
    groups = min(32, max(1, channels // 4))
    while channels % groups != 0:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * -scale
        )
        values = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((values.sin(), values.cos()), dim=1)
        if self.dimension % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(
            normalisation_groups(input_channels),
            input_channels,
        )
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dimension, output_channels)
        self.norm2 = nn.GroupNorm(
            normalisation_groups(output_channels),
            output_channels,
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(inputs)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        heads = min(heads, channels)
        while channels % heads != 0:
            heads -= 1
        self.heads = heads
        self.norm = nn.GroupNorm(normalisation_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.projection = nn.Conv1d(channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        normalised = self.norm(inputs).reshape(batch, channels, height * width)
        query, key, value = self.qkv(normalised).chunk(3, dim=1)
        head_channels = channels // self.heads

        query = query.reshape(batch, self.heads, head_channels, height * width)
        key = key.reshape(batch, self.heads, head_channels, height * width)
        value = value.reshape(batch, self.heads, head_channels, height * width)

        scale = head_channels**-0.5
        attention = torch.einsum("bhcn,bhcm->bhnm", query * scale, key)
        attention = attention.softmax(dim=-1)
        hidden = torch.einsum("bhnm,bhcm->bhcn", attention, value)
        hidden = hidden.reshape(batch, channels, height * width)
        hidden = self.projection(hidden).reshape(batch, channels, height, width)
        return inputs + hidden


class Downsample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Upsample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        return self.conv(inputs)


class DiffusionUNet(nn.Module):
    """Compact U-Net suitable for 32-256 px educational diffusion training."""

    def __init__(
        self,
        base_channels: int = 64,
        dropout: float = 0.1,
        attention: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        base = int(base_channels)
        time_dimension = base * 4
        self.base_channels = base
        self.dropout_rate = float(dropout)
        self.uses_attention = bool(attention)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base),
            nn.Linear(base, time_dimension),
            nn.SiLU(),
            nn.Linear(time_dimension, time_dimension),
        )
        self.input_conv = nn.Conv2d(3, base, 3, padding=1)

        self.down1a = ResidualBlock(base, base, time_dimension, dropout)
        self.down1b = ResidualBlock(base, base, time_dimension, dropout)
        self.downsample1 = Downsample(base, base * 2)

        self.down2a = ResidualBlock(base * 2, base * 2, time_dimension, dropout)
        self.down2b = ResidualBlock(base * 2, base * 2, time_dimension, dropout)
        self.down2_attention = (
            AttentionBlock(base * 2) if attention else nn.Identity()
        )
        self.downsample2 = Downsample(base * 2, base * 4)

        self.middle1 = ResidualBlock(base * 4, base * 4, time_dimension, dropout)
        self.middle_attention = (
            AttentionBlock(base * 4) if attention else nn.Identity()
        )
        self.middle2 = ResidualBlock(base * 4, base * 4, time_dimension, dropout)

        self.upsample2 = Upsample(base * 4, base * 2)
        self.up2a = ResidualBlock(base * 4, base * 2, time_dimension, dropout)
        self.up2b = ResidualBlock(base * 2, base * 2, time_dimension, dropout)
        self.up2_attention = (
            AttentionBlock(base * 2) if attention else nn.Identity()
        )

        self.upsample1 = Upsample(base * 2, base)
        self.up1a = ResidualBlock(base * 2, base, time_dimension, dropout)
        self.up1b = ResidualBlock(base, base, time_dimension, dropout)
        self.output_norm = nn.GroupNorm(normalisation_groups(base), base)
        self.output_conv = nn.Conv2d(base, 3, 3, padding=1)

    def _run(self, module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return activation_checkpoint(
                module,
                *inputs,
                use_reentrant=False,
            )
        return module(*inputs)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)
        hidden = self.input_conv(noisy_images)

        hidden = self._run(self.down1a, hidden, time_embedding)
        skip1 = self._run(self.down1b, hidden, time_embedding)
        hidden = self.downsample1(skip1)

        hidden = self._run(self.down2a, hidden, time_embedding)
        hidden = self._run(self.down2b, hidden, time_embedding)
        skip2 = self._run(self.down2_attention, hidden)
        hidden = self.downsample2(skip2)

        hidden = self._run(self.middle1, hidden, time_embedding)
        hidden = self._run(self.middle_attention, hidden)
        hidden = self._run(self.middle2, hidden, time_embedding)

        hidden = self.upsample2(hidden)
        hidden = torch.cat((hidden, skip2), dim=1)
        hidden = self._run(self.up2a, hidden, time_embedding)
        hidden = self._run(self.up2b, hidden, time_embedding)
        hidden = self._run(self.up2_attention, hidden)

        hidden = self.upsample1(hidden)
        hidden = torch.cat((hidden, skip1), dim=1)
        hidden = self._run(self.up1a, hidden, time_embedding)
        hidden = self._run(self.up1b, hidden, time_embedding)
        return self.output_conv(F.silu(self.output_norm(hidden)))


# -----------------------------------------------------------------------------
# Gaussian diffusion and DDIM sampling
# -----------------------------------------------------------------------------
def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "linear":
        scale = 1000.0 / timesteps
        betas = torch.linspace(
            scale * 0.0001,
            scale * 0.02,
            timesteps,
            dtype=torch.float64,
        )
        return betas.clamp(max=0.999).float()

    if schedule == "cosine":
        offset = 0.008
        points = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos(
            ((points / timesteps + offset) / (1 + offset)) * math.pi * 0.5
        ) ** 2
        cumulative = cumulative / cumulative[0]
        betas = 1.0 - cumulative[1:] / cumulative[:-1]
        return betas.clamp(0.000001, 0.999).float()

    raise ValueError(f"Unsupported beta schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "epsilon",
    ) -> None:
        super().__init__()
        betas = make_beta_schedule(beta_schedule, timesteps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = F.pad(cumulative[:-1], (1, 0), value=1.0)

        self.timesteps = int(timesteps)
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("alphas_cumprod_prev", previous)
        self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            (1.0 - cumulative).sqrt(),
        )

    @staticmethod
    def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        extracted = values.gather(0, timesteps.long())
        return extracted.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_images)
        sqrt_alpha = self.extract(
            self.sqrt_alphas_cumprod,
            timesteps,
            clean_images.shape,
        )
        sqrt_one_minus = self.extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            clean_images.shape,
        )
        return sqrt_alpha * clean_images + sqrt_one_minus * noise

    def training_target(
        self,
        clean_images: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "x0":
            return clean_images
        if self.prediction_type == "v":
            sqrt_alpha = self.extract(
                self.sqrt_alphas_cumprod,
                timesteps,
                clean_images.shape,
            )
            sqrt_one_minus = self.extract(
                self.sqrt_one_minus_alphas_cumprod,
                timesteps,
                clean_images.shape,
            )
            return sqrt_alpha * noise - sqrt_one_minus * clean_images
        raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

    def model_predictions(
        self,
        model: nn.Module,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_output = model(noisy_images, timesteps)
        sqrt_alpha = self.extract(
            self.sqrt_alphas_cumprod,
            timesteps,
            noisy_images.shape,
        )
        sqrt_one_minus = self.extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            noisy_images.shape,
        )

        if self.prediction_type == "epsilon":
            predicted_noise = model_output
            predicted_clean = (
                noisy_images - sqrt_one_minus * predicted_noise
            ) / sqrt_alpha.clamp_min(1e-8)
        elif self.prediction_type == "x0":
            predicted_clean = model_output
            predicted_noise = (
                noisy_images - sqrt_alpha * predicted_clean
            ) / sqrt_one_minus.clamp_min(1e-8)
        elif self.prediction_type == "v":
            predicted_clean = sqrt_alpha * noisy_images - sqrt_one_minus * model_output
            predicted_noise = sqrt_one_minus * noisy_images + sqrt_alpha * model_output
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

        return predicted_noise, predicted_clean.clamp(-1.0, 1.0)

    def loss(
        self,
        model: nn.Module,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_images)
        noisy = self.q_sample(clean_images, timesteps, noise)
        prediction = model(noisy, timesteps)
        target = self.training_target(clean_images, noise, timesteps)
        return F.mse_loss(prediction.float(), target.float())

    def ddim_step(
        self,
        model: nn.Module,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        previous_timestep: int,
        eta: float,
    ) -> torch.Tensor:
        predicted_noise, predicted_clean = self.model_predictions(
            model,
            images,
            timesteps,
        )
        alpha_current = self.extract(
            self.alphas_cumprod,
            timesteps,
            images.shape,
        )

        if previous_timestep < 0:
            return predicted_clean

        previous = torch.full_like(timesteps, previous_timestep)
        alpha_previous = self.extract(
            self.alphas_cumprod,
            previous,
            images.shape,
        )
        sigma = eta * torch.sqrt(
            (
                (1.0 - alpha_previous)
                / (1.0 - alpha_current).clamp_min(1e-8)
                * (1.0 - alpha_current / alpha_previous.clamp_min(1e-8))
            ).clamp_min(0.0)
        )
        direction = torch.sqrt(
            (1.0 - alpha_previous - sigma.square()).clamp_min(0.0)
        ) * predicted_noise
        random_noise = torch.randn_like(images) if eta > 0 else 0.0
        return (
            alpha_previous.sqrt() * predicted_clean
            + direction
            + sigma * random_noise
        )


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in unwrap_model(model).state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = unwrap_model(model).state_dict()
        for name, value in state.items():
            if name not in self.shadow:
                self.shadow[name] = value.detach().clone()
            elif value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(),
                    alpha=1.0 - self.decay,
                )
            else:
                self.shadow[name].copy_(value)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = state["shadow"]


def model_config_from_args(args: argparse.Namespace, base_channels: Optional[int] = None) -> Dict[str, Any]:
    return {
        "base_channels": int(
            args.base_channels if base_channels is None else base_channels
        ),
        "dropout": float(args.dropout),
        "attention": bool(args.attention),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
    }


def diffusion_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "timesteps": int(args.diffusion_steps),
        "beta_schedule": args.beta_schedule,
        "prediction_type": args.prediction_type,
    }


def build_diffusion_checkpoint(
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[Any],
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    training_file_count: int,
    model_config: Optional[Dict[str, Any]] = None,
    diffusion_config: Optional[Dict[str, Any]] = None,
    distillation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_state = {
        name: value.detach().cpu()
        for name, value in unwrap_model(model).state_dict().items()
    }
    ema_state = {
        name: value.detach().cpu()
        for name, value in ema.shadow.items()
    }
    payload: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "task": "image_diffusion",
        "model_config": model_config or model_config_from_args(args),
        "diffusion_config": diffusion_config or diffusion_config_from_args(args),
        "image_size": int(args.image_size),
        "model_state_dict": ema_state,
        "raw_model_state_dict": raw_state,
        "ema_state_dict": {"decay": ema.decay, "shadow": ema_state},
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_loss": float(best_validation_loss),
        "training_file_count": int(training_file_count),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "reproducibility": {
            "seed": int(args.seed),
            "torch_version": torch.__version__,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        },
        "capabilities": [
            "image_generation",
            "image_editing",
            "video_generation",
            "video_editing",
            "distributed_training",
            "multimodal_evaluation",
            "diffusion_distillation",
        ],
    }
    if distillation is not None:
        payload["distillation"] = distillation
    return payload


def load_diffusion_pipeline(
    model_file: str | Path,
    device: torch.device,
    gradient_checkpointing: bool = False,
) -> Tuple[DiffusionUNet, GaussianDiffusion, Dict[str, Any]]:
    checkpoint = load_torch_checkpoint(model_file, map_location="cpu")
    if checkpoint.get("task") != "image_diffusion":
        raise ValueError(
            f"'{model_file}' is not an image diffusion checkpoint."
        )
    config = dict(checkpoint["model_config"])
    config["gradient_checkpointing"] = gradient_checkpointing
    model = DiffusionUNet(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    diffusion = GaussianDiffusion(**checkpoint["diffusion_config"]).to(device)
    return model, diffusion, checkpoint


# -----------------------------------------------------------------------------
# Diffusion training, generation, editing, and video
# -----------------------------------------------------------------------------
def reduce_loss(
    total_loss: float,
    sample_count: int,
    runtime: Runtime,
) -> float:
    values = torch.tensor(
        [total_loss, float(sample_count)],
        dtype=torch.float64,
        device=runtime.device,
    )
    if runtime.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if values[1].item() == 0:
        return math.inf
    return float((values[0] / values[1]).item())


@torch.no_grad()
def validate_diffusion(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    loader: Optional[DataLoader],
    runtime: Runtime,
    mixed_precision: bool,
) -> float:
    if loader is None:
        return math.inf
    model.eval()
    total_loss = 0.0
    sample_count = 0
    for clean_images in loader:
        clean_images = clean_images.to(runtime.device, non_blocking=True)
        timesteps = torch.randint(
            0,
            diffusion.timesteps,
            (clean_images.shape[0],),
            device=runtime.device,
        )
        with torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.float16,
            enabled=mixed_precision and runtime.device.type == "cuda",
        ):
            loss = diffusion.loss(model, clean_images, timesteps)
        total_loss += float(loss.item()) * clean_images.shape[0]
        sample_count += clean_images.shape[0]
    return reduce_loss(total_loss, sample_count, runtime)


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, float(step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def make_gradient_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_diffusion(args: argparse.Namespace, runtime: Runtime) -> None:
    seed_everything(args.seed, runtime.rank)
    files = find_image_files(args.data_dir, args.max_files)
    training_files, validation_files = split_image_files(
        files,
        args.validation_fraction,
        args.seed,
    )
    training_dataset = ImageListDataset(
        training_files,
        args.image_size,
        training=True,
        horizontal_flip=args.horizontal_flip,
    )
    validation_dataset = (
        ImageListDataset(
            validation_files,
            args.image_size,
            training=False,
            horizontal_flip=False,
        )
        if validation_files
        else None
    )
    training_loader, training_sampler = make_loader(
        training_dataset,
        runtime,
        args.batch_size,
        args.workers,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader: Optional[DataLoader] = None
    if validation_dataset is not None:
        validation_loader, _ = make_loader(
            validation_dataset,
            runtime,
            args.batch_size,
            args.workers,
            shuffle=False,
            seed=args.seed,
        )

    model = DiffusionUNet(**model_config_from_args(args)).to(runtime.device)
    if runtime.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
        )
    diffusion = GaussianDiffusion(**diffusion_config_from_args(args)).to(
        runtime.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    updates_per_epoch = max(
        1,
        math.ceil(len(training_loader) / args.gradient_accumulation),
    )
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        total_steps=updates_per_epoch * args.epochs,
        warmup_steps=args.warmup_steps,
    )
    scaler = make_gradient_scaler(
        args.mixed_precision and runtime.device.type == "cuda"
    )
    ema = ExponentialMovingAverage(model, args.ema_decay)
    start_epoch = 1
    global_step = 0
    best_validation_loss = math.inf

    if args.resume:
        checkpoint = load_torch_checkpoint(args.resume, map_location="cpu")
        if checkpoint.get("task") != "image_diffusion":
            raise ValueError("The resume file is not an image diffusion checkpoint.")
        unwrap_model(model).load_state_dict(
            checkpoint.get(
                "raw_model_state_dict",
                checkpoint["model_state_dict"],
            ),
            strict=True,
        )
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if checkpoint.get("ema_state_dict"):
            ema.load_state_dict(checkpoint["ema_state_dict"])
            ema.shadow = {
                name: value.to(runtime.device)
                for name, value in ema.shadow.items()
            }
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", math.inf)
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    rank_print(runtime, f"Device: {runtime.device}")
    rank_print(runtime, f"World size: {runtime.world_size}")
    rank_print(runtime, f"Training images: {len(training_files)}")
    rank_print(runtime, f"Validation images: {len(validation_files)}")
    rank_print(runtime, f"Trainable parameters: {parameter_count:,}")

    for epoch in range(start_epoch, args.epochs + 1):
        if training_sampler is not None:
            training_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_start = time.time()

        for batch_index, clean_images in enumerate(training_loader):
            clean_images = clean_images.to(runtime.device, non_blocking=True)
            timesteps = torch.randint(
                0,
                diffusion.timesteps,
                (clean_images.shape[0],),
                device=runtime.device,
            )
            should_update = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(training_loader)
            )
            synchronisation = (
                model.no_sync()
                if runtime.distributed and not should_update
                else contextlib.nullcontext()
            )
            with synchronisation:
                with torch.autocast(
                    device_type=runtime.device.type,
                    dtype=torch.float16,
                    enabled=(
                        args.mixed_precision and runtime.device.type == "cuda"
                    ),
                ):
                    loss = diffusion.loss(model, clean_images, timesteps)
                    scaled_loss = loss / args.gradient_accumulation
                scaler.scale(scaled_loss).backward()

            epoch_loss += float(loss.detach().item()) * clean_images.shape[0]
            epoch_samples += clean_images.shape[0]

            if should_update:
                scaler.unscale_(optimizer)
                if args.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                global_step += 1

                if runtime.is_main and global_step % args.log_every == 0:
                    print(
                        f"Epoch {epoch:>3}/{args.epochs} | "
                        f"step {global_step:>7} | "
                        f"loss {loss.item():.5f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

        training_loss = reduce_loss(epoch_loss, epoch_samples, runtime)
        validation_loss = validate_diffusion(
            model,
            diffusion,
            validation_loader,
            runtime,
            args.mixed_precision,
        )
        selection_loss = (
            validation_loss if math.isfinite(validation_loss) else training_loss
        )
        elapsed = time.time() - epoch_start

        rank_print(
            runtime,
            f"Epoch {epoch:>3}/{args.epochs} completed in {elapsed:.1f}s | "
            f"train {training_loss:.5f} | "
            f"validation "
            f"{validation_loss:.5f}" if math.isfinite(validation_loss)
            else f"Epoch {epoch:>3}/{args.epochs} completed in {elapsed:.1f}s | "
            f"train {training_loss:.5f} | validation n/a",
        )

        improved = selection_loss < best_validation_loss
        if improved:
            best_validation_loss = selection_loss

        should_checkpoint = (
            improved
            or epoch == args.epochs
            or epoch % args.checkpoint_every == 0
        )
        if runtime.is_main and should_checkpoint:
            payload = build_diffusion_checkpoint(
                model,
                ema,
                optimizer,
                scheduler,
                scaler,
                args,
                epoch,
                global_step,
                best_validation_loss,
                len(files),
            )
            atomic_torch_save(payload, args.model_file)
            print(f"Saved checkpoint: {args.model_file}", flush=True)

        if runtime.distributed:
            dist.barrier()

    rank_print(runtime, "Diffusion training completed.")


def sampling_timesteps(
    start_timestep: int,
    sampling_steps: int,
) -> List[int]:
    sampling_steps = max(1, min(sampling_steps, start_timestep + 1))
    values = np.linspace(start_timestep, 0, sampling_steps)
    rounded = [int(round(value)) for value in values]
    unique: List[int] = []
    for value in rounded:
        if not unique or value != unique[-1]:
            unique.append(value)
    if unique[-1] != 0:
        unique.append(0)
    return unique


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    initial_images: torch.Tensor,
    start_timestep: int,
    sampling_steps: int,
    eta: float,
    source_images: Optional[torch.Tensor] = None,
    edit_mask: Optional[torch.Tensor] = None,
    source_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    model.eval()
    images = initial_images
    steps = sampling_timesteps(start_timestep, sampling_steps)

    for index, current in enumerate(steps):
        previous = steps[index + 1] if index + 1 < len(steps) else -1
        timestep_batch = torch.full(
            (images.shape[0],),
            current,
            dtype=torch.long,
            device=images.device,
        )
        images = diffusion.ddim_step(
            model,
            images,
            timestep_batch,
            previous,
            eta,
        )

        if source_images is not None and edit_mask is not None:
            if previous >= 0:
                previous_batch = torch.full_like(timestep_batch, previous)
                preserved = diffusion.q_sample(
                    source_images,
                    previous_batch,
                    source_noise,
                )
            else:
                preserved = source_images
            images = edit_mask * images + (1.0 - edit_mask) * preserved

    return images.clamp(-1.0, 1.0)


def generate_images(args: argparse.Namespace, runtime: Runtime) -> None:
    seed_everything(args.seed, runtime.rank)
    model, diffusion, checkpoint = load_diffusion_pipeline(
        args.model_file,
        runtime.device,
    )
    image_size = int(checkpoint["image_size"])
    initial_noise = torch.randn(
        args.num_images,
        3,
        image_size,
        image_size,
        device=runtime.device,
    )
    images = ddim_sample(
        model,
        diffusion,
        initial_noise,
        start_timestep=diffusion.timesteps - 1,
        sampling_steps=args.sampling_steps,
        eta=args.eta,
    )
    if runtime.is_main:
        saved = save_image_batch(images, args.output_file)
        print(f"Generated {args.num_images} image(s) from seed {args.seed}.")
        print(f"Output: {saved[0]}")


def load_mask(mask_file: str | Path, image_size: int) -> torch.Tensor:
    with Image.open(mask_file) as mask:
        mask = ImageOps.exif_transpose(mask).convert("L")
        mask = ImageOps.fit(
            mask,
            (image_size, image_size),
            method=RESAMPLE_LANCZOS,
            centering=(0.5, 0.5),
        )
        array = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def edit_image(args: argparse.Namespace, runtime: Runtime) -> None:
    if not Path(args.seed_image).exists():
        raise FileNotFoundError(f"Seed image '{args.seed_image}' was not found.")
    seed_everything(args.seed, runtime.rank)
    model, diffusion, checkpoint = load_diffusion_pipeline(
        args.model_file,
        runtime.device,
    )
    image_size = int(checkpoint["image_size"])
    with Image.open(args.seed_image) as image:
        source = pil_to_tensor(image, image_size)
    source = source.unsqueeze(0).repeat(args.num_images, 1, 1, 1)
    source = source.to(runtime.device)
    noise = torch.randn_like(source)
    start_timestep = max(
        1,
        min(
            diffusion.timesteps - 1,
            int(round(args.strength * (diffusion.timesteps - 1))),
        ),
    )
    timestep_batch = torch.full(
        (source.shape[0],),
        start_timestep,
        dtype=torch.long,
        device=runtime.device,
    )
    initial = diffusion.q_sample(source, timestep_batch, noise)

    mask: Optional[torch.Tensor] = None
    if args.mask_file:
        if not Path(args.mask_file).exists():
            raise FileNotFoundError(f"Mask image '{args.mask_file}' was not found.")
        loaded_mask = load_mask(args.mask_file, image_size)
        mask_tensor = loaded_mask.unsqueeze(0).repeat(
            args.num_images,
            1,
            1,
            1,
        )
        mask_tensor = mask_tensor.to(runtime.device)
        initial = mask_tensor * initial + (1.0 - mask_tensor) * source
        mask = mask_tensor

    images = ddim_sample(
        model,
        diffusion,
        initial,
        start_timestep=start_timestep,
        sampling_steps=min(args.sampling_steps, start_timestep + 1),
        eta=args.eta,
        source_images=source if mask is not None else None,
        edit_mask=mask,
        source_noise=noise if mask is not None else None,
    )
    if runtime.is_main:
        saved = save_image_batch(images, args.output_file)
        operation = "inpainted" if mask is not None else "edited"
        print(
            f"{operation.capitalize()} {args.num_images} image(s) "
            f"with strength {args.strength:.2f}."
        )
        print(f"Output: {saved[0]}")


def temporally_correlated_noise(
    frame_count: int,
    image_size: int,
    correlation: float,
    motion_x: int,
    motion_y: int,
    device: torch.device,
) -> torch.Tensor:
    correlation = min(max(correlation, 0.0), 0.9999)
    shared = torch.randn(1, 3, image_size, image_size, device=device)
    frames = []
    independent_scale = math.sqrt(max(0.0, 1.0 - correlation**2))
    for index in range(frame_count):
        shifted = torch.roll(
            shared,
            shifts=(index * motion_y, index * motion_x),
            dims=(-2, -1),
        )
        independent = torch.randn_like(shared)
        frames.append(correlation * shifted + independent_scale * independent)
    return torch.cat(frames, dim=0)


def save_video_frames(
    frames: Sequence[Image.Image],
    output_file: str | Path,
    fps: int,
) -> Path:
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".gif":
        duration = max(1, int(round(1000 / fps)))
        frames[0].save(
            destination,
            save_all=True,
            append_images=list(frames[1:]),
            duration=duration,
            loop=0,
        )
        return destination

    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError(
            "MP4 output requires imageio and imageio-ffmpeg. "
            "Install them or choose an output ending in .gif."
        ) from error

    arrays = [np.asarray(frame.convert("RGB")) for frame in frames]
    imageio.mimsave(destination, arrays, fps=fps, macro_block_size=1)
    return destination


def batched_ddim_from_noise(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    all_noise: torch.Tensor,
    batch_size: int,
    sampling_steps: int,
    eta: float,
) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for start in range(0, all_noise.shape[0], batch_size):
        batch = all_noise[start : start + batch_size]
        generated = ddim_sample(
            model,
            diffusion,
            batch,
            diffusion.timesteps - 1,
            sampling_steps,
            eta,
        )
        frames.extend(tensor_to_pil(image) for image in generated)
    return frames


def generate_video(args: argparse.Namespace, runtime: Runtime) -> None:
    seed_everything(args.seed, runtime.rank)
    model, diffusion, checkpoint = load_diffusion_pipeline(
        args.model_file,
        runtime.device,
    )
    image_size = int(checkpoint["image_size"])
    noise = temporally_correlated_noise(
        args.video_frames,
        image_size,
        args.temporal_correlation,
        args.motion_x,
        args.motion_y,
        runtime.device,
    )
    frames = batched_ddim_from_noise(
        model,
        diffusion,
        noise,
        args.video_batch_size,
        args.sampling_steps,
        args.eta,
    )
    if runtime.is_main:
        destination = save_video_frames(frames, args.video_output, args.fps)
        print(
            f"Generated {len(frames)} coherent frames from seed {args.seed}."
        )
        print(f"Output: {destination}")


def read_video_frames(
    video_file: str | Path,
    image_size: int,
    max_frames: int,
) -> List[torch.Tensor]:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError("Video input requires imageio.") from error

    frames: List[torch.Tensor] = []
    reader = imageio.get_reader(str(video_file))
    try:
        for index, array in enumerate(reader):
            if max_frames > 0 and index >= max_frames:
                break
            frames.append(pil_to_tensor(Image.fromarray(array), image_size))
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames could be decoded from '{video_file}'.")
    return frames


def edit_video(args: argparse.Namespace, runtime: Runtime) -> None:
    if not Path(args.seed_video).exists():
        raise FileNotFoundError(f"Seed video '{args.seed_video}' was not found.")
    seed_everything(args.seed, runtime.rank)
    model, diffusion, checkpoint = load_diffusion_pipeline(
        args.model_file,
        runtime.device,
    )
    image_size = int(checkpoint["image_size"])
    source_frames = read_video_frames(
        args.seed_video,
        image_size,
        args.max_video_frames,
    )
    sources = torch.stack(source_frames).to(runtime.device)
    noise = temporally_correlated_noise(
        len(source_frames),
        image_size,
        args.temporal_correlation,
        args.motion_x,
        args.motion_y,
        runtime.device,
    )
    start_timestep = max(
        1,
        min(
            diffusion.timesteps - 1,
            int(round(args.strength * (diffusion.timesteps - 1))),
        ),
    )
    output_frames: List[Image.Image] = []

    for start in range(0, len(source_frames), args.video_batch_size):
        source_batch = sources[start : start + args.video_batch_size]
        noise_batch = noise[start : start + args.video_batch_size]
        timestep_batch = torch.full(
            (source_batch.shape[0],),
            start_timestep,
            dtype=torch.long,
            device=runtime.device,
        )
        initial = diffusion.q_sample(source_batch, timestep_batch, noise_batch)
        generated = ddim_sample(
            model,
            diffusion,
            initial,
            start_timestep,
            min(args.sampling_steps, start_timestep + 1),
            args.eta,
        )
        output_frames.extend(tensor_to_pil(image) for image in generated)

    if runtime.is_main:
        destination = save_video_frames(
            output_frames,
            args.video_output,
            args.fps,
        )
        print(
            f"Edited {len(output_frames)} frames with strength "
            f"{args.strength:.2f}."
        )
        print(f"Output: {destination}")


# -----------------------------------------------------------------------------
# Teacher/student diffusion distillation
# -----------------------------------------------------------------------------
def distill_diffusion(args: argparse.Namespace, runtime: Runtime) -> None:
    seed_everything(args.seed, runtime.rank)
    teacher, diffusion, teacher_checkpoint = load_diffusion_pipeline(
        args.teacher_model_file,
        runtime.device,
    )
    teacher.requires_grad_(False)
    teacher.eval()

    teacher_image_size = int(teacher_checkpoint["image_size"])
    if args.image_size != teacher_image_size:
        rank_print(
            runtime,
            f"Using teacher image size {teacher_image_size} instead of "
            f"requested {args.image_size}.",
        )
        args.image_size = teacher_image_size

    files = find_image_files(args.data_dir, args.max_files)
    dataset = ImageListDataset(
        files,
        args.image_size,
        training=True,
        horizontal_flip=args.horizontal_flip,
    )
    loader, sampler = make_loader(
        dataset,
        runtime,
        args.batch_size,
        args.workers,
        shuffle=True,
        seed=args.seed,
    )
    student_config = model_config_from_args(
        args,
        base_channels=args.student_base_channels,
    )
    student = DiffusionUNet(**student_config).to(runtime.device)
    if runtime.distributed:
        student = DistributedDataParallel(
            student,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
        )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    updates_per_epoch = max(
        1,
        math.ceil(len(loader) / args.gradient_accumulation),
    )
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        updates_per_epoch * args.epochs,
        args.warmup_steps,
    )
    scaler = make_gradient_scaler(
        args.mixed_precision and runtime.device.type == "cuda"
    )
    ema = ExponentialMovingAverage(student, args.ema_decay)
    global_step = 0
    best_loss = math.inf

    teacher_parameters = sum(
        parameter.numel() for parameter in teacher.parameters()
    )
    student_parameters = sum(
        parameter.numel() for parameter in student.parameters()
    )
    rank_print(runtime, f"Teacher parameters: {teacher_parameters:,}")
    rank_print(runtime, f"Student parameters: {student_parameters:,}")
    rank_print(
        runtime,
        f"Compression ratio: {teacher_parameters / student_parameters:.2f}x",
    )

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        sample_count = 0

        for batch_index, clean_images in enumerate(loader):
            clean_images = clean_images.to(runtime.device, non_blocking=True)
            timesteps = torch.randint(
                0,
                diffusion.timesteps,
                (clean_images.shape[0],),
                device=runtime.device,
            )
            noise = torch.randn_like(clean_images)
            noisy_images = diffusion.q_sample(clean_images, timesteps, noise)
            hard_target = diffusion.training_target(
                clean_images,
                noise,
                timesteps,
            )

            with torch.no_grad(), torch.autocast(
                device_type=runtime.device.type,
                dtype=torch.float16,
                enabled=args.mixed_precision and runtime.device.type == "cuda",
            ):
                teacher_target = teacher(noisy_images, timesteps)

            should_update = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            synchronisation = (
                student.no_sync()
                if runtime.distributed and not should_update
                else contextlib.nullcontext()
            )
            with synchronisation:
                with torch.autocast(
                    device_type=runtime.device.type,
                    dtype=torch.float16,
                    enabled=(
                        args.mixed_precision and runtime.device.type == "cuda"
                    ),
                ):
                    student_output = student(noisy_images, timesteps)
                    teacher_loss = F.mse_loss(
                        student_output.float(),
                        teacher_target.float(),
                    )
                    hard_loss = F.mse_loss(
                        student_output.float(),
                        hard_target.float(),
                    )
                    loss = (
                        args.distill_alpha * teacher_loss
                        + (1.0 - args.distill_alpha) * hard_loss
                    )
                    scaled_loss = loss / args.gradient_accumulation
                scaler.scale(scaled_loss).backward()

            epoch_loss += float(loss.detach().item()) * clean_images.shape[0]
            sample_count += clean_images.shape[0]

            if should_update:
                scaler.unscale_(optimizer)
                if args.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        student.parameters(),
                        args.gradient_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(student)
                global_step += 1

        mean_loss = reduce_loss(epoch_loss, sample_count, runtime)
        rank_print(
            runtime,
            f"Distillation epoch {epoch:>3}/{args.epochs} | "
            f"loss {mean_loss:.5f}",
        )
        best_loss = min(best_loss, mean_loss)

        if runtime.is_main and (
            mean_loss <= best_loss
            or epoch == args.epochs
            or epoch % args.checkpoint_every == 0
        ):
            distillation_metadata = {
                "teacher_checkpoint": str(args.teacher_model_file),
                "teacher_parameters": teacher_parameters,
                "student_parameters": student_parameters,
                "compression_ratio": teacher_parameters / student_parameters,
                "teacher_weight": float(args.distill_alpha),
                "hard_target_weight": float(1.0 - args.distill_alpha),
                "recommended_sampling_steps": int(
                    args.student_sampling_steps
                ),
            }
            payload = build_diffusion_checkpoint(
                student,
                ema,
                optimizer,
                scheduler,
                scaler,
                args,
                epoch,
                global_step,
                best_loss,
                len(files),
                model_config=student_config,
                diffusion_config=dict(teacher_checkpoint["diffusion_config"]),
                distillation=distillation_metadata,
            )
            atomic_torch_save(payload, args.distilled_model_file)
            print(f"Saved distilled model: {args.distilled_model_file}")

    rank_print(runtime, "Diffusion distillation completed.")


# -----------------------------------------------------------------------------
# Multimodal data production
# -----------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), RESAMPLE_LANCZOS)
    values = np.asarray(grayscale, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    number = 0
    for bit in bits.flatten():
        number = (number << 1) | int(bit)
    return f"{number:016x}"


def image_quality_metrics(image: Image.Image) -> Dict[str, float]:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    grayscale = rgb.convert("L")
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    brightness = float(array.mean())
    contrast = float(array.std())
    edge_image = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_array = np.asarray(edge_image, dtype=np.float32)
    sharpness = float(edge_array.var())
    histogram = np.histogram(array, bins=64, range=(0.0, 1.0))[0].astype(
        np.float64
    )
    probabilities = histogram / max(histogram.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    exposure_score = max(0.0, 1.0 - abs(brightness - 0.5) / 0.5)
    contrast_score = min(1.0, contrast / 0.20)
    sharpness_score = 1.0 - math.exp(-sharpness / 800.0)
    entropy_score = min(1.0, entropy / 5.5)
    quality_score = (
        0.25 * exposure_score
        + 0.25 * contrast_score
        + 0.30 * sharpness_score
        + 0.20 * entropy_score
    )
    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "entropy": entropy,
        "quality_score": float(quality_score),
    }


def caption_for_image(path: Path, root: Path) -> Tuple[str, str]:
    sidecar = path.with_suffix(".txt")
    if sidecar.exists():
        caption = sidecar.read_text(encoding="utf-8").strip()
        if caption:
            return caption, "sidecar"

    relative_parent = path.parent.relative_to(root)
    if relative_parent.parts:
        label = " ".join(relative_parent.parts).replace("_", " ").replace("-", " ")
        return label.strip(), "directory_label"

    label = path.stem.replace("_", " ").replace("-", " ")
    return label.strip(), "filename"


def build_multimodal_data(args: argparse.Namespace) -> None:
    root = Path(args.data_dir).resolve()
    files = find_image_files(root, args.max_files)
    manifest_path = Path(args.manifest_file).resolve()
    records: List[Dict[str, Any]] = []
    hash_groups: Dict[str, List[str]] = defaultdict(list)

    for index, path in enumerate(files, start=1):
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                quality = image_quality_metrics(image)
                perceptual_hash = difference_hash(image)
            caption, caption_source = caption_for_image(path, root)
            aspect_ratio = max(width / height, height / width)
            rejection_reasons: List[str] = []
            if min(width, height) < args.min_resolution:
                rejection_reasons.append("low_resolution")
            if aspect_ratio > args.max_aspect_ratio:
                rejection_reasons.append("extreme_aspect_ratio")
            if quality["sharpness"] < args.min_sharpness:
                rejection_reasons.append("low_sharpness")

            relative_to_manifest = os.path.relpath(path, manifest_path.parent)
            relative_to_dataset = str(path.relative_to(root))
            semantic_group = str(Path(relative_to_dataset).parent)
            if semantic_group in ("", "."):
                semantic_group = "dataset_root"
            record = {
                "id": hashlib.sha1(
                    relative_to_dataset.encode("utf-8")
                ).hexdigest()[:16],
                "image": relative_to_manifest,
                "relative_path": relative_to_dataset,
                "semantic_group": semantic_group,
                "caption": caption,
                "prompt": "Describe the image.",
                "answer": caption,
                "caption_source": caption_source,
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "sha256": sha256_file(path),
                "dhash": perceptual_hash,
                **quality,
                "accepted": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
            }
            records.append(record)
            hash_groups[perceptual_hash].append(record["id"])
        except Exception as error:
            print(f"[{index}/{len(files)}] skipped {path}: {error}")

    duplicate_group_lookup: Dict[str, List[str]] = {}
    for group in hash_groups.values():
        if len(group) > 1:
            for record_id in group:
                duplicate_group_lookup[record_id] = group

    for record in records:
        record["near_duplicate_ids"] = [
            record_id
            for record_id in duplicate_group_lookup.get(record["id"], [])
            if record_id != record["id"]
        ]

    output_records = (
        records
        if args.include_rejected
        else [record for record in records if record["accepted"]]
    )
    manifest_count = write_jsonl(manifest_path, output_records)

    preference_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in output_records:
        semantic_key = str(
            record.get("semantic_group") or record["caption"]
        ).strip().lower()
        preference_groups[semantic_key].append(record)

    preference_records: List[Dict[str, Any]] = []
    for semantic_key, group in preference_groups.items():
        if len(group) < 2:
            continue
        sorted_group = sorted(group, key=lambda item: item["quality_score"])
        rejected = sorted_group[0]
        chosen = sorted_group[-1]
        if chosen["id"] == rejected["id"]:
            continue
        preference_path = Path(args.preference_file).resolve()
        chosen_absolute = (
            manifest_path.parent / chosen["image"]
        ).resolve()
        rejected_absolute = (
            manifest_path.parent / rejected["image"]
        ).resolve()
        preference_records.append(
            {
                "prompt": (
                    "Select the higher-quality image from semantic group "
                    f"'{semantic_key}'."
                ),
                "chosen_image": os.path.relpath(
                    chosen_absolute,
                    preference_path.parent,
                ),
                "rejected_image": os.path.relpath(
                    rejected_absolute,
                    preference_path.parent,
                ),
                "chosen_score": chosen["quality_score"],
                "rejected_score": rejected["quality_score"],
                "source": "automatic_quality_bootstrap",
            }
        )
    preference_count = write_jsonl(args.preference_file, preference_records)

    accepted_count = sum(bool(record["accepted"]) for record in records)
    duplicate_count = sum(
        bool(record["near_duplicate_ids"]) for record in records
    )
    print(f"Scanned images: {len(records)}")
    print(f"Accepted images: {accepted_count}")
    print(f"Potential duplicates: {duplicate_count}")
    print(f"Manifest records written: {manifest_count}")
    print(f"Preference pairs written: {preference_count}")
    print(f"Manifest: {manifest_path}")


# -----------------------------------------------------------------------------
# Image and multimodal evaluation
# -----------------------------------------------------------------------------
def evaluate_basic_image_set(files: Sequence[Path]) -> Dict[str, float]:
    metrics: List[Dict[str, float]] = []
    low_resolution_features: List[np.ndarray] = []
    for path in files:
        with Image.open(path) as image:
            metrics.append(image_quality_metrics(image))
            small = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGB"),
                (32, 32),
                method=RESAMPLE_BICUBIC,
            )
            vector = np.asarray(small, dtype=np.float32).reshape(-1) / 255.0
            low_resolution_features.append(vector)

    feature_array = np.stack(low_resolution_features)
    if len(feature_array) > 1:
        centred = feature_array - feature_array.mean(axis=1, keepdims=True)
        normalised = centred / (
            np.linalg.norm(centred, axis=1, keepdims=True) + 1e-8
        )
        similarities = normalised @ normalised.T
        upper = similarities[np.triu_indices(len(similarities), k=1)]
        diversity = float(1.0 - upper.mean())
    else:
        diversity = 0.0

    return {
        "count": float(len(files)),
        "mean_quality_score": float(
            np.mean([item["quality_score"] for item in metrics])
        ),
        "mean_sharpness": float(
            np.mean([item["sharpness"] for item in metrics])
        ),
        "mean_brightness": float(
            np.mean([item["brightness"] for item in metrics])
        ),
        "mean_contrast": float(
            np.mean([item["contrast"] for item in metrics])
        ),
        "pixel_diversity": diversity,
    }


class EvaluationImageDataset(Dataset):
    def __init__(self, files: Sequence[Path], transform: Any) -> None:
        self.files = list(files)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as image:
            return self.transform(ImageOps.exif_transpose(image).convert("RGB"))


@torch.no_grad()
def inception_features(
    files: Sequence[Path],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError as error:
        raise RuntimeError(
            "Inception evaluation requires torchvision."
        ) from error

    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()
    model.eval().to(device)
    dataset = EvaluationImageDataset(files, weights.transforms())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    collected: List[np.ndarray] = []
    for images in loader:
        features = model(images.to(device, non_blocking=True))
        if isinstance(features, tuple):
            features = features[0]
        collected.append(features.float().cpu().numpy())
    return np.concatenate(collected, axis=0)


def deterministic_projection(
    features: np.ndarray,
    output_dimension: int,
    seed: int,
) -> np.ndarray:
    if output_dimension <= 0 or output_dimension >= features.shape[1]:
        return features
    generator = np.random.default_rng(seed)
    projection = generator.normal(
        0.0,
        1.0 / math.sqrt(output_dimension),
        size=(features.shape[1], output_dimension),
    )
    return features @ projection


def psd_matrix_square_root(matrix: torch.Tensor) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(0.0).sqrt()
    return (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T


def projected_frechet_distance(
    reference_features: np.ndarray,
    generated_features: np.ndarray,
) -> float:
    reference = torch.from_numpy(reference_features).double()
    generated = torch.from_numpy(generated_features).double()
    mean_reference = reference.mean(dim=0)
    mean_generated = generated.mean(dim=0)
    covariance_reference = torch.cov(reference.T)
    covariance_generated = torch.cov(generated.T)
    sqrt_reference = psd_matrix_square_root(covariance_reference)
    middle = sqrt_reference @ covariance_generated @ sqrt_reference
    covariance_mean = psd_matrix_square_root(middle)
    mean_term = (mean_reference - mean_generated).square().sum()
    trace_term = torch.trace(
        covariance_reference
        + covariance_generated
        - 2.0 * covariance_mean
    )
    return float((mean_term + trace_term).clamp_min(0.0).item())


def polynomial_kernel_inception_distance(
    reference_features: np.ndarray,
    generated_features: np.ndarray,
) -> float:
    reference = torch.from_numpy(reference_features).double()
    generated = torch.from_numpy(generated_features).double()
    dimension = reference.shape[1]

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return ((left @ right.T) / dimension + 1.0) ** 3

    rr = kernel(reference, reference)
    gg = kernel(generated, generated)
    rg = kernel(reference, generated)
    reference_count = reference.shape[0]
    generated_count = generated.shape[0]
    rr_mean = (
        (rr.sum() - rr.diagonal().sum())
        / max(reference_count * (reference_count - 1), 1)
    )
    gg_mean = (
        (gg.sum() - gg.diagonal().sum())
        / max(generated_count * (generated_count - 1), 1)
    )
    return float((rr_mean + gg_mean - 2.0 * rg.mean()).item())


def manifest_image_text_pairs(
    manifest_file: str | Path,
) -> List[Tuple[Path, str]]:
    manifest_path = Path(manifest_file).resolve()
    pairs: List[Tuple[Path, str]] = []
    for record in read_jsonl(manifest_path):
        image_value = record.get("image")
        text_value = (
            record.get("prompt")
            or record.get("caption")
            or record.get("answer")
        )
        if not image_value or not text_value:
            continue
        image_path = Path(str(image_value))
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        pairs.append((image_path.resolve(), str(text_value)))
    return pairs


@torch.no_grad()
def clip_score(
    manifest_file: str | Path,
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> float:
    try:
        from transformers import AutoProcessor, CLIPModel
    except ImportError as error:
        raise RuntimeError(
            "CLIPScore requires transformers. Install it or omit "
            "--prompt-manifest."
        ) from error

    pairs = manifest_image_text_pairs(manifest_file)
    if not pairs:
        raise ValueError("The prompt manifest contained no image/text pairs.")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    scores: List[torch.Tensor] = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        images = []
        for path, _ in batch:
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        texts = [text for _, text in batch]
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        image_embeddings = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )
        text_arguments = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
        }
        text_embeddings = model.get_text_features(**text_arguments)
        image_embeddings = F.normalize(image_embeddings.float(), dim=-1)
        text_embeddings = F.normalize(text_embeddings.float(), dim=-1)
        scores.append((image_embeddings * text_embeddings).sum(dim=-1).cpu())
    return float(torch.cat(scores).mean().item() * 100.0)


def evaluate_generation(args: argparse.Namespace, runtime: Runtime) -> None:
    generated_files = find_image_files(
        args.generated_dir,
        args.eval_max_images,
    )
    report: Dict[str, Any] = {
        "generated": evaluate_basic_image_set(generated_files),
        "configuration": {
            "generated_dir": str(Path(args.generated_dir).resolve()),
            "real_dir": str(Path(args.real_dir).resolve())
            if args.real_dir
            else None,
            "feature_backbone": args.feature_backbone,
        },
    }

    if args.real_dir:
        real_files = find_image_files(args.real_dir, args.eval_max_images)
        report["real"] = evaluate_basic_image_set(real_files)
        generated_hashes = set()
        real_hashes = set()
        for path in generated_files:
            with Image.open(path) as image:
                generated_hashes.add(difference_hash(image))
        for path in real_files:
            with Image.open(path) as image:
                real_hashes.add(difference_hash(image))
        report["exact_dhash_overlap_rate"] = len(
            generated_hashes & real_hashes
        ) / max(len(generated_hashes), 1)

        if args.feature_backbone == "inception":
            if min(len(real_files), len(generated_files)) < 2:
                raise ValueError(
                    "At least two real and generated images are required "
                    "for distribution metrics."
                )
            reference_features = inception_features(
                real_files,
                runtime.device,
                args.eval_batch_size,
                args.workers,
            )
            generated_features = inception_features(
                generated_files,
                runtime.device,
                args.eval_batch_size,
                args.workers,
            )
            reference_features = deterministic_projection(
                reference_features,
                args.eval_feature_dim,
                args.seed,
            )
            generated_features = deterministic_projection(
                generated_features,
                args.eval_feature_dim,
                args.seed,
            )
            report["projected_fid"] = projected_frechet_distance(
                reference_features,
                generated_features,
            )
            report["kid"] = polynomial_kernel_inception_distance(
                reference_features,
                generated_features,
            )
            report["feature_dimension"] = int(reference_features.shape[1])

    if args.prompt_manifest:
        report["clip_score"] = clip_score(
            args.prompt_manifest,
            args.clip_model_name,
            runtime.device,
            args.eval_batch_size,
        )

    if runtime.is_main:
        write_json(args.eval_report, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Evaluation report: {args.eval_report}")


# -----------------------------------------------------------------------------
# Lightweight VLM post-training: SFT, reward modelling, and RL
# -----------------------------------------------------------------------------
class SimpleTokenizer:
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"]
    TOKEN_PATTERN = re.compile(r"[\w']+|[^\w\s]", flags=re.UNICODE)

    def __init__(self, token_to_id: Dict[str, int]) -> None:
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {
            index: token for token, index in self.token_to_id.items()
        }
        for token in self.SPECIAL_TOKENS:
            if token not in self.token_to_id:
                raise ValueError(f"Tokenizer is missing special token {token}.")

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        vocabulary_size: int,
        minimum_frequency: int,
    ) -> "SimpleTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(cls.tokenise(text))
        available = max(0, vocabulary_size - len(cls.SPECIAL_TOKENS))
        selected = [
            token
            for token, count in counts.most_common()
            if count >= minimum_frequency
        ][:available]
        ordered = cls.SPECIAL_TOKENS + selected
        return cls({token: index for index, token in enumerate(ordered)})

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SimpleTokenizer":
        return cls({str(k): int(v) for k, v in value["token_to_id"].items()})

    def to_dict(self) -> Dict[str, Any]:
        return {"token_to_id": self.token_to_id}

    @classmethod
    def tokenise(cls, text: str) -> List[str]:
        return cls.TOKEN_PATTERN.findall(str(text).lower())

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def sep_id(self) -> int:
        return self.token_to_id["<sep>"]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode_tokens(self, text: str) -> List[int]:
        return [
            self.token_to_id.get(token, self.unk_id)
            for token in self.tokenise(text)
        ]

    def prefix(self, prompt: str) -> List[int]:
        return [self.bos_id] + self.encode_tokens(prompt) + [self.sep_id]

    def encode_pair(
        self,
        prompt: str,
        answer: str,
        max_length: int,
    ) -> Tuple[List[int], List[int], int]:
        prefix = self.prefix(prompt)[: max_length - 1]
        remaining_answer_space = max(0, max_length - len(prefix) - 1)
        answer_ids = self.encode_tokens(answer)[:remaining_answer_space]
        sequence = prefix + answer_ids + [self.eos_id]
        prefix_length = len(prefix)
        labels = list(sequence)
        for index in range(prefix_length):
            labels[index] = -100
        return sequence, labels, prefix_length

    def decode(self, token_ids: Iterable[int]) -> str:
        tokens: List[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            if token == "<eos>":
                break
            if token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:%)\]])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        return text.strip()


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def sft_record_values(record: Dict[str, Any]) -> Tuple[str, str, str]:
    image = record.get("image")
    prompt = record.get("prompt", "Describe the image.")
    answer = record.get("answer", record.get("caption", record.get("text")))
    if not image or answer is None:
        raise ValueError(
            "Each SFT record needs 'image' and 'answer', 'caption', or 'text'."
        )
    return str(image), str(prompt), str(answer)


class VlmSftDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        manifest_file: str | Path,
        tokenizer: SimpleTokenizer,
        image_size: int,
        max_length: int,
        training: bool,
    ) -> None:
        self.manifest_path = Path(manifest_file).resolve()
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.training = training
        self.items: List[Tuple[Path, str, str]] = []
        for record in records:
            image, prompt, answer = sft_record_values(record)
            image_path = resolve_manifest_path(image, self.manifest_path)
            if image_path.exists():
                self.items.append((image_path, prompt, answer))
        if not self.items:
            raise ValueError("No usable image/text records were found for VLM SFT.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, prompt, answer = self.items[index]
        with Image.open(image_path) as image:
            image_tensor = training_pil_to_tensor(
                image,
                self.image_size,
                random_crop=self.training,
                horizontal_flip=self.training,
            )
        token_ids, labels, _ = self.tokenizer.encode_pair(
            prompt,
            answer,
            self.max_length,
        )
        return (
            image_tensor,
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_sft(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, sequences, labels = zip(*batch)
    maximum = max(sequence.shape[0] for sequence in sequences)
    input_ids = torch.full(
        (len(batch), maximum),
        pad_id,
        dtype=torch.long,
    )
    label_tensor = torch.full(
        (len(batch), maximum),
        -100,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (len(batch), maximum),
        dtype=torch.bool,
    )
    for index, (sequence, label) in enumerate(zip(sequences, labels)):
        length = sequence.shape[0]
        input_ids[index, :length] = sequence
        label_tensor[index, :length] = label
        attention_mask[index, :length] = True
    return torch.stack(images), input_ids, label_tensor, attention_mask


class VisionEncoder(nn.Module):
    def __init__(self, width: int, output_dimension: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2),
            nn.GroupNorm(normalisation_groups(width), width),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GroupNorm(normalisation_groups(width * 2), width * 2),
            nn.SiLU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.GroupNorm(normalisation_groups(width * 4), width * 4),
            nn.SiLU(),
            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1),
            nn.GroupNorm(normalisation_groups(width * 4), width * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(width * 4, output_dimension)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).flatten(1)
        return self.projection(features)


class VisionLanguagePolicy(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        vision_width: int,
        embedding_dimension: int,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vision = VisionEncoder(vision_width, hidden_dimension)
        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension,
        )
        self.hidden_init = nn.Linear(hidden_dimension, hidden_dimension)
        self.gru = nn.GRU(
            embedding_dimension,
            hidden_dimension,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(hidden_dimension, vocabulary_size)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.vision(images)
        initial_hidden = torch.tanh(
            self.hidden_init(image_features)
        ).unsqueeze(0)
        embeddings = self.token_embedding(input_ids)
        hidden, _ = self.gru(embeddings, initial_hidden)
        return self.output_head(self.dropout(hidden))


class VisionLanguageRewardModel(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        vision_width: int,
        embedding_dimension: int,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vision = VisionEncoder(vision_width, hidden_dimension)
        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension,
        )
        self.hidden_init = nn.Linear(hidden_dimension, hidden_dimension)
        self.gru = nn.GRU(
            embedding_dimension,
            hidden_dimension,
            batch_first=True,
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.vision(images)
        initial_hidden = torch.tanh(
            self.hidden_init(image_features)
        ).unsqueeze(0)
        embeddings = self.token_embedding(input_ids)
        hidden, _ = self.gru(embeddings, initial_hidden)
        lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        final_hidden = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            lengths,
        ]
        return self.reward_head(final_hidden).squeeze(-1)


def vlm_model_config(args: argparse.Namespace, vocabulary_size: int) -> Dict[str, Any]:
    return {
        "vocabulary_size": int(vocabulary_size),
        "vision_width": int(args.vlm_vision_width),
        "embedding_dimension": int(args.vlm_embedding_dim),
        "hidden_dimension": int(args.vlm_hidden_dim),
        "dropout": float(args.vlm_dropout),
    }


def make_custom_loader(
    dataset: Dataset,
    runtime: Runtime,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    collate_function: Any,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    sampler: Optional[DistributedSampler] = None
    if runtime.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=runtime.device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        collate_fn=collate_function,
        drop_last=False,
    )
    return loader, sampler


def split_records(
    records: Sequence[Dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2 or validation_fraction <= 0:
        return shuffled, []
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def masked_language_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def validate_vlm_policy(
    model: nn.Module,
    loader: Optional[DataLoader],
    runtime: Runtime,
    mixed_precision: bool,
) -> float:
    if loader is None:
        return math.inf
    model.eval()
    total = 0.0
    count = 0
    for images, input_ids, labels, _ in loader:
        images = images.to(runtime.device, non_blocking=True)
        input_ids = input_ids.to(runtime.device, non_blocking=True)
        labels = labels.to(runtime.device, non_blocking=True)
        with torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.float16,
            enabled=mixed_precision and runtime.device.type == "cuda",
        ):
            logits = model(images, input_ids)
            loss = masked_language_loss(logits, labels)
        total += float(loss.item()) * images.shape[0]
        count += images.shape[0]
    return reduce_loss(total, count, runtime)


def train_vlm_sft(args: argparse.Namespace, runtime: Runtime) -> None:
    seed_everything(args.seed, runtime.rank)
    records = read_jsonl(args.vlm_sft_file)
    if not records:
        raise ValueError("The SFT JSONL file is empty.")

    tokenizer: SimpleTokenizer
    resume_checkpoint: Optional[Dict[str, Any]] = None
    if args.vlm_resume:
        resume_checkpoint = load_torch_checkpoint(
            args.vlm_resume,
            map_location="cpu",
        )
        tokenizer = SimpleTokenizer.from_dict(resume_checkpoint["tokenizer"])
    else:
        training_texts: List[str] = []
        for record in records:
            _, prompt, answer = sft_record_values(record)
            training_texts.extend((prompt, answer))
        tokenizer = SimpleTokenizer.build(
            training_texts,
            args.vlm_vocab_size,
            args.vlm_min_frequency,
        )

    training_records, validation_records = split_records(
        records,
        args.validation_fraction,
        args.seed,
    )
    training_dataset = VlmSftDataset(
        training_records,
        args.vlm_sft_file,
        tokenizer,
        args.vlm_image_size,
        args.vlm_max_length,
        training=True,
    )
    validation_dataset = (
        VlmSftDataset(
            validation_records,
            args.vlm_sft_file,
            tokenizer,
            args.vlm_image_size,
            args.vlm_max_length,
            training=False,
        )
        if validation_records
        else None
    )
    collate = partial(collate_sft, pad_id=tokenizer.pad_id)
    training_loader, sampler = make_custom_loader(
        training_dataset,
        runtime,
        args.vlm_batch_size,
        args.workers,
        True,
        args.seed,
        collate,
    )
    validation_loader: Optional[DataLoader] = None
    if validation_dataset is not None:
        validation_loader, _ = make_custom_loader(
            validation_dataset,
            runtime,
            args.vlm_batch_size,
            args.workers,
            False,
            args.seed,
            collate,
        )

    configuration = (
        resume_checkpoint["model_config"]
        if resume_checkpoint
        else vlm_model_config(args, len(tokenizer))
    )
    model = VisionLanguagePolicy(**configuration).to(runtime.device)
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
    if runtime.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.vlm_learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = make_gradient_scaler(
        args.mixed_precision and runtime.device.type == "cuda"
    )
    best_validation = math.inf
    global_step = 0
    rank_print(runtime, f"VLM SFT records: {len(training_dataset)}")
    rank_print(runtime, f"Tokenizer vocabulary: {len(tokenizer)}")

    for epoch in range(1, args.vlm_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_total = 0.0
        epoch_count = 0

        for batch_index, (images, input_ids, labels, _) in enumerate(
            training_loader
        ):
            images = images.to(runtime.device, non_blocking=True)
            input_ids = input_ids.to(runtime.device, non_blocking=True)
            labels = labels.to(runtime.device, non_blocking=True)
            should_update = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(training_loader)
            )
            synchronisation = (
                model.no_sync()
                if runtime.distributed and not should_update
                else contextlib.nullcontext()
            )
            with synchronisation:
                with torch.autocast(
                    device_type=runtime.device.type,
                    dtype=torch.float16,
                    enabled=(
                        args.mixed_precision and runtime.device.type == "cuda"
                    ),
                ):
                    logits = model(images, input_ids)
                    loss = masked_language_loss(logits, labels)
                    scaled_loss = loss / args.gradient_accumulation
                scaler.scale(scaled_loss).backward()

            epoch_total += float(loss.detach().item()) * images.shape[0]
            epoch_count += images.shape[0]
            if should_update:
                scaler.unscale_(optimizer)
                if args.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        training_loss = reduce_loss(epoch_total, epoch_count, runtime)
        validation_loss = validate_vlm_policy(
            model,
            validation_loader,
            runtime,
            args.mixed_precision,
        )
        selection = (
            validation_loss if math.isfinite(validation_loss) else training_loss
        )
        improved = selection < best_validation
        best_validation = min(best_validation, selection)
        rank_print(
            runtime,
            f"VLM SFT epoch {epoch:>3}/{args.vlm_epochs} | "
            f"train {training_loss:.5f} | "
            + (
                f"validation {validation_loss:.5f}"
                if math.isfinite(validation_loss)
                else "validation n/a"
            ),
        )

        if runtime.is_main and (improved or epoch == args.vlm_epochs):
            checkpoint = {
                "format_version": FORMAT_VERSION,
                "task": "vlm_policy",
                "post_training_stage": "sft",
                "model_config": configuration,
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in unwrap_model(model).state_dict().items()
                },
                "tokenizer": tokenizer.to_dict(),
                "image_size": int(args.vlm_image_size),
                "max_length": int(args.vlm_max_length),
                "epoch": epoch,
                "global_step": global_step,
                "best_validation_loss": best_validation,
                "training_records": len(training_dataset),
            }
            atomic_torch_save(checkpoint, args.vlm_policy_file)
            print(f"Saved VLM SFT policy: {args.vlm_policy_file}")

    rank_print(runtime, "VLM supervised fine-tuning completed.")


def load_vlm_policy(
    checkpoint_file: str | Path,
    device: torch.device,
) -> Tuple[VisionLanguagePolicy, SimpleTokenizer, Dict[str, Any]]:
    checkpoint = load_torch_checkpoint(checkpoint_file, map_location="cpu")
    if checkpoint.get("task") != "vlm_policy":
        raise ValueError(f"'{checkpoint_file}' is not a VLM policy checkpoint.")
    tokenizer = SimpleTokenizer.from_dict(checkpoint["tokenizer"])
    model = VisionLanguagePolicy(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, tokenizer, checkpoint


@torch.no_grad()
def generate_vlm_answer(
    model: VisionLanguagePolicy,
    tokenizer: SimpleTokenizer,
    image: torch.Tensor,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> Tuple[str, List[int]]:
    sequence = tokenizer.prefix(prompt)
    input_ids = torch.tensor(
        sequence,
        dtype=torch.long,
        device=image.device,
    ).unsqueeze(0)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        logits = model(image, input_ids)[0, -1]
        if temperature <= 0:
            next_token = int(logits.argmax().item())
        else:
            logits = logits / max(temperature, 1e-4)
            if top_k > 0 and top_k < logits.shape[0]:
                threshold = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < threshold, -float("inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_token = int(torch.multinomial(probabilities, 1).item())
        if next_token == tokenizer.eos_id:
            break
        generated.append(next_token)
        input_ids = torch.cat(
            (
                input_ids,
                torch.tensor(
                    [[next_token]],
                    dtype=torch.long,
                    device=image.device,
                ),
            ),
            dim=1,
        )
    return tokenizer.decode(generated), generated


def vlm_generate(args: argparse.Namespace, runtime: Runtime) -> None:
    if not Path(args.seed_image).exists():
        raise FileNotFoundError(f"Seed image '{args.seed_image}' was not found.")
    seed_everything(args.seed, runtime.rank)
    model, tokenizer, checkpoint = load_vlm_policy(
        args.vlm_policy_file,
        runtime.device,
    )
    with Image.open(args.seed_image) as image:
        image_tensor = pil_to_tensor(
            image,
            int(checkpoint["image_size"]),
        ).unsqueeze(0)
    image_tensor = image_tensor.to(runtime.device)
    answer, _ = generate_vlm_answer(
        model,
        tokenizer,
        image_tensor,
        args.prompt,
        args.vlm_max_new_tokens,
        args.vlm_temperature,
        args.vlm_top_k,
    )
    print(answer)
    if args.vlm_text_output:
        Path(args.vlm_text_output).write_text(answer + "\n", encoding="utf-8")
        print(f"Text output: {args.vlm_text_output}")


class VlmPreferenceDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        preference_file: str | Path,
        tokenizer: SimpleTokenizer,
        image_size: int,
        max_length: int,
        training: bool,
    ) -> None:
        self.preference_path = Path(preference_file).resolve()
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.training = training
        self.items: List[Tuple[Path, str, Path, str]] = []

        for record in records:
            prompt = str(record.get("prompt", "Judge this response."))
            if record.get("chosen_image") and record.get("rejected_image"):
                chosen_image = resolve_manifest_path(
                    str(record["chosen_image"]),
                    self.preference_path,
                )
                rejected_image = resolve_manifest_path(
                    str(record["rejected_image"]),
                    self.preference_path,
                )
                chosen_text = prompt
                rejected_text = prompt
            elif record.get("image") and record.get("chosen") is not None and record.get("rejected") is not None:
                chosen_image = resolve_manifest_path(
                    str(record["image"]),
                    self.preference_path,
                )
                rejected_image = chosen_image
                chosen_text = (
                    prompt + " <sep> " + str(record["chosen"])
                )
                rejected_text = (
                    prompt + " <sep> " + str(record["rejected"])
                )
            else:
                continue

            if chosen_image.exists() and rejected_image.exists():
                self.items.append(
                    (
                        chosen_image,
                        chosen_text,
                        rejected_image,
                        rejected_text,
                    )
                )
        if not self.items:
            raise ValueError(
                "No usable preference pairs were found. Use either "
                "{image,prompt,chosen,rejected} or "
                "{chosen_image,rejected_image,prompt} records."
            )

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return training_pil_to_tensor(
                image,
                self.image_size,
                random_crop=self.training,
                horizontal_flip=False,
            )

    def _encode_text(self, text: str) -> torch.Tensor:
        ids = (
            [self.tokenizer.bos_id]
            + self.tokenizer.encode_tokens(text)
            + [self.tokenizer.eos_id]
        )
        ids = ids[: self.max_length]
        if ids[-1] != self.tokenizer.eos_id:
            ids[-1] = self.tokenizer.eos_id
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_image, chosen_text, rejected_image, rejected_text = self.items[index]
        return (
            self._load_image(chosen_image),
            self._encode_text(chosen_text),
            self._load_image(rejected_image),
            self._encode_text(rejected_text),
        )


def pad_token_batch(
    sequences: Sequence[torch.Tensor],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    maximum = max(sequence.shape[0] for sequence in sequences)
    values = torch.full(
        (len(sequences), maximum),
        pad_id,
        dtype=torch.long,
    )
    mask = torch.zeros((len(sequences), maximum), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        values[index, : sequence.shape[0]] = sequence
        mask[index, : sequence.shape[0]] = True
    return values, mask


def collate_preferences(
    batch: Sequence[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    pad_id: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    chosen_images, chosen_ids, rejected_images, rejected_ids = zip(*batch)
    chosen_tokens, chosen_mask = pad_token_batch(chosen_ids, pad_id)
    rejected_tokens, rejected_mask = pad_token_batch(rejected_ids, pad_id)
    return (
        torch.stack(chosen_images),
        chosen_tokens,
        chosen_mask,
        torch.stack(rejected_images),
        rejected_tokens,
        rejected_mask,
    )


def initialise_reward_from_policy(
    reward_model: VisionLanguageRewardModel,
    policy_state: Dict[str, torch.Tensor],
) -> None:
    reward_state = reward_model.state_dict()
    compatible = {
        name: value
        for name, value in policy_state.items()
        if name in reward_state and reward_state[name].shape == value.shape
    }
    reward_model.load_state_dict(compatible, strict=False)


@torch.no_grad()
def validate_reward_model(
    model: nn.Module,
    loader: Optional[DataLoader],
    runtime: Runtime,
) -> Tuple[float, float]:
    if loader is None:
        return math.inf, 0.0
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for (
        chosen_images,
        chosen_ids,
        chosen_mask,
        rejected_images,
        rejected_ids,
        rejected_mask,
    ) in loader:
        chosen_images = chosen_images.to(runtime.device)
        chosen_ids = chosen_ids.to(runtime.device)
        chosen_mask = chosen_mask.to(runtime.device)
        rejected_images = rejected_images.to(runtime.device)
        rejected_ids = rejected_ids.to(runtime.device)
        rejected_mask = rejected_mask.to(runtime.device)
        chosen_rewards = model(chosen_images, chosen_ids, chosen_mask)
        rejected_rewards = model(rejected_images, rejected_ids, rejected_mask)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        total_loss += float(loss.item()) * chosen_images.shape[0]
        total_correct += int((chosen_rewards > rejected_rewards).sum().item())
        total_count += chosen_images.shape[0]
    return (
        reduce_loss(total_loss, total_count, runtime),
        reduce_loss(float(total_correct), total_count, runtime),
    )


def train_vlm_reward_model(
    args: argparse.Namespace,
    runtime: Runtime,
) -> None:
    seed_everything(args.seed, runtime.rank)
    policy_checkpoint = load_torch_checkpoint(
        args.vlm_policy_file,
        map_location="cpu",
    )
    if policy_checkpoint.get("task") != "vlm_policy":
        raise ValueError("Reward modelling requires a VLM policy checkpoint.")
    tokenizer = SimpleTokenizer.from_dict(policy_checkpoint["tokenizer"])
    records = read_jsonl(args.vlm_preference_file)
    training_records, validation_records = split_records(
        records,
        args.validation_fraction,
        args.seed,
    )
    training_dataset = VlmPreferenceDataset(
        training_records,
        args.vlm_preference_file,
        tokenizer,
        int(policy_checkpoint["image_size"]),
        args.vlm_max_length,
        training=True,
    )
    validation_dataset = (
        VlmPreferenceDataset(
            validation_records,
            args.vlm_preference_file,
            tokenizer,
            int(policy_checkpoint["image_size"]),
            args.vlm_max_length,
            training=False,
        )
        if validation_records
        else None
    )
    collate = partial(collate_preferences, pad_id=tokenizer.pad_id)
    training_loader, sampler = make_custom_loader(
        training_dataset,
        runtime,
        args.vlm_batch_size,
        args.workers,
        True,
        args.seed,
        collate,
    )
    validation_loader: Optional[DataLoader] = None
    if validation_dataset is not None:
        validation_loader, _ = make_custom_loader(
            validation_dataset,
            runtime,
            args.vlm_batch_size,
            args.workers,
            False,
            args.seed,
            collate,
        )

    configuration = policy_checkpoint["model_config"]
    model = VisionLanguageRewardModel(**configuration).to(runtime.device)
    initialise_reward_from_policy(
        model,
        policy_checkpoint["model_state_dict"],
    )
    if runtime.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.vlm_learning_rate,
        weight_decay=args.weight_decay,
    )
    best_accuracy = -math.inf
    rank_print(runtime, f"Reward-model preference pairs: {len(training_dataset)}")

    for epoch in range(1, args.vlm_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_count = 0

        for (
            chosen_images,
            chosen_ids,
            chosen_mask,
            rejected_images,
            rejected_ids,
            rejected_mask,
        ) in training_loader:
            chosen_images = chosen_images.to(runtime.device)
            chosen_ids = chosen_ids.to(runtime.device)
            chosen_mask = chosen_mask.to(runtime.device)
            rejected_images = rejected_images.to(runtime.device)
            rejected_ids = rejected_ids.to(runtime.device)
            rejected_mask = rejected_mask.to(runtime.device)

            optimizer.zero_grad(set_to_none=True)
            chosen_rewards = model(chosen_images, chosen_ids, chosen_mask)
            rejected_rewards = model(
                rejected_images,
                rejected_ids,
                rejected_mask,
            )
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            epoch_loss += float(loss.detach().item()) * chosen_images.shape[0]
            epoch_correct += int(
                (chosen_rewards > rejected_rewards).sum().item()
            )
            epoch_count += chosen_images.shape[0]

        training_loss = reduce_loss(epoch_loss, epoch_count, runtime)
        training_accuracy = reduce_loss(
            float(epoch_correct),
            epoch_count,
            runtime,
        )
        validation_loss, validation_accuracy = validate_reward_model(
            model,
            validation_loader,
            runtime,
        )
        selection_accuracy = (
            validation_accuracy
            if validation_loader is not None
            else training_accuracy
        )
        improved = selection_accuracy > best_accuracy
        best_accuracy = max(best_accuracy, selection_accuracy)
        rank_print(
            runtime,
            f"VLM RM epoch {epoch:>3}/{args.vlm_epochs} | "
            f"train loss {training_loss:.5f} | "
            f"train acc {training_accuracy:.3f} | "
            + (
                f"validation loss {validation_loss:.5f} | "
                f"validation acc {validation_accuracy:.3f}"
                if validation_loader is not None
                else "validation n/a"
            ),
        )

        if runtime.is_main and (improved or epoch == args.vlm_epochs):
            checkpoint = {
                "format_version": FORMAT_VERSION,
                "task": "vlm_reward_model",
                "post_training_stage": "rm",
                "model_config": configuration,
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in unwrap_model(model).state_dict().items()
                },
                "tokenizer": tokenizer.to_dict(),
                "image_size": int(policy_checkpoint["image_size"]),
                "max_length": int(args.vlm_max_length),
                "epoch": epoch,
                "best_accuracy": best_accuracy,
                "preference_records": len(training_dataset),
                "base_policy": str(args.vlm_policy_file),
                "objective": "Bradley-Terry pairwise preference loss",
            }
            atomic_torch_save(checkpoint, args.vlm_reward_file)
            print(f"Saved VLM reward model: {args.vlm_reward_file}")

    rank_print(runtime, "VLM reward-model training completed.")


def load_vlm_reward_model(
    checkpoint_file: str | Path,
    device: torch.device,
) -> Tuple[VisionLanguageRewardModel, SimpleTokenizer, Dict[str, Any]]:
    checkpoint = load_torch_checkpoint(checkpoint_file, map_location="cpu")
    if checkpoint.get("task") != "vlm_reward_model":
        raise ValueError(
            f"'{checkpoint_file}' is not a VLM reward-model checkpoint."
        )
    tokenizer = SimpleTokenizer.from_dict(checkpoint["tokenizer"])
    model = VisionLanguageRewardModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, tokenizer, checkpoint


def sample_policy_with_logprob(
    policy: VisionLanguagePolicy,
    tokenizer: SimpleTokenizer,
    image: torch.Tensor,
    prompt: str,
    maximum_new_tokens: int,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    prefix = tokenizer.prefix(prompt)
    sequence = torch.tensor(
        prefix,
        dtype=torch.long,
        device=image.device,
    ).unsqueeze(0)
    log_probabilities: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []

    for _ in range(maximum_new_tokens):
        logits = policy(image, sequence)[0, -1]
        distribution = torch.distributions.Categorical(
            logits=logits / max(temperature, 1e-4)
        )
        token = distribution.sample()
        log_probabilities.append(distribution.log_prob(token))
        entropies.append(distribution.entropy())
        sequence = torch.cat((sequence, token.reshape(1, 1)), dim=1)
        if int(token.item()) == tokenizer.eos_id:
            break

    log_probability = torch.stack(log_probabilities).sum()
    entropy = torch.stack(entropies).mean()
    return sequence, log_probability, entropy, len(prefix)


@torch.no_grad()
def sequence_log_probability(
    policy: VisionLanguagePolicy,
    image: torch.Tensor,
    sequence: torch.Tensor,
    prefix_length: int,
    temperature: float,
) -> torch.Tensor:
    logits = policy(image, sequence[:, :-1])
    targets = sequence[:, 1:]
    log_probabilities = F.log_softmax(
        logits.float() / max(temperature, 1e-4),
        dim=-1,
    )
    selected = log_probabilities.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    generated_start = max(0, prefix_length - 1)
    return selected[:, generated_start:].sum(dim=1)[0]


def train_vlm_rl(args: argparse.Namespace, runtime: Runtime) -> None:
    if runtime.distributed:
        raise RuntimeError(
            "The compact autoregressive RL loop is single-process. "
            "Use DDP for SFT/RM, then run RL on one GPU."
        )
    seed_everything(args.seed, runtime.rank)
    policy, tokenizer, policy_checkpoint = load_vlm_policy(
        args.vlm_policy_file,
        runtime.device,
    )
    reference_policy = copy.deepcopy(policy).eval()
    reference_policy.requires_grad_(False)
    reward_model, reward_tokenizer, reward_checkpoint = load_vlm_reward_model(
        args.vlm_reward_file,
        runtime.device,
    )
    if tokenizer.token_to_id != reward_tokenizer.token_to_id:
        raise ValueError("Policy and reward-model tokenizers do not match.")
    if policy_checkpoint["model_config"] != reward_checkpoint["model_config"]:
        raise ValueError("Policy and reward-model architectures do not match.")

    records = read_jsonl(args.vlm_sft_file)
    usable_records: List[Tuple[Path, str]] = []
    manifest_path = Path(args.vlm_sft_file).resolve()
    for record in records:
        try:
            image_value, prompt, _ = sft_record_values(record)
        except ValueError:
            continue
        image_path = resolve_manifest_path(image_value, manifest_path)
        if image_path.exists():
            usable_records.append((image_path, prompt))
    if not usable_records:
        raise ValueError("No usable image/prompt records were found for RL.")

    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.rl_learning_rate,
        weight_decay=args.weight_decay,
    )
    reward_baseline = 0.0
    baseline_initialised = False
    global_step = 0

    for epoch in range(1, args.rl_epochs + 1):
        random.Random(args.seed + epoch).shuffle(usable_records)
        epoch_rewards: List[float] = []
        epoch_kls: List[float] = []

        for start in range(0, len(usable_records), args.rl_batch_size):
            batch = usable_records[start : start + args.rl_batch_size]
            sample_log_probabilities: List[torch.Tensor] = []
            sample_entropies: List[torch.Tensor] = []
            reference_log_probabilities: List[torch.Tensor] = []
            rewards: List[torch.Tensor] = []

            for image_path, prompt in batch:
                with Image.open(image_path) as image:
                    image_tensor = pil_to_tensor(
                        image,
                        int(policy_checkpoint["image_size"]),
                    ).unsqueeze(0)
                image_tensor = image_tensor.to(runtime.device)
                sequence, policy_log_probability, entropy, prefix_length = (
                    sample_policy_with_logprob(
                        policy,
                        tokenizer,
                        image_tensor,
                        prompt,
                        args.vlm_max_new_tokens,
                        args.vlm_temperature,
                    )
                )
                with torch.no_grad():
                    reference_log_probability = sequence_log_probability(
                        reference_policy,
                        image_tensor,
                        sequence,
                        prefix_length,
                        args.vlm_temperature,
                    )
                    attention_mask = torch.ones_like(
                        sequence,
                        dtype=torch.bool,
                    )
                    reward = reward_model(
                        image_tensor,
                        sequence,
                        attention_mask,
                    )[0]

                sample_log_probabilities.append(policy_log_probability)
                sample_entropies.append(entropy)
                reference_log_probabilities.append(reference_log_probability)
                rewards.append(reward)

            policy_log_probs = torch.stack(sample_log_probabilities)
            entropy_values = torch.stack(sample_entropies)
            reference_log_probs = torch.stack(reference_log_probabilities)
            reward_values = torch.stack(rewards).float()

            mean_reward = float(reward_values.mean().item())
            if not baseline_initialised:
                reward_baseline = mean_reward
                baseline_initialised = True
            advantages = reward_values - reward_baseline
            if advantages.numel() > 1:
                advantages = advantages / (
                    advantages.std(unbiased=False) + 1e-6
                )
            sampled_kl = policy_log_probs - reference_log_probs
            loss = (
                -(advantages.detach() * policy_log_probs).mean()
                + args.rl_kl_beta * sampled_kl.mean()
                - args.rl_entropy_coefficient * entropy_values.mean()
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    args.gradient_clip,
                )
            optimizer.step()
            reward_baseline = (
                args.rl_baseline_decay * reward_baseline
                + (1.0 - args.rl_baseline_decay) * mean_reward
            )
            global_step += 1
            epoch_rewards.extend(reward_values.detach().cpu().tolist())
            epoch_kls.extend(sampled_kl.detach().cpu().tolist())

        mean_epoch_reward = float(np.mean(epoch_rewards))
        mean_epoch_kl = float(np.mean(epoch_kls))
        print(
            f"VLM RL epoch {epoch:>3}/{args.rl_epochs} | "
            f"reward {mean_epoch_reward:.4f} | "
            f"sampled KL {mean_epoch_kl:.4f}"
        )
        checkpoint = {
            "format_version": FORMAT_VERSION,
            "task": "vlm_policy",
            "post_training_stage": "rl",
            "model_config": policy_checkpoint["model_config"],
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in policy.state_dict().items()
            },
            "tokenizer": tokenizer.to_dict(),
            "image_size": int(policy_checkpoint["image_size"]),
            "max_length": int(policy_checkpoint["max_length"]),
            "epoch": epoch,
            "global_step": global_step,
            "base_policy": str(args.vlm_policy_file),
            "reward_model": str(args.vlm_reward_file),
            "mean_reward": mean_epoch_reward,
            "mean_sampled_kl": mean_epoch_kl,
            "objective": "KL-regularised reward-guided policy gradient",
        }
        atomic_torch_save(checkpoint, args.vlm_rl_output_file)
        print(f"Saved RL-aligned VLM policy: {args.vlm_rl_output_file}")

    print("VLM reinforcement-learning post-training completed.")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Train, generate, edit, distil, and evaluate image/video models, "
            "or run VLM SFT -> RM -> RL post-training."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "generate",
            "edit",
            "video-generate",
            "video-edit",
            "distill",
            "build-data",
            "evaluate",
            "vlm-sft",
            "vlm-rm",
            "vlm-rl",
            "vlm-generate",
        ],
        default="generate",
    )

    runtime_group = parser.add_argument_group("runtime and reproducibility")
    runtime_group.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    runtime_group.add_argument("--seed", type=int, default=42)
    runtime_group.add_argument("--workers", type=int, default=0)
    runtime_group.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    runtime_group.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    runtime_group.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
    )
    runtime_group.add_argument("--gradient-clip", type=float, default=1.0)
    runtime_group.add_argument("--log-every", type=int, default=25)

    diffusion_group = parser.add_argument_group("diffusion model and data")
    diffusion_group.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    diffusion_group.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    diffusion_group.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    diffusion_group.add_argument("--image-size", type=int, default=64)
    diffusion_group.add_argument("--base-channels", type=int, default=64)
    diffusion_group.add_argument("--dropout", type=float, default=0.1)
    diffusion_group.add_argument(
        "--attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    diffusion_group.add_argument("--diffusion-steps", type=int, default=1000)
    diffusion_group.add_argument(
        "--beta-schedule",
        choices=["linear", "cosine"],
        default="cosine",
    )
    diffusion_group.add_argument(
        "--prediction-type",
        choices=["epsilon", "v", "x0"],
        default="epsilon",
    )
    diffusion_group.add_argument("--max-files", type=int, default=0)
    diffusion_group.add_argument(
        "--horizontal-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    training_group = parser.add_argument_group("diffusion training")
    training_group.add_argument("--batch-size", type=int, default=32)
    training_group.add_argument("--epochs", type=int, default=50)
    training_group.add_argument("--learning-rate", type=float, default=2e-4)
    training_group.add_argument("--weight-decay", type=float, default=1e-4)
    training_group.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )
    training_group.add_argument("--warmup-steps", type=int, default=500)
    training_group.add_argument("--ema-decay", type=float, default=0.9999)
    training_group.add_argument("--checkpoint-every", type=int, default=1)
    training_group.add_argument(
        "--resume",
        default="",
        help="Resume complete optimiser/training state from a checkpoint.",
    )

    inference_group = parser.add_argument_group(
        "image generation and editing"
    )
    inference_group.add_argument("--num-images", type=int, default=1)
    inference_group.add_argument("--sampling-steps", type=int, default=50)
    inference_group.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity; zero is deterministic.",
    )
    inference_group.add_argument("--seed-image", default=DEFAULT_SEED_IMAGE)
    inference_group.add_argument(
        "--mask-file",
        default="",
        help="White pixels are regenerated; black pixels are preserved.",
    )
    inference_group.add_argument("--strength", type=float, default=0.65)

    video_group = parser.add_argument_group("video generation and editing")
    video_group.add_argument("--video-output", default="generated_video.gif")
    video_group.add_argument("--video-frames", type=int, default=16)
    video_group.add_argument("--video-batch-size", type=int, default=4)
    video_group.add_argument("--fps", type=int, default=8)
    video_group.add_argument(
        "--temporal-correlation",
        type=float,
        default=0.95,
    )
    video_group.add_argument("--motion-x", type=int, default=1)
    video_group.add_argument("--motion-y", type=int, default=0)
    video_group.add_argument("--seed-video", default="Seed.mp4")
    video_group.add_argument("--max-video-frames", type=int, default=64)

    distillation_group = parser.add_argument_group("diffusion distillation")
    distillation_group.add_argument(
        "--teacher-model-file",
        default="teacher_image_diffusion.pth",
    )
    distillation_group.add_argument(
        "--distilled-model-file",
        default="distilled_image_diffusion.pth",
    )
    distillation_group.add_argument(
        "--student-base-channels",
        type=int,
        default=32,
    )
    distillation_group.add_argument("--distill-alpha", type=float, default=0.8)
    distillation_group.add_argument(
        "--student-sampling-steps",
        type=int,
        default=20,
    )

    data_group = parser.add_argument_group(
        "multimodal data production"
    )
    data_group.add_argument(
        "--manifest-file",
        default="multimodal_manifest.jsonl",
    )
    data_group.add_argument(
        "--preference-file",
        default="image_preferences.jsonl",
    )
    data_group.add_argument("--min-resolution", type=int, default=64)
    data_group.add_argument("--max-aspect-ratio", type=float, default=3.0)
    data_group.add_argument("--min-sharpness", type=float, default=0.0)
    data_group.add_argument(
        "--include-rejected",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    evaluation_group = parser.add_argument_group(
        "image and multimodal evaluation"
    )
    evaluation_group.add_argument("--real-dir", default="")
    evaluation_group.add_argument("--generated-dir", default="generated_images")
    evaluation_group.add_argument(
        "--eval-report",
        default="generation_evaluation.json",
    )
    evaluation_group.add_argument(
        "--feature-backbone",
        choices=["none", "inception"],
        default="none",
    )
    evaluation_group.add_argument("--eval-feature-dim", type=int, default=256)
    evaluation_group.add_argument("--eval-max-images", type=int, default=1000)
    evaluation_group.add_argument("--eval-batch-size", type=int, default=32)
    evaluation_group.add_argument("--prompt-manifest", default="")
    evaluation_group.add_argument(
        "--clip-model-name",
        default="openai/clip-vit-base-patch32",
    )

    vlm_group = parser.add_argument_group("VLM post-training")
    vlm_group.add_argument(
        "--vlm-sft-file",
        default="vlm_sft.jsonl",
        help=(
            "JSONL schema: "
            "{image, prompt, answer}; caption/text may replace answer."
        ),
    )
    vlm_group.add_argument(
        "--vlm-preference-file",
        default="vlm_preferences.jsonl",
        help=(
            "JSONL schema: {image,prompt,chosen,rejected} or "
            "{chosen_image,rejected_image,prompt}."
        ),
    )
    vlm_group.add_argument(
        "--vlm-policy-file",
        default="vlm_sft_policy.pth",
    )
    vlm_group.add_argument(
        "--vlm-reward-file",
        default="vlm_reward_model.pth",
    )
    vlm_group.add_argument(
        "--vlm-rl-output-file",
        default="vlm_rl_policy.pth",
    )
    vlm_group.add_argument("--vlm-resume", default="")
    vlm_group.add_argument("--vlm-text-output", default="")
    vlm_group.add_argument("--vlm-image-size", type=int, default=128)
    vlm_group.add_argument("--vlm-vision-width", type=int, default=48)
    vlm_group.add_argument("--vlm-embedding-dim", type=int, default=256)
    vlm_group.add_argument("--vlm-hidden-dim", type=int, default=384)
    vlm_group.add_argument("--vlm-dropout", type=float, default=0.1)
    vlm_group.add_argument("--vlm-vocab-size", type=int, default=16000)
    vlm_group.add_argument("--vlm-min-frequency", type=int, default=1)
    vlm_group.add_argument("--vlm-max-length", type=int, default=96)
    vlm_group.add_argument("--vlm-max-new-tokens", type=int, default=32)
    vlm_group.add_argument("--vlm-batch-size", type=int, default=16)
    vlm_group.add_argument("--vlm-epochs", type=int, default=5)
    vlm_group.add_argument("--vlm-learning-rate", type=float, default=1e-4)
    vlm_group.add_argument("--vlm-temperature", type=float, default=0.8)
    vlm_group.add_argument("--vlm-top-k", type=int, default=50)
    vlm_group.add_argument(
        "--prompt",
        default="Describe the image in detail.",
    )

    rl_group = parser.add_argument_group("VLM reinforcement learning")
    rl_group.add_argument("--rl-epochs", type=int, default=2)
    rl_group.add_argument("--rl-batch-size", type=int, default=4)
    rl_group.add_argument("--rl-learning-rate", type=float, default=1e-6)
    rl_group.add_argument("--rl-kl-beta", type=float, default=0.02)
    rl_group.add_argument(
        "--rl-entropy-coefficient",
        type=float,
        default=0.001,
    )
    rl_group.add_argument("--rl-baseline-decay", type=float, default=0.9)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be at least 1")
    if args.image_size < 16 or args.image_size % 4 != 0:
        raise ValueError("image-size must be at least 16 and divisible by 4")
    if args.base_channels < 8 or args.student_base_channels < 8:
        raise ValueError("base channel counts must be at least 8")
    if args.diffusion_steps < 10:
        raise ValueError("diffusion-steps must be at least 10")
    if args.sampling_steps < 1:
        raise ValueError("sampling-steps must be positive")
    if args.num_images < 1:
        raise ValueError("num-images must be positive")
    if not 0.0 <= args.eta <= 1.0:
        raise ValueError("eta must be between 0 and 1")
    if not 0.0 < args.strength <= 1.0:
        raise ValueError("strength must be in (0, 1]")
    if args.batch_size < 1 or args.vlm_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.epochs < 1 or args.vlm_epochs < 1 or args.rl_epochs < 1:
        raise ValueError("epoch counts must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in [0, 1)")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema-decay must be in (0, 1)")
    if args.checkpoint_every < 1 or args.log_every < 1:
        raise ValueError("checkpoint-every and log-every must be positive")
    if not 0.0 <= args.distill_alpha <= 1.0:
        raise ValueError("distill-alpha must be between 0 and 1")
    if not 0.0 <= args.temporal_correlation < 1.0:
        raise ValueError("temporal-correlation must be in [0, 1)")
    if args.video_frames < 1 or args.video_batch_size < 1 or args.fps < 1:
        raise ValueError("video frame, batch, and FPS values must be positive")
    if args.vlm_image_size < 16:
        raise ValueError("vlm-image-size must be at least 16")
    if args.vlm_max_length < 8 or args.vlm_max_new_tokens < 1:
        raise ValueError("VLM sequence lengths are too small")
    if args.vlm_vocab_size <= len(SimpleTokenizer.SPECIAL_TOKENS):
        raise ValueError("vlm-vocab-size is too small")
    if not 0.0 <= args.rl_baseline_decay < 1.0:
        raise ValueError("rl-baseline-decay must be in [0, 1)")


def dispatch(args: argparse.Namespace, runtime: Runtime) -> None:
    operations = {
        "train": train_diffusion,
        "generate": generate_images,
        "edit": edit_image,
        "video-generate": generate_video,
        "video-edit": edit_video,
        "distill": distill_diffusion,
        "evaluate": evaluate_generation,
        "vlm-sft": train_vlm_sft,
        "vlm-rm": train_vlm_reward_model,
        "vlm-rl": train_vlm_rl,
        "vlm-generate": vlm_generate,
    }
    if args.mode == "build-data":
        if runtime.is_main:
            build_multimodal_data(args)
        return
    operations[args.mode](args, runtime)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_arguments(args)
    runtime = setup_runtime(args.device)
    try:
        dispatch(args, runtime)
    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
