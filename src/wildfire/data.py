from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow loading truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Colour preprocessing transforms ─────────────────────────────────────────

class WarmColorBoost:
    """
    Amplifies saturation + brightness of warm-hued pixels (red/orange/yellow).
    Subtle enhancement — keeps background intact.

    Config keys (under data.preprocessing):
      use_warm_color_boost: true
      warm_boost_saturation_scale: 1.35
      warm_boost_value_scale: 1.1
    """

    def __init__(self, saturation_scale: float = 1.35, value_scale: float = 1.1) -> None:
        self.saturation_scale = saturation_scale
        self.value_scale = value_scale

    def __call__(self, img):
        hsv = np.array(img.convert("HSV"), dtype=np.float32) / 255.0
        h = hsv[..., 0]
        warm_mask = (h >= 0.0) & (h <= 0.18)
        hsv[..., 1] = np.where(warm_mask, np.clip(hsv[..., 1] * self.saturation_scale, 0.0, 1.0), hsv[..., 1])
        hsv[..., 2] = np.where(warm_mask, np.clip(hsv[..., 2] * self.value_scale, 0.0, 1.0), hsv[..., 2])
        boosted = (hsv * 255.0).astype(np.uint8)
        return transforms.functional.to_pil_image(boosted, mode="HSV").convert("RGB")


class WarmColorSelectiveGrayscale:
    """
    Keeps warm fire tones in full colour; converts everything else to greyscale.
    Aggressive background suppression — risk: model learns grey-texture shortcut.

    Config keys:
      use_selective_grayscale: true
      selective_grayscale_saturation_threshold: 0.35
      selective_grayscale_warm_hue_low: 0.0
      selective_grayscale_warm_hue_high: 0.18
      selective_grayscale_min_value: 0.15
      selective_grayscale_blend_strength: 1.0
    """

    def __init__(
        self,
        saturation_threshold: float = 0.35,
        warm_hue_low: float = 0.0,
        warm_hue_high: float = 0.18,
        min_value: float = 0.15,
        blend_strength: float = 1.0,
    ) -> None:
        self.saturation_threshold = saturation_threshold
        self.warm_hue_low = warm_hue_low
        self.warm_hue_high = warm_hue_high
        self.min_value = min_value
        self.blend_strength = blend_strength

    def __call__(self, img):
        hsv = np.array(img.convert("HSV"), dtype=np.float32) / 255.0
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        warm_mask = (
            (h >= self.warm_hue_low)
            & (h <= self.warm_hue_high)
            & (s >= self.saturation_threshold)
            & (v >= self.min_value)
        )
        rgb = np.array(img.convert("RGB"), dtype=np.float32)
        gray = np.mean(rgb, axis=2, keepdims=True)
        gray_rgb = np.repeat(gray, 3, axis=2)
        mixed = gray_rgb * (1.0 - self.blend_strength) + rgb * self.blend_strength
        out = np.where(warm_mask[..., None], rgb, mixed)
        return transforms.functional.to_pil_image(np.clip(out, 0, 255).astype(np.uint8))


class FireContrastEnhance:
    """
    RECOMMENDED preprocessing for the final report run.

    Stage 1 — boost warm pixels: amplifies saturation + brightness of fire
               tones so red/orange/yellow stands out clearly.
    Stage 2 — de-emphasise background: partially desaturates cool/neutral
               pixels so the model is not distracted by non-fire colour cues,
               while still retaining weak background context (unlike full grey).

    Config keys:
      use_fire_contrast_enhance: true
      fire_enhance_saturation_scale: 1.4
      fire_enhance_value_scale: 1.12
      fire_enhance_bg_desat_factor: 0.4   (0 = full grey bg, 1 = no change)
      fire_enhance_warm_hue_high: 0.18
      fire_enhance_saturation_threshold: 0.2
    """

    def __init__(
        self,
        saturation_scale: float = 1.4,
        value_scale: float = 1.12,
        bg_desat_factor: float = 0.4,
        warm_hue_high: float = 0.18,
        saturation_threshold: float = 0.2,
    ) -> None:
        self.saturation_scale = saturation_scale
        self.value_scale = value_scale
        self.bg_desat_factor = bg_desat_factor
        self.warm_hue_high = warm_hue_high
        self.saturation_threshold = saturation_threshold

    def __call__(self, img):
        hsv = np.array(img.convert("HSV"), dtype=np.float32) / 255.0
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        warm_mask = (h <= self.warm_hue_high) & (s >= self.saturation_threshold)

        # Stage 1: boost warm pixels
        hsv[..., 1] = np.where(warm_mask, np.clip(s * self.saturation_scale, 0.0, 1.0), s)
        hsv[..., 2] = np.where(warm_mask, np.clip(v * self.value_scale, 0.0, 1.0), v)

        # Stage 2: de-emphasise background
        hsv[..., 1] = np.where(~warm_mask, hsv[..., 1] * self.bg_desat_factor, hsv[..., 1])

        result = (hsv * 255.0).astype(np.uint8)
        return transforms.functional.to_pil_image(result, mode="HSV").convert("RGB")


# ── Transform builder ────────────────────────────────────────────────────────

def _build_color_steps(preprocessing_cfg: dict) -> list:
    """Return colour preprocessing transforms from config. FireContrastEnhance
    takes priority — do not stack all three transforms together."""
    steps = []
    if preprocessing_cfg.get("use_fire_contrast_enhance", False):
        steps.append(FireContrastEnhance(
            saturation_scale=preprocessing_cfg.get("fire_enhance_saturation_scale", 1.4),
            value_scale=preprocessing_cfg.get("fire_enhance_value_scale", 1.12),
            bg_desat_factor=preprocessing_cfg.get("fire_enhance_bg_desat_factor", 0.4),
            warm_hue_high=preprocessing_cfg.get("fire_enhance_warm_hue_high", 0.18),
            saturation_threshold=preprocessing_cfg.get("fire_enhance_saturation_threshold", 0.2),
        ))
    else:
        if preprocessing_cfg.get("use_warm_color_boost", False):
            steps.append(WarmColorBoost(
                saturation_scale=preprocessing_cfg.get("warm_boost_saturation_scale", 1.35),
                value_scale=preprocessing_cfg.get("warm_boost_value_scale", 1.1),
            ))
        if preprocessing_cfg.get("use_selective_grayscale", False):
            steps.append(WarmColorSelectiveGrayscale(
                saturation_threshold=preprocessing_cfg.get("selective_grayscale_saturation_threshold", 0.35),
                warm_hue_low=preprocessing_cfg.get("selective_grayscale_warm_hue_low", 0.0),
                warm_hue_high=preprocessing_cfg.get("selective_grayscale_warm_hue_high", 0.18),
                min_value=preprocessing_cfg.get("selective_grayscale_min_value", 0.15),
                blend_strength=preprocessing_cfg.get("selective_grayscale_blend_strength", 1.0),
            ))
    return steps


def _build_transform(config: dict, train: bool) -> transforms.Compose:
    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    preprocessing_cfg = data_cfg.get("preprocessing", {})

    pipeline = [transforms.Resize((image_size, image_size))]

    if train:
        pipeline.extend(_build_color_steps(preprocessing_cfg))
        pipeline.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),      # satellite has no canonical up
            transforms.RandomRotation(15),
            transforms.ColorJitter(               # simulate sensor/lighting variation
                brightness=0.15,
                contrast=0.15,
                saturation=0.1,
            ),
        ])
    else:
        if preprocessing_cfg.get("apply_color_preprocess_on_eval", True):
            pipeline.extend(_build_color_steps(preprocessing_cfg))

    pipeline.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(pipeline)


def create_dataloader(config: dict, split: str) -> DataLoader:
    split_name = config["data"][f"{split}_split"]
    root = Path(config["data"]["root_dir"]) / split_name
    dataset = datasets.ImageFolder(
        root=str(root),
        transform=_build_transform(config, train=(split == "train")),
    )

    num_workers = config["data"].get("num_workers", 0)

    # BUG FIX: pin_memory and persistent_workers both crash when num_workers=0.
    # Guard them so the code works on CPU (num_workers=0) and GPU alike.
    return DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )