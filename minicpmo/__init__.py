"""Vendored MiniCPM-o 4.5 model code. This package IS the source of truth for inference:
weights load from the read-only checkpoint dir, code comes from here (no trust_remote_code,
no shadow dirs, no patches). Edit modeling_minicpmo.py / utils.py directly; git tracks it.

Divergences from the upstream checkpoint code are marked with `# duplex_minicpm:` comments.
Current divergences:
  - chat(): audio input no longer force-enables the TTS template (tool calling stays alive);
    the spoken register is only used when generate_audio=True or use_tts_template=True.
"""

import os

import torch

from .configuration_minicpmo import MiniCPMOConfig, MiniCPMTTSConfig
from .modeling_minicpmo import MiniCPMO, MiniCPMODuplex
from .processing_minicpmo import MiniCPMAAudioProcessor, MiniCPMOProcessor, MiniCPMVImageProcessor
from .tokenization_minicpmo_fast import MiniCPMOTokenizerFast

BASE_WEIGHTS = "/dataset_rc_rl/suns7@xiaopeng.com/models/MiniCPM-o-4_5"  # upstream, vocab 151748, read-only
WEIGHTS = "/dataset_rc_rl/suns7@xiaopeng.com/models/MiniCPM-o-4_5-act"  # default: +<|act_end|> (151749), training base
CHAT_TEMPLATE = os.path.join(os.path.dirname(__file__), "chat_template.jinja")


def load_model(weights=WEIGHTS, device="cuda", dtype=torch.bfloat16, **config_overrides):
    """Load MiniCPM-o 4.5 with THIS package's code — weights and raw configs come from the
    checkpoint dir; every class AND the chat template come from this repo. The processor is
    built here explicitly so prepare_processor()'s trust_remote_code fallback never runs.
    Returns (model, tokenizer); the full processor sits at model.processor."""
    cfg = MiniCPMOConfig.from_pretrained(weights, **config_overrides)
    model = MiniCPMO.from_pretrained(weights, config=cfg, torch_dtype=dtype, attn_implementation="sdpa")
    model.eval()
    if device:
        model = model.to(device)

    tokenizer = MiniCPMOTokenizerFast.from_pretrained(weights)
    tokenizer.chat_template = open(CHAT_TEMPLATE).read()

    processor = MiniCPMOProcessor(
        image_processor=MiniCPMVImageProcessor.from_pretrained(weights),
        audio_processor=MiniCPMAAudioProcessor.from_pretrained(weights),
        tokenizer=tokenizer,
        audio_pool_step=cfg.audio_pool_step,
    )
    model.prepare_processor(processor=processor, tokenizer=tokenizer)
    return model, tokenizer
