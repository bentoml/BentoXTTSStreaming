from __future__ import annotations

import os
import pathlib
import typing as t
from pathlib import Path

import bentoml
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from streaming_utils import StreamingInputs, predict_streaming_generator


MODEL_ID = "coqui/XTTS-v2"

runtime_image = bentoml.images.PythonImage(
    python_version="3.11",
    base_image="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
).requirements_file("requirements.txt")

streaming_app = FastAPI()


@bentoml.service(
    name="bentoxtts-service",
    image=runtime_image,
    envs=[{"name": "COQUI_TOS_AGREED", "value": "1"}],
    traffic={
        "timeout": 300,
        "concurrency": 3,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
    workers=3,
)
@bentoml.asgi_app(streaming_app, path="/tts")
class XTTSStreaming:

    hf_model = bentoml.models.HuggingFaceModel(MODEL_ID)

    def __init__(self) -> None:
        import torch

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        config = XttsConfig()
        config.load_json(os.path.join(self.hf_model, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_dir=self.hf_model,
            eval=True,
            use_deepspeed=True if self.device == "cuda" else False
        )
        self.model.to(self.device)
        print("XTTS Loaded.", flush=True)

        cdir = pathlib.Path(__file__).parent.resolve()
        voice_path = cdir / "female.wav"
        _t = self.model.get_conditioning_latents(voice_path)
        self.gpt_cond_latent = _t[0]
        self.speaker_embedding = _t[1]

        
    @streaming_app.post("/stream")
    def tts_stream(self, inp: StreamingInputs):
        gen = predict_streaming_generator(
            model=self.model,
            text=inp.text,
            language=inp.language,
            speaker_embedding=self.speaker_embedding,
            gpt_cond_latent=self.gpt_cond_latent,
            stream_chunk_size=inp.stream_chunk_size,
            add_wav_header=inp.add_wav_header,
        )
        return StreamingResponse(gen, media_type="audio/wav")
