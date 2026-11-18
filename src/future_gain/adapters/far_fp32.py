"""Explicit native FP32 FAR-B construction; the BF16 adapter is unchanged."""
import json
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4
from future_gain.adapters.far import (FARAdapter, _sha256, _verify_far_module_origins,
    BAIR_TRANSFORMER_SHA256, BAIR_CODEC_SHA256)
from future_gain.audit import audit_upstream
from future_gain.runtime import require_slurm_compute

class FARFP32Adapter(FARAdapter):
    """New numerical setting, using inherited native Euler/KV/packet operations."""
    @classmethod
    def from_pretrained(cls, upstream_root, transformer_checkpoint, codec_checkpoint, *, device='cuda:0', mixed_precision='no'):
        import sys
        allocation=require_slurm_compute()
        if device!='cuda:0' or mixed_precision!='no':raise ValueError('Explicit single-visible FP32 setting required')
        root=Path(upstream_root).resolve(strict=True);audit_upstream(root)
        for name,want in [('diffusers','0.32.2'),('accelerate','1.4.0')]:
            if version(name)!=want:raise ValueError('Pinned dependency changed')
        paths=[Path(transformer_checkpoint).resolve(strict=True),Path(codec_checkpoint).resolve(strict=True)]
        hashes=[_sha256(p) for p in paths]
        if hashes!=[BAIR_TRANSFORMER_SHA256,BAIR_CODEC_SHA256]:raise ValueError('Pinned weights changed')
        _verify_far_module_origins(root)
        if str(root) not in sys.path:sys.path.insert(0,str(root))
        import torch
        import far
        from accelerate import Accelerator
        from diffusers import FlowMatchEulerDiscreteScheduler
        from far.models import build_model
        from far.pipelines.pipeline_far import FARPipeline
        _verify_far_module_origins(root)
        if [Path(p).resolve() for p in far.__path__]!=[root/'far']:raise ValueError('Foreign FAR package')
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,0) or 'V100' not in torch.cuda.get_device_name():raise ValueError('One actual V100 required; no precision/device fallback')
        accelerator=Accelerator(mixed_precision='no')
        if accelerator.num_processes!=1 or accelerator.device.type!='cuda' or accelerator.mixed_precision!='no':raise ValueError('Native FP32 single process required')
        model=build_model('FAR_B')(short_term_ctx_winsize=30)
        model.load_state_dict(torch.load(paths[0],map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False);model.eval()
        model=accelerator.unwrap_model(accelerator.prepare(model))
        if next(model.parameters()).dtype!=torch.float32:raise ValueError('Transformer must remain native FP32')
        codec=build_model('MyAutoencoderDC').from_config(json.loads((root/'options/model_cfg/dcae/model_8x_c32_config.json').read_text()))
        codec.load_state_dict(torch.load(paths[1],map_location='cpu',weights_only=True),strict=True)
        codec.to(accelerator.device,dtype=torch.float32);codec.requires_grad_(False);codec.eval();codec.enable_slicing()
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(json.loads((root/'options/model_cfg/far/scheduler_config.json').read_text()))
        pipeline=FARPipeline(transformer=model,vae=codec,scheduler=scheduler);pipeline.execution_device=accelerator.device;pipeline.set_progress_bar_config(disable=True)
        self=cls();self.torch=torch;self.pipeline=pipeline;self.accelerator=accelerator;self.device=next(model.parameters()).device;self.dtype=torch.float32
        self.owner_id=uuid4().hex;self.denoiser_calls=self.vae_encode_calls=self.vae_decode_calls=0
        self.checkpoint_hashes={'transformer':hashes[0],'codec':hashes[1]};self._scheduler_class=FlowMatchEulerDiscreteScheduler;self._scheduler_config=dict(scheduler.config);self._first_grid=self._new_scheduler(50)
        if float(self._first_grid.timesteps[0])!=1000 or float(self._first_grid.sigmas[0])!=1:raise ValueError('Original first Euler point changed')
        def count(_module,_inputs):self.denoiser_calls+=1
        self._call_counter_handle=model.register_forward_pre_hook(count)
        return self
