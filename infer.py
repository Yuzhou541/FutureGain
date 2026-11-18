#!/usr/bin/env python3
"""Frozen H0/H3 BAIR 2-to-28 inference with exact per-video compute budgets."""
import sys
sys.dont_write_bytecode=True
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from future_gain.audit import canonical_hash,sha256_file
from future_gain.calibration.features import FeatureExtractor
from future_gain.calibration.ridge import RidgeRouter
from future_gain.sampling.noise import NoiseIdentity,NoiseBank,tensor_sha256
from future_gain.sampling.budget import EmpiricalThreshold
from future_gain.sampling.online import OnlineAllocation,generate_video
from future_gain.runtime import require_slurm_compute

POLICY_SHA='73f7a79478ff4bb980b69d8ca4ee994716d1bd29b129e78b89386a878dbbdb56'
REVISION='776ad3b7d6591b06a01b8f91b1d89496445db77c'
METHODS=('current_gain','future_gain')


def require(value,message):
    if not value:raise ValueError(message)


def write_json(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def verify_bundle():
    manifest=json.loads((ROOT/'release_manifest.json').read_text());canonical_hash(manifest)
    require(manifest['status']=='source_release',
            'This entrypoint needs its reviewed release manifest')
    for name,digest in manifest['files'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe candidate file name')
        path=ROOT/p;require(path.is_file() and not path.is_symlink() and sha256_file(path)==digest,'Candidate source/export changed: '+name)
    for name,module in tuple(sys.modules.items()):
        if name=='future_gain' or name.startswith('future_gain.'):
            source=getattr(module,'__file__',None)
            if source:require(Path(source).resolve().is_relative_to(ROOT/'src'),'Foreign future_gain module')
            namespace=getattr(module,'__path__',None)
            if namespace:require(all(Path(p).resolve().is_relative_to(ROOT/'src') for p in namespace),'Foreign future_gain namespace')
    return manifest


def load_policy():
    require(sha256_file(ROOT/'policy.json')==POLICY_SHA,'Frozen head/CDF export changed')
    value=json.loads((ROOT/'policy.json').read_text());canonical_hash(value)
    require(value['schema_version']==1 and value['status']=='inherited_frozen_policy_no_refit'
            and list(value['policies'])==list(METHODS) and value['dataset']['revision']==REVISION
            and value['controller']=={'low_nfe':8,'high_nfe':16,'high_frames':14,'total_calls':336,'quota_rng_kind':'quota_v1'}
            and value['cdf_metadata']=={'weighting':'equal_per_calibration_val_context_seed_frame; all_28_frames',
                'tie_tolerance':1e-12,'quantile':'inverse_empirical_cdf_nearest_rank'},'Frozen deployment policy changed')
    for method,h in zip(METHODS,(0,3)):
        entry=value['policies'][method]
        require(entry['lookahead']==h and len(entry['validation_scores'])==1792,'Wrong inherited head/CDF scope')
        RidgeRouter.from_dict(entry['model']);EmpiricalThreshold(entry['validation_scores'])
    return value


def noise_and_quota_seed(dataset_revision,original_video_id,trajectory_seed):
    require(dataset_revision==REVISION,'Only the original declared BAIR revision is supported')
    require(type(original_video_id)is str and re.fullmatch(r'(train|test)/[0-9]+',original_video_id) is not None,
            'Supply the original BAIR train/<id> or test/<id>, never a filename-derived identity')
    require(type(trajectory_seed)is int and trajectory_seed>=0,'Explicit nonnegative trajectory seed required')
    nid=NoiseIdentity(dataset_revision,original_video_id,tuple(range(30)),trajectory_seed)
    control_seed=int(canonical_hash({'noise_identity':nid.frame_hash(0),'kind':'quota_v1'})[:16],16)
    return nid,control_seed


def prefix_array(reader):
    require(len(reader)>=2,'Need two native observed frames')
    # Decoder is called only for0/1, before any FAR import can change its bridge.
    value=reader.get_batch([0,1]).asnumpy()
    require(value.shape==(2,64,64,3) and str(value.dtype)=='uint8','Native64x64 RGB required; no resize, crop or range fallback')
    return value


def offline_cache(cache,far):
    require(cache.is_dir() and cache.stat().st_uid==os.getuid() and not cache.is_relative_to(ROOT)
            and not cache.is_relative_to(far),'Supply an existing owned cache outside source/FAR checkouts')
    require('torch' not in sys.modules,'Configure offline/cache/deterministic settings before torch import')
    settings={'HF_HOME':str(cache/'huggingface'),'HF_HUB_CACHE':str(cache/'huggingface'/'hub'),
        'HUGGINGFACE_HUB_CACHE':str(cache/'huggingface'/'hub'),'XDG_CACHE_HOME':str(cache/'xdg'),
        'TORCH_HOME':str(cache/'torch'),'CUDA_CACHE_PATH':str(cache/'cuda'),
        'TRITON_CACHE_DIR':str(cache/'triton'),'TORCHINDUCTOR_CACHE_DIR':str(cache/'torchinductor'),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'DIFFUSERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    os.environ.update(settings)
    return settings


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    for name in ('far-root','transformer-checkpoint','codec-checkpoint','input-clip','cache-dir','output'):
        result.add_argument('--'+name,type=Path,required=True)
    result.add_argument('--clip-sha256',required=True);result.add_argument('--dataset-revision',required=True)
    result.add_argument('--original-video-id',required=True);result.add_argument('--trajectory-seed',type=int,required=True)
    result.add_argument('--method',choices=METHODS,required=True)
    return result


def main():
    args=parser().parse_args()
    require(re.fullmatch(r'[0-9a-f]{64}',args.clip_sha256) is not None,'Explicit lowercase clip SHA256 required')
    nid,control_seed=noise_and_quota_seed(args.dataset_revision,args.original_video_id,args.trajectory_seed)
    allocation=require_slurm_compute()  # Before any dataset, checkpoint or CUDA import/load.
    require(sys.version_info[:3]==(3,10,12),'Frozen inference requires the original Python3.10.12 deployment environment')
    manifest=verify_bundle();policy=load_policy()
    far=args.far_root.resolve(strict=True);cache=args.cache_dir.resolve(strict=True)
    require(not far.is_relative_to(ROOT),'Supply FAR externally; no external model source is bundled')
    settings=offline_cache(cache,far)
    output=args.output.absolute();parent=output.parent.resolve(strict=True)
    require(parent==output.parent and parent.stat().st_uid==os.getuid() and not output.exists()
            and not output.is_relative_to(ROOT) and not output.is_relative_to(far),'New owned output outside both source checkouts required')
    output.mkdir();started=time.perf_counter();adapter=None
    record={'status':'running','allocation':allocation,'method':args.method,'release_manifest_sha256':sha256_file(ROOT/'release_manifest.json'),
        'generator_calls':0,'vae_encode_calls':0,'vae_decode_calls':0,'metric_model_calls':0,
        'counter_scope':'tracked adapter forward/codec entries; failed calls may be partial; excludes setup and allocation cost'}
    write_json(output/'execution.started.json',record)
    try:
        clip=args.input_clip.resolve(strict=True)
        require(clip.is_file() and not clip.is_relative_to(ROOT) and sha256_file(clip)==args.clip_sha256,'Input clip bytes changed')
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.init()
        from decord import VideoReader,cpu
        observed_array=prefix_array(VideoReader(str(clip),ctx=cpu(0),num_threads=1))
        observed=torch.from_numpy(observed_array.copy()).permute(0,3,1,2).float().unsqueeze(0)/255
        # Import/load FAR only after prefix decoding; its namespace may change decord's bridge.
        from future_gain.adapters.far import FARAdapter
        adapter=FARAdapter.from_pretrained(far,args.transformer_checkpoint,args.codec_checkpoint)
        runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'diffusers':version('diffusers'),'precision':'bf16_autocast_fp32_transformer','tf32':False,'deterministic':True}
        require(runtime==policy['generation']['runtime'] and adapter.checkpoint_hashes==policy['generation']['checkpoint_sha256'],
                'Original runtime/GPU/checkpoint identity required; no numerical fallback')
        entry=policy['policies'][args.method];extractor=FeatureExtractor();router=RidgeRouter.from_dict(entry['model'])
        bank=NoiseBank(nid,(1,1,32,8,8),device=str(adapter.device),dtype='bfloat16')
        # No path or clip checksum enters noise, features, controller or quota RNG.
        online=OnlineAllocation(kind='future_gain',validation_scores=entry['validation_scores'],high_frames=14,
            seed=control_seed,extractor=extractor,router=router)
        value=generate_video(adapter,observed,bank,policy=online)
        require(value['calls']==adapter.denoiser_calls==336 and value['vae_encode_calls']==adapter.vae_encode_calls==1
                and value['vae_decode_calls']==adapter.vae_decode_calls==1 and value['controller_calls']==28
                and [r['nfe'] for r in value['trace']].count(16)==14
                and [r['nfe'] for r in value['trace']].count(8)==14,'Actual fixed budget differs')
        require(value['rgb_full'].shape==(1,30,3,64,64) and value['latents'].shape==(1,30,32,8,8)
                and value['rgb_full'].dtype==value['latents'].dtype==torch.bfloat16
                and value['rgb_full'].device.type==value['latents'].device.type=='cpu'
                and torch.isfinite(value['latents']).all(),'Native lossless output shape/precision differs')
        metadata={'status':'completed','candidate_status':'frozen_policy_source_release',
            'method':args.method,'lookahead':entry['lookahead'],'policy_export_sha256':POLICY_SHA,
            'release_manifest_sha256':record['release_manifest_sha256'],'core_provenance_sha256':sha256_file(ROOT/'core_provenance.json'),
            'clip':{'path':str(clip),'sha256':args.clip_sha256},'decoded_input_frame_indices':[0,1],
            'input_prefix_tensor_sha256':tensor_sha256(observed),'noise_identity':asdict(nid),'noise_manifest':bank.manifest(28),
            'quota_seed':control_seed,'runtime':runtime,'decord_version':version('decord'),'python_version':sys.version,
            'far_commit':policy['generation']['upstream_commit'],'checkpoint_sha256':adapter.checkpoint_hashes,
            'controller_model_sha256':canonical_hash(entry['model']),'cdf_sha256':canonical_hash(entry['validation_scores']),
            'actual_calls':336,'vae_encode_calls':1,'vae_decode_calls':1,'controller_calls':28,'metric_model_calls':0,
            'timing_status':'single_candidate_run_diagnostic_not_controlled_comparison','cache_environment':settings}
        payload=output/'native.pt'
        with payload.open('xb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
        write_json(output/'trace.json',value['trace']);write_json(output/'metadata.json',metadata)
        require(sha256_file(clip)==args.clip_sha256,'Input changed during inference')
        require(subprocess.check_output(['git','-C',str(far),'rev-parse','HEAD'],text=True).strip()==policy['generation']['upstream_commit']
                and not subprocess.check_output(['git','-C',str(far),'status','--porcelain','--untracked-files=all'],text=True).strip(),
                'FAR checkout changed; preserve failure evidence')
        verify_bundle()
        completion={'status':'completed','release_manifest_sha256':record['release_manifest_sha256'],'policy_export_sha256':POLICY_SHA,
            'outputs':{name:sha256_file(output/name) for name in ('native.pt','trace.json','metadata.json')},
            'generator_calls':336,'metric_model_calls':0,'exact_relocated_equality':'not_verified_by_this_entrypoint'}
        write_json(output/'completion.json',completion);record.update(status='completed',completion_sha256=sha256_file(output/'completion.json'))
    except Exception as error:record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        if adapter is not None:record.update(generator_calls=adapter.denoiser_calls,vae_encode_calls=adapter.vae_encode_calls,vae_decode_calls=adapter.vae_decode_calls)
        record['complete_rollout_cost_verified']=record['status']=='completed'
        record['retry_requires_cost_audit']=record['status']!='completed'
        record['wall_seconds']=time.perf_counter()-started;write_json(output/'execution.finished.json',record)
    if record['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
