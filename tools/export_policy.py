#!/usr/bin/env python3
"""Export only inherited H0/H3 parameters and CDF inputs from exact small JSONs."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED={
 'fit_manifest.json':'546b35cc24326881fa901b538cabf2cae2172d5952e6b3ac4c49b58e9fa80405',
 'protocol.json':'1c183d85b035d7950c53e070a11e191b5848f4b33721e06f3416edf00231cef6',
 'routers.json':'470d806d758a28c879b251f361ac7dfe365e4b86b76a23291984be7a56f67f09',
 'validation_scores.json':'de859038d3ea64b2edfc53f3e505f545b4e69d74727e1ce559b0203df9558fd8'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def export(fit_dir):
    saved={}
    for name,digest in EXPECTED.items():
        path=fit_dir/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size>8*1024**2 or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Exact original small fit JSON required: '+name)
        saved[name]=json.loads(path.read_text());canonical_hash(saved[name])
    manifest,protocol,routers,scores=(saved[k] for k in EXPECTED)
    if (manifest['status']!='completed' or manifest['scope']!='full'
            or canonical_hash(protocol)!=manifest['protocol_sha256'] or routers['fit_protocol_sha256']!=manifest['protocol_sha256']
            or routers['validation_scores_sha256']!=EXPECTED['validation_scores.json']
            or any(manifest['outputs'][k]!=EXPECTED[k] for k in ('protocol.json','routers.json','validation_scores.json'))):
        raise ValueError('Frozen fit lineage differs')
    core=protocol['collector_protocol']['generation_core']
    return {'schema_version':1,'status':'inherited_frozen_policy_no_refit','source_file_sha256':EXPECTED,
        'fit_protocol_content_sha256':manifest['protocol_sha256'],
        'projection':{'models':'routers.json/models/{0,3}/model','cdf_inputs':'validation_scores.json/distributions/gain_H{0,3}',
            'preservation':'unchanged JSON numeric values and array order; omitted other heads and per-video validation rows'},
        'generation':{k:core[k] for k in ('upstream_commit','checkpoint_sha256','runtime','context_frames','prediction_frames','resolution','guidance_scale')},
        'dataset':{k:protocol['dataset'][k] for k in ('repo_id','revision')},
        'controller':{'low_nfe':8,'high_nfe':16,'high_frames':14,'total_calls':336,'quota_rng_kind':'quota_v1'},
        'cdf_metadata':{k:scores[k] for k in ('weighting','tie_tolerance','quantile')},
        'policies':{method:{'lookahead':int(h),'label_definition':routers['models'][h]['label_definition'],
            'model':routers['models'][h]['model'],'validation_scores':scores['distributions']['gain_H'+h]}
            for method,h in (('current_gain','0'),('future_gain','3'))}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fit-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args();value=export(args.fit_dir)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
