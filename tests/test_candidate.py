"""Small source/JSON/identity tests only; no torch, decoder, model or data loads."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import infer
from future_gain.runtime import require_slurm_compute
from future_gain.sampling.budget import ExactQuotaController,EmpiricalThreshold
from future_gain.calibration.ridge import RidgeRouter


class FakeArray:
    shape=(2,64,64,3)
    dtype='uint8'


class FakeVideo:
    def __init__(self,count=30):self.count=count;self.requests=[];self.array=FakeArray()
    def __len__(self):return self.count
    def get_batch(self,indices):self.requests.append(indices);return SimpleNamespace(asnumpy=lambda:self.array)


class CandidateTests(unittest.TestCase):
    def test_candidate_source_manifest_and_only_runtime_text_delta(self):
        infer.verify_bundle();p=json.loads((ROOT/'core_provenance.json').read_text())
        self.assertEqual(len(p['files']),11)
        for name,record in p['files'].items():
            data=(ROOT/name).read_bytes();self.assertEqual(hashlib.sha256(data).hexdigest(),record['candidate_sha256'])
            if record['change']:
                change=record['change'];self.assertEqual(name,'src/future_gain/runtime.py')
                self.assertEqual(data.decode().count(change['after']),1)
                self.assertEqual(len(change['before_literal_sha256']),64)
                def normalized(value):
                    if isinstance(value,ast.AST):
                        fields={}
                        for key,item in ast.iter_fields(value):
                            if key=='type_params':
                                self.assertEqual(item,[]);continue
                            fields[key]=normalized(item)
                        if isinstance(value,ast.Constant) and value.value==change['after']:
                            fields['value']='<RUNTIME_LOCATION_MESSAGE>'
                        return {'node':type(value).__name__,**fields}
                    if isinstance(value,list):return [normalized(item) for item in value]
                    return value
                digest=hashlib.sha256(json.dumps(normalized(ast.parse(data.decode())),sort_keys=True,separators=(',',':')).encode()).hexdigest()
                self.assertEqual(digest,change['normalized_ast_sha256'])
            else:self.assertEqual(record['candidate_sha256'],record['original_sha256'])

    def test_exported_heads_and_cdfs_are_frozen_and_finite(self):
        value=infer.load_policy();self.assertEqual(set(value['policies']),{'current_gain','future_gain'})
        for method,entry in value['policies'].items():
            router=RidgeRouter.from_dict(entry['model']);self.assertEqual(router.to_dict(),entry['model'])
            scores=entry['validation_scores'];self.assertEqual(len(scores),1792)
            distribution=EmpiricalThreshold(scores)
            self.assertEqual(list(distribution.scores),sorted(scores))
            quota=ExactQuotaController(28,14,8,16,distribution,seed=123)
            decisions=[quota.choose(router.predict([i/100]*12)) for i in range(28)]
            self.assertEqual(sum(d.nfe_requested for d in decisions),336)
            self.assertEqual(sum(d.nfe_requested==16 for d in decisions),14);quota.assert_complete(336)

    def test_original_noise_identity_and_quota_seed_are_path_hash_independent(self):
        args={'dataset_revision':infer.REVISION,'original_video_id':'train/123','trajectory_seed':303}
        first,seed=infer.noise_and_quota_seed(**args);second,other=infer.noise_and_quota_seed(**args)
        self.assertEqual(first,second);self.assertEqual(seed,other)
        self.assertEqual(seed,int(infer.canonical_hash({'noise_identity':first.frame_hash(0),'kind':'quota_v1'})[:16],16))
        self.assertEqual(first.clip_frame_indices,tuple(range(30)))
        self.assertNotEqual(first.frame_hash(0),first.frame_hash(1))
        changed,_=infer.noise_and_quota_seed(**{**args,'trajectory_seed':404});self.assertNotEqual(first.frame_hash(0),changed.frame_hash(0))
        # The real call accepts only explicit logical identity: no file/checksum/selected method argument.
        tree=ast.parse((ROOT/'infer.py').read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='noise_and_quota_seed')
        self.assertEqual([a.attr for a in call.args],['dataset_revision','original_video_id','trajectory_seed'])

    def test_identity_rejects_revision_or_logical_id_substitution(self):
        for values in (('wrong','train/123',303),(infer.REVISION,'clip.mp4',303),(infer.REVISION,'../123',303),
                       (infer.REVISION,'train/123',True),(infer.REVISION,'train/123',-1)):
            with self.assertRaises(ValueError):infer.noise_and_quota_seed(*values)

    def test_prefix_reader_requests_only_first_two_native_frames(self):
        for count in (2,30,100):
            reader=FakeVideo(count);self.assertIs(infer.prefix_array(reader),reader.array);self.assertEqual(reader.requests,[[0,1]])
        with self.assertRaises(ValueError):infer.prefix_array(FakeVideo(1))
        reader=FakeVideo();reader.array.shape=(2,128,128,3)
        with self.assertRaises(ValueError):infer.prefix_array(reader)
        reader=FakeVideo();reader.array.dtype='float32'
        with self.assertRaises(ValueError):infer.prefix_array(reader)

    def test_runtime_guard_rejects_local_foreign_owner_and_unallocated_node(self):
        with patch('future_gain.runtime.platform.system',return_value='Darwin'):
            with self.assertRaises(RuntimeError):require_slurm_compute()
        for owner,state,nodes in (('other','RUNNING','compute1'),('synthetic','PENDING','compute1'),('synthetic','RUNNING','othernode')):
            results=[SimpleNamespace(stdout=f'{owner}|{state}|compute1\n'),SimpleNamespace(stdout=nodes+'\n')]
            with patch('future_gain.runtime.platform.system',return_value='Linux'),patch('future_gain.runtime.os.environ',{'SLURM_JOB_ID':'123'}), \
                 patch('future_gain.runtime.socket.gethostname',return_value='compute1'),patch('future_gain.runtime.getpass.getuser',return_value='synthetic'), \
                 patch('future_gain.runtime.subprocess.run',side_effect=results):
                with self.assertRaises(RuntimeError):require_slurm_compute()

    def test_offline_and_bytecode_configuration_precedes_torch(self):
        self.assertTrue(sys.dont_write_bytecode)
        self.assertEqual(infer.os.environ['PYTHONDONTWRITEBYTECODE'],'1');self.assertNotIn('torch',sys.modules)
        tree=ast.parse((ROOT/'infer.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        flat=list(ast.walk(main));decode=next(n.lineno for n in flat if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='prefix_array')
        far=next(n.lineno for n in flat if isinstance(n,ast.ImportFrom) and n.module=='future_gain.adapters.far')
        self.assertLess(decode,far)

    def test_cache_and_offline_settings_are_explicit_and_bounded(self):
        cache=ROOT.parent;far=cache/'synthetic_external_far'
        with patch.dict(infer.os.environ,{},clear=True),patch('infer.os.getuid',return_value=cache.stat().st_uid):
            settings=infer.offline_cache(cache,far)
            for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','HF_HUB_OFFLINE',
                        'HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','DIFFUSERS_OFFLINE','PYTHONDONTWRITEBYTECODE'):
                self.assertEqual(settings[key],'1')
            self.assertEqual(settings['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
            for key in ('HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
                self.assertTrue(Path(settings[key]).is_relative_to(cache))
            self.assertNotIn('HOME',settings)
            with self.assertRaises(ValueError):infer.offline_cache(ROOT,far)
            with self.assertRaises(ValueError):infer.offline_cache(cache,cache)



if __name__=='__main__':unittest.main()
