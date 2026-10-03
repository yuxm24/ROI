import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from zvla.posttrain.recipes import BENCHMARKS, recipe
from zvla.posttrain.prepare import codec_statistics
from zvla.posttrain.sources import convert_rows
from zvla.datasets.benchmark import BenchmarkDataset
from zvla.datasets.utils.shard_samplers import FaMShardAwareBatchSampler
from zvla.actions.mixed_native7 import codec_class
from zvla.models.fam_wam import FaMWAMConfig, FaMWAMPostTrainModel, FaMWAMPolicy
from zvla.models.fam_wam.deployment import export_deployment_artifact, load_deployment_artifact


def rows(bench, n=9):
    if bench in ('libero','libero-plus'):
        state, action = np.zeros((n,8)), np.zeros((n,7))
        state[:,0] = np.arange(n)*.01; action[:,0] = np.arange(n)*.1
    elif bench=='metaworld':
        state, action = np.zeros((n,4)), np.zeros((n,4))
        action[:,0] = np.arange(n)*.3
    elif bench=='robodojo':
        state=np.zeros((n,16));state[:,[3,11]]=1;action=state.copy();action[:,0]=np.arange(n)*.01
    else:
        state=np.zeros((n,20));state[:,[3,7,13,17]]=1;action=state.copy();action[:,0]=np.arange(n)*.01
    return convert_rows(bench,state,action)


def dataset(tmp_path, bench):
    state,target,sv,av=rows(bench)
    shard=tmp_path/'episodes/000000';shard.mkdir(parents=True)
    for key,arr in dict(states=state,targets=target,state_valid=sv,target_valid=av,
        video=np.zeros((3,48,42,14),np.float16), geometry=np.zeros((3,3,49,2048),np.float16)).items():
        np.save(shard/(key+'.npy'),arr)
    np.save(tmp_path/'text_embeddings.npy',np.zeros((1,64,4096),np.float16))
    (tmp_path/'dataset.json').write_text(json.dumps(dict(benchmark=bench,dt=BENCHMARKS[bench]['dt'],
        camera_count=len(BENCHMARKS[bench]['cameras']),episodes=[dict(shard='episodes/000000',native_rows=9,text_ids=[0])])))
    return BenchmarkDataset(str(tmp_path))


@pytest.mark.parametrize('bench',BENCHMARKS)
def test_native_action_alignment_tail_and_codec(tmp_path,bench):
    ds=dataset(tmp_path,bench)
    assert len(ds)==3
    data=rows(bench)
    sample=ds[2]
    torch.testing.assert_close(sample['target_state_abs'][0],torch.from_numpy(data[1][8]))
    assert not sample['target_state_abs_valid'][1:].any()
    assert not sample['future_spatial_valid'].any()
    assert sample['geometry_valid'].sum()==len(BENCHMARKS[bench]['cameras'])*49
    assert not ds[-1]['target_state_abs_valid'].any()
    stats=codec_statistics(bench,[data])
    cfg=FaMWAMConfig.tiny(metaworld_native4=bench=='metaworld',state_codec_embodiments=tuple(stats['embodiments']))
    codec=codec_class(cfg)(stats,embodiments=cfg.state_codec_embodiments)
    args=[sample[k][None] for k in ('current_state_abs','current_state_abs_valid','target_state_abs','target_state_abs_valid','embodiment_id')]
    encoded,valid,_,_=codec.encode_training_targets(*args)
    decoded,dv=codec.decode_predictions(encoded,args[0],args[1],args[4],valid)
    torch.testing.assert_close(decoded[dv],args[2][dv],atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('world',[1,2,3,7])
def test_arbitrary_rank_count_exact_once_and_resume(tmp_path,world):
    ds=dataset(tmp_path,'libero')
    draws=[]
    for rank in range(world):
        sampler=FaMShardAwareBatchSampler(ds,2,world,rank,drop_last=False,exact_once=True,global_batch_mixing=True)
        iterator = iter(sampler)
        first = next(iterator)
        cursor = sampler.state_dict()
        rest = list(iterator)
        draws += [x for batch in [first]+rest for x in batch if x!=-1]
        restored=FaMShardAwareBatchSampler(ds,2,world,rank,drop_last=False,exact_once=True,global_batch_mixing=True)
        restored.load_state_dict(cursor)
        assert list(restored)==rest
    assert sorted(draws)==list(range(len(ds)))


@pytest.mark.parametrize('bench',BENCHMARKS)
def test_recipe(bench):
    cfg=recipe(bench)
    assert cfg['training']['grad_accumulation_steps']==1
    assert cfg['model']['pretrained_checkpoint'].endswith('zvla-base.safetensors')
    FaMWAMConfig.from_dict(cfg['model']['config'])


def test_checkpoint_loader_preserves_target_codec_and_derives_native4(tmp_path):
    model=FaMWAMPostTrainModel(FaMWAMConfig.tiny().as_dict())
    source={k:v.contiguous() for k,v in model.state_dict().items() if not k.startswith('state_codec.')}
    path=tmp_path/'base.safetensors';save_file(source,str(path))
    cfg=FaMWAMConfig.tiny(metaworld_native4=True,state_codec_embodiments=('metaworld_sawyer_applied4',))
    loaded=FaMWAMPostTrainModel(cfg.as_dict(),pretrained_checkpoint=str(path))
    expected={k:v.clone() for k,v in loaded.state_codec.state_dict().items()}
    loaded.load_pretrained_weights()
    for k,v in loaded.state_codec.state_dict().items():assert torch.equal(v,expected[k])
    for k,v in loaded.state_dict().items():
        if k.startswith('state_codec.'):continue
        ref=source[k.replace('native4_','native7_')]
        if 'native4_' in k:
            cols=torch.tensor([0,1,2,6])
            ref=ref.index_select(0 if 'output_projection' in k else 1,cols) if 'output_projection' in k or k.endswith('.weight') else ref
        assert torch.equal(v,ref),k


def test_deployment_roundtrip(tmp_path):
    policy=FaMWAMPolicy(FaMWAMConfig.tiny())
    export_deployment_artifact(policy,tmp_path,action_schema={'benchmark':'libero'},camera_schema={},normalization_stats={},vae_encoder_provenance={})
    loaded,manifest=load_deployment_artifact(tmp_path)
    assert manifest['provenance']=={}
    for k,v in policy.state_dict().items():assert torch.equal(v,loaded.state_dict()[k])


def test_training_loader_constructs_real_registered_batch(tmp_path):
    import zvla.collators
    from zvla.datasets.utils.dataloader_builder import build_train_dataloader
    dataset(tmp_path,'libero')
    cfg=recipe('libero',data_root=tmp_path,batch_size=2)['data']['train']
    loader=build_train_dataloader(cfg['dataset'],cfg['collator'],2,num_workers=0)
    batch=next(iter(loader))
    assert batch['target_state_abs'].shape==(2,16,20)
    assert batch['geometry_features'].shape==(2,147,2048)


def test_lerobot_v3_file_offsets_and_train_split(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from zvla.posttrain.sources import lerobot_episodes
    meta=tmp_path/'meta';(meta/'episodes').mkdir(parents=True)
    cameras=('observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist')
    info=dict(codebase_version='v3.0',splits={'train':'1:2'},fps=25,features={},
              data_path='file-{file_index}.parquet',video_path='{video_key}-{file_index}.mp4')
    (meta/'info.json').write_text(json.dumps(info))
    state=np.zeros((8,16));state[:,[3,11]]=1
    action=state.copy();action[:,0]=np.arange(8)*.1
    pq.write_table(pa.table({'observation.state':state.tolist(),'action':action.tolist()}),tmp_path/'file-0.parquet')
    episodes=[]
    for ep in range(2):
        row=dict(episode_index=ep,length=4,tasks=['move the object'],**{'data/chunk_index':0,'data/file_index':0,
            'dataset_from_index':100+4*ep,'dataset_to_index':104+4*ep})
        for cam in cameras:
            row.update({'videos/'+cam+'/chunk_index':0,'videos/'+cam+'/file_index':0,'videos/'+cam+'/from_timestamp':.16*ep})
        episodes.append(row)
    pq.write_table(pa.Table.from_pylist(episodes),meta/'episodes/episodes.parquet')
    selected=list(lerobot_episodes(tmp_path,'robodojo'))
    assert len(selected)==1 and selected[0][0]=='1'
    np.testing.assert_allclose(selected[0][3][:,0],action[4:,0])


def test_libero_rlds_preserves_every_native_command(tmp_path):
    import struct
    from zvla.posttrain.tfrecord import TF_EXAMPLE
    from zvla.posttrain.sources import libero_episodes
    ex=TF_EXAMPLE();f=ex.features.feature
    f['steps/action'].float_list.value.extend(np.arange(21,dtype=np.float32)/21)
    f['steps/observation/state'].float_list.value.extend(np.zeros(24))
    f['steps/language_instruction'].bytes_list.value.extend([b'move the bowl']*3)
    value=ex.SerializeToString()
    (tmp_path/'train.tfrecord-00000').write_bytes(struct.pack('<Q',len(value))+b'\0'*4+value+b'\0'*4)
    row=next(libero_episodes(tmp_path))
    assert row[2].shape==(3,20)
    np.testing.assert_allclose(row[3][:,:7],(np.arange(21,dtype=np.float32)/21).reshape(3,7))


def test_decode_video_selects_only_requested_episode(tmp_path):
    import av
    from zvla.posttrain.sources import decode_video
    path=tmp_path/'shared.mp4'
    with av.open(str(path),'w') as out:
        stream=out.add_stream('libx264',rate=25);stream.width=32;stream.height=32;stream.pix_fmt='yuv420p'
        for i in range(12):
            frame=av.VideoFrame.from_ndarray(np.full((32,32,3),i*16,np.uint8),format='rgb24')
            for packet in stream.encode(frame):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    selected=decode_video(path,4/25,4,25)
    assert selected.shape==(4,32,32,3)
    np.testing.assert_allclose(selected.mean(axis=(1,2,3)),np.arange(4,8)*16,atol=3)


def test_metaworld_episode_sampling_and_resume(tmp_path):
    from zvla.datasets.utils.sampler_builder import EpisodeBalancedBatchSampler
    ds=dataset(tmp_path,'metaworld')
    global_sampler=EpisodeBalancedBatchSampler(ds,6,1,0,steps=5)
    expected=list(global_sampler)
    shards=[]
    for rank in range(3):
        sampler=EpisodeBalancedBatchSampler(ds,2,3,rank,steps=5)
        iterator=iter(sampler);first=next(iterator)
        saved=sampler.state_dict();rest=list(iterator)
        restored=EpisodeBalancedBatchSampler(ds,2,3,rank,steps=5)
        restored.load_state_dict(saved)
        assert list(restored)==rest
        shards.append([first]+rest)
    assert [sum([shards[r][i] for r in range(3)],[]) for i in range(5)]==expected
    assert recipe('metaworld')['data']['train']['dataset']['sampler']['type']=='EpisodeBalancedBatchSampler'
