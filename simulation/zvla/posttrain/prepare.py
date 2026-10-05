"""Public raw demonstrations -> visual cache and target-robot statistics."""
import gc
import json
import os
import shutil
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from .cache import (
    ROOT_RECEIPT,
    PreparationIdentityChanged,
    atomic_json,
    cancel_new_shards,
    check_root_identity,
    clear_staging,
    episode_identity,
    existing_root_receipt,
    file_identity,
    identity_digest,
    make_root_receipt,
    output_lock,
    publish_shard,
    publish_summary,
    shard_path,
    validate_shard,
    validate_summary,
)
from .cache import (
    write_json as write_durable_json,
)
from .recipes import BENCHMARKS
from .sources import episodes


def codec_statistics(bench, rows):
    from zvla_runtime.robotwin_action20 import relative_action20

    from zvla.actions.mixed_native7 import MINMAX_FORMAT, MINMAX_NORMALIZATION, NATIVE7_LAYOUT
    from zvla.actions.state_trajectory import (
        ACTION20_COORDINATE_CONVENTION,
        INTERNAL_STATE_ACTION_LAYOUT,
        STATE_TRAJECTORY_LAYOUT,
    )
    spec = BENCHMARKS[bench]
    bounds = {m: [np.full(20, np.inf), np.full(20, -np.inf), np.zeros(20, bool)] for m in ('state', 'action')}
    for states, targets, sv, _av in rows:
        anchors = np.arange(0, len(states), 4)
        for mode, values, masks in [('state', states[anchors], sv[anchors])]:
            lo, hi, valid = bounds[mode]
            lo[:] = np.minimum(lo, np.where(masks, values, np.inf).min(0))
            hi[:] = np.maximum(hi, np.where(masks, values, -np.inf).max(0))
            valid |= masks.any(0)
        if bench in ('robotwin', 'robodojo'):
            # Fit the same current-observation -> 16 future command windows
            # that enter the action loss, including partial episode tails.
            for anchor in anchors:
                target = targets[anchor:anchor+16]
                action = relative_action20(np.broadcast_to(states[anchor], target.shape), target)
                lo, hi, valid = bounds['action']
                lo[:] = np.minimum(lo, action.min(0))
                hi[:] = np.maximum(hi, action.max(0))
                valid[:] = True
        else:
            lo, hi, valid = bounds['action']
            lo[:], hi[:] = -1, 1
            valid[:4 if bench == 'metaworld' else 7] = True
    result = dict(format=MINMAX_FORMAT, normalization=MINMAX_NORMALIZATION,
        state_layout=list(STATE_TRAJECTORY_LAYOUT), action_layout=list(INTERNAL_STATE_ACTION_LAYOUT),
        native7_layout=list(NATIVE7_LAYOUT), coordinate_convention=ACTION20_COORDINATE_CONVENTION,
        embodiments=[spec['robot']])
    if bench == 'metaworld':
        from metaworld_benchmark.contract import (
            ACTION_LAYOUT,
            PROTOCOL,
            STATE_SLOTS,
            STATS_FORMAT,
        )
        result.update(format=STATS_FORMAT, target_contract=PROTOCOL, state_slots=list(STATE_SLOTS),
                      action_layout=list(ACTION_LAYOUT), normalization='state_minmax_action_applied4_identity')
    for mode, (lo, hi, valid) in bounds.items():
        lo, hi = np.where(valid, lo, 0), np.where(valid, hi, 1)
        result.update({mode+'_min': [lo.tolist()], mode+'_max': [hi.tolist()],
            mode+'_stat_valid': [valid.tolist()], mode+'_degenerate': [((hi-lo)<1e-6).tolist()]})
    return result


def visual_cache(streams, vae, geometry, device):
    from zvla.vision.letterbox import resize_canonical_canvas, resize_with_pad_rgb
    from zvla.vision.vggt_features import encode_vggt_views
    from zvla.vision.wan_vae import encode_wan22_vae_camera_windows, wan_vae_rgb_window_bounds
    count = (len(streams[0])+3)//4
    latent = np.zeros((count, 48, 42, 14), np.float16)
    geo = np.zeros((count, 3, 49, 2048), np.float16)
    def image(rgb, size):
        return resize_canonical_canvas(resize_with_pad_rgb(rgb)[0], size)
    for c, frames in enumerate(streams):
        windows = (np.stack([image(f, 224) for f in frames[start:stop]])
                   for _, _, start, stop in wan_vae_rgb_window_bounds(count, window_latents=32))
        cursor = 0
        for values in encode_wan22_vae_camera_windows(vae, windows, device=device, latent_frames=count, window_latents=32):
            latent[cursor:cursor+len(values), :, 14*c:14*(c+1)] = values
            cursor += len(values)
    for anchor in range(count):
        rgb = np.stack([image(frames[4*anchor], 518) for frames in streams])[None]
        geo[anchor, :len(streams)] = encode_vggt_views(geometry, rgb, device)[0]
    return latent, geo


def prepare(args):
    from .asset_identity import prepare_asset_identity

    output = Path(args.output).expanduser().resolve()
    for name in ('source', 'wan_repo', 'wan_model', 'vggt_repo', 'vggt_model'):
        input_root = Path(getattr(args, name)).expanduser().resolve()
        if output == input_root or input_root in output.parents or output in input_root.parents:
            raise ValueError('Preparation output must be separate from source and encoder asset directories')
    limit = args.limit
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError('--limit must be a positive integer')
    spec = BENCHMARKS[args.bench]
    device = torch.device(args.device)

    def selected_episodes():
        iterator = episodes(args.source, args.bench)
        try:
            yield from islice(iterator, limit)
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()

    with output_lock(output):
        existing = existing_root_receipt(output, getattr(args, 'resume', False))
        assets = prepare_asset_identity(args)

        def verify_assets():
            try:
                current = prepare_asset_identity(args)
            except Exception as error:
                raise PreparationIdentityChanged('Preparation inputs became unreadable or changed during processing') from error
            if current != assets:
                raise PreparationIdentityChanged('Preparation source or encoder assets changed during processing')

        # Decode while each generator yield is active: RoboTwin owns an open
        # HDF5 handle until the generator advances. Retain only small receipts.
        planned, keys = [], set()
        for key, prompts, state, target, sv, av, decode in selected_episodes():
            if key in keys:
                raise ValueError(f'Duplicate preparation source episode key: {key}')
            keys.add(key)
            streams = decode()
            planned.append(episode_identity(key, prompts, state, target, sv, av,
                                            streams, len(spec['cameras'])))
            del streams
        if not planned:
            raise ValueError('Preparation source contains no selected episodes')
        config = {'benchmark': args.bench, 'spec': spec, 'limit': limit,
                  'device_type': device.type, 'assets': assets}
        # Normalize tuples and other JSON container distinctions once, so a
        # receipt read from disk compares exactly to the current invocation.
        config = json.loads(json.dumps(config, allow_nan=False))
        root = make_root_receipt(config, planned)
        check_root_identity(existing, root)
        root_digest = root['identity_sha256']
        receipts = [validate_shard(output, index, source, root_digest)
                    for index, source in enumerate(planned)]
        summary_complete = validate_summary(output, root_digest, receipts)
        verify_assets()
        if summary_complete:
            clear_staging(output)
            print(f'{args.bench}: preparation already complete and verified', flush=True)
            return
        output.mkdir(parents=True, exist_ok=True)
        if existing is None:
            atomic_json(output / ROOT_RECEIPT, root)
        clear_staging(output)

        new_shards, staging = [], None
        vae = geometry = encoder = table = None
        try:
            if any(receipt is None for receipt in receipts):
                from zvla.vision.vggt_features import load_vggt_aggregator
                from zvla.vision.wan_vae import load_wan22_vae
                vae = load_wan22_vae(Path(args.wan_model) / 'Wan2.2_VAE.pth', args.wan_repo, device)
                geometry = load_vggt_aggregator(args.vggt_repo, args.vggt_model, device)
                seen = 0
                for index, (key, prompts, state, target, sv, av, decode) in enumerate(selected_episodes()):
                    seen += 1
                    if index >= len(planned) or key != planned[index]['key']:
                        raise PreparationIdentityChanged('Preparation episode inventory changed during processing')
                    if receipts[index] is not None:
                        continue
                    streams = decode()
                    try:
                        current = episode_identity(key, prompts, state, target, sv, av,
                                                   streams, len(spec['cameras']))
                    except ValueError as error:
                        raise PreparationIdentityChanged('Preparation episode changed during processing') from error
                    if current != planned[index]:
                        raise PreparationIdentityChanged('Preparation episode content changed during processing')
                    video, geo = visual_cache(streams, vae, geometry, device)
                    del streams
                    receipts[index] = publish_shard(output, index, current, root_digest,
                        dict(states=state, targets=target, state_valid=sv,
                             target_valid=av, video=video, geometry=geo))
                    new_shards.append(index)
                    del video, geo
                    print(f'{args.bench}: prepared {index + 1} episodes', flush=True)
                if seen != len(planned):
                    raise PreparationIdentityChanged('Preparation episode inventory changed during processing')
            vae = geometry = None
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            records, texts, text_index = [], [], {}
            for index, source in enumerate(planned):
                ids = []
                for text in source['prompts']:
                    if text not in text_index:
                        text_index[text] = len(texts)
                        texts.append(text)
                    ids.append(text_index[text])
                records.append(dict(shard=str(shard_path(output, index).relative_to(output)),
                    source_key=source['key'], native_rows=source['native_rows'], text_ids=ids,
                    source_identity_sha256=source['identity_sha256'],
                    output_identity_sha256=identity_digest(receipts[index]['files'])))
            staging = Path(tempfile.mkdtemp(prefix='.summary-staging-', dir=output))

            def rows():
                for record in records:
                    yield tuple(np.load(output / record['shard'] / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                                for name in ('states', 'targets', 'state_valid', 'target_valid'))

            write_durable_json(staging / 'codec.json', codec_statistics(args.bench, rows()))
            from zvla.vision.wan_assets import _load_text_encoder
            encoder = _load_text_encoder(Path(args.wan_model), Path(args.wan_repo), device, initialize_on_cpu=True)
            table = np.lib.format.open_memmap(staging / 'text_embeddings.npy', mode='w+', dtype=np.float16,
                                            shape=(len(texts), 64, 4096))
            with torch.inference_mode():
                for index, prompt in enumerate(texts):
                    embedding = encoder([prompt], device)[0]
                    if embedding.ndim != 2 or embedding.shape[1] != 4096 or not 0 < len(embedding) <= 64:
                        raise ValueError(f'Instruction must produce 1..64 tokens of width 4096: {prompt}')
                    values = embedding.float().cpu().numpy().astype(np.float16)
                    if not np.isfinite(values).all():
                        raise ValueError(f'Instruction produced non-finite float16 embeddings: {prompt}')
                    table[index] = 0
                    table[index, :len(embedding)] = values
            table.flush()
            table = None
            # Flush the mapped file to durable storage before publishing.
            with (staging / 'text_embeddings.npy').open('rb') as handle:
                os.fsync(handle.fileno())
            write_durable_json(staging / 'tasks.json', texts)
            write_durable_json(staging / 'dataset.json', dict(benchmark=args.bench, dt=spec['dt'],
                camera_count=len(spec['cameras']), episodes=records, prepare_identity_sha256=root_digest,
                summary_files={name: file_identity(staging / name) for name in
                               ('codec.json', 'tasks.json', 'text_embeddings.npy')}))
            verify_assets()
            publish_summary(output, staging, root_digest)
        except PreparationIdentityChanged:
            cancel_new_shards(output, new_shards)
            raise
        except BaseException:
            # Ordinary interruption preserves progress, provided the input
            # identity still matches. Drift cancels only this call's markers.
            if new_shards:
                try:
                    verify_assets()
                except PreparationIdentityChanged:
                    cancel_new_shards(output, new_shards)
            raise
        finally:
            vae = geometry = encoder = table = None
            if staging is not None and staging.exists():
                shutil.rmtree(staging)
