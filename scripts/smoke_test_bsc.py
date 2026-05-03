"""BSC environment smoke-test for the ConvLSTM precipitation pipeline.

Trains a minimal ConvLSTM (or Bidirectional) model on synthetic data so the
researcher can validate the BSC installation - GPU detection, MirroredStrategy,
file-system writes, library versions - BEFORE running the full spec 002
pipeline. Zero external data dependencies.

Quick examples:

    # env probe only, no training
    python scripts/smoke_test_bsc.py --check-only

    # single-GPU tiny training
    python scripts/smoke_test_bsc.py --gpus 1 --epochs 2 --model ConvLSTM

    # multi-GPU mirror with Bidirectional
    python scripts/smoke_test_bsc.py --gpus 4 --model ConvLSTM_Bidirectional \\
        --epochs 3 --batch-size 4 --grid 31 31 --n-samples 40

Exit code 0 on success; non-zero on any failure with named stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _env_report() -> dict:
    """Collect environment info without importing tensorflow first."""
    import platform
    info = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'hostname': platform.node(),
        'cwd': str(Path.cwd()),
        'env_vars': {
            k: os.environ.get(k, '<unset>') for k in (
                'CUDA_VISIBLE_DEVICES', 'TF_CPP_MIN_LOG_LEVEL',
                'TF_ENABLE_ONEDNN_OPTS', 'SLURM_JOB_ID', 'SLURM_NTASKS',
                'SLURM_GPUS_ON_NODE', 'XLA_FLAGS',
            )
        },
    }
    return info


def _tf_report() -> dict:
    """Collect TensorFlow / GPU info."""
    import tensorflow as tf
    try:
        cuda_info = tf.sysconfig.get_build_info()
    except Exception:
        cuda_info = {}
    physical = tf.config.list_physical_devices('GPU')
    logical = tf.config.list_logical_devices('GPU')
    gpu_details = []
    for dev in physical:
        try:
            d = tf.config.experimental.get_device_details(dev)
            gpu_details.append({'name': dev.name, **{k: str(v) for k, v in d.items()}})
        except Exception as e:
            gpu_details.append({'name': dev.name, 'error': str(e)})
    return {
        'tensorflow': tf.__version__,
        'keras': getattr(tf.keras, '__version__', 'bundled'),
        'cuda_version': cuda_info.get('cuda_version', 'n/a'),
        'cudnn_version': cuda_info.get('cudnn_version', 'n/a'),
        'n_physical_gpus': len(physical),
        'n_logical_gpus': len(logical),
        'gpu_details': gpu_details,
    }


def _build_convlstm(n_feats: int, lat: int, lon: int, horizon: int):
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow as tf
    inp = keras.Input(shape=(None, lat, lon, n_feats), name='input_layer')
    x = layers.ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True,
                           activation='tanh', name='conv_lstm2d')(inp)
    x = layers.ConvLSTM2D(8, (3, 3), padding='same', return_sequences=False,
                           activation='tanh', name='conv_lstm2d_1')(x)
    x = layers.Conv2D(horizon, (1, 1), padding='same', activation='linear', name='head_conv1x1')(x)
    x = layers.Lambda(lambda t: tf.transpose(t, [0, 3, 1, 2]),
                      output_shape=(horizon, lat, lon), name='head_transpose')(x)
    x = layers.Lambda(lambda t: tf.expand_dims(t, -1),
                      output_shape=(horizon, lat, lon, 1), name='head_expand_dim')(x)
    return keras.Model(inp, x, name='smoke_convlstm')


def _build_bidirectional(n_feats: int, lat: int, lon: int, horizon: int):
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow as tf
    inp = keras.Input(shape=(None, lat, lon, n_feats), name='input_layer')
    fwd = layers.ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True,
                             activation='tanh', name='conv_lstm2d')(inp)
    bwd = layers.ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True,
                             activation='tanh', go_backwards=True, name='conv_lstm2d_1')(inp)
    x = layers.Concatenate(name='concatenate')([fwd, bwd])
    x = layers.ConvLSTM2D(8, (3, 3), padding='same', return_sequences=False,
                           activation='tanh', name='conv_lstm2d_2')(x)
    x = layers.Conv2D(horizon, (1, 1), padding='same', activation='linear', name='head_conv1x1')(x)
    x = layers.Lambda(lambda t: tf.transpose(t, [0, 3, 1, 2]),
                      output_shape=(horizon, lat, lon), name='head_transpose')(x)
    x = layers.Lambda(lambda t: tf.expand_dims(t, -1),
                      output_shape=(horizon, lat, lon, 1), name='head_expand_dim')(x)
    return keras.Model(inp, x, name='smoke_bidirectional')


MODEL_FACTORIES = {
    'ConvLSTM': _build_convlstm,
    'ConvLSTM_Bidirectional': _build_bidirectional,
}


def _build_synthetic(n_samples: int, input_window: int, lat: int, lon: int,
                     n_feats: int, horizon: int, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, input_window, lat, lon, n_feats)).astype('float32')
    y = rng.standard_normal((n_samples, horizon, lat, lon, 1)).astype('float32')
    return X, y


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--gpus', type=int, default=-1,
                   help='Number of GPUs to use: -1 auto-detect (default), 0 CPU-only, '
                        '1 single-GPU, >=2 MirroredStrategy.')
    p.add_argument('--model', choices=tuple(MODEL_FACTORIES), default='ConvLSTM',
                   help='Architecture to probe (default: ConvLSTM; 79K params).')
    p.add_argument('--epochs', type=int, default=2, help='Training epochs (default 2).')
    p.add_argument('--batch-size', type=int, default=2,
                   help='Global batch size (will be split across replicas).')
    p.add_argument('--grid', type=int, nargs=2, metavar=('LAT', 'LON'), default=[21, 21],
                   help='Grid dimensions for synthetic data (default 21x21).')
    p.add_argument('--input-window', type=int, default=30,
                   help='Timesteps per sample (default 30; real pipeline uses 60).')
    p.add_argument('--horizon', type=int, default=12,
                   help='Forecast horizon (default 12).')
    p.add_argument('--n-features', type=int, default=12,
                   help='Feature channels (default 12, matches BASIC bundle).')
    p.add_argument('--n-samples', type=int, default=20,
                   help='Synthetic sample count (default 20 - split 80/20 train/val).')
    p.add_argument('--seed', type=int, default=42, help='Random seed.')
    p.add_argument('--out-dir', type=Path, default=REPO / 'scripts' / 'smoke_test_output',
                   help='Where to write env report + model weights.')
    p.add_argument('--check-only', action='store_true',
                   help='Only print env info and exit (no training).')
    p.add_argument('--mixed-precision', action='store_true',
                   help='Enable mixed-precision training (float16 compute, float32 variables).')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    report: dict = {
        'timestamp': datetime.now().isoformat(),
        'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'stage': 'env_probe',
        'env': _env_report(),
    }

    print('=' * 72)
    print('BSC SMOKE TEST')
    print('=' * 72)
    print(f'python        {report["env"]["python"]}')
    print(f'platform      {report["env"]["platform"]}')
    print(f'hostname      {report["env"]["hostname"]}')
    for k, v in report['env']['env_vars'].items():
        print(f'{k:<24s} {v}')

    # --- TensorFlow probe ---
    try:
        report['tf'] = _tf_report()
    except Exception as e:
        report['stage_error'] = f'tf_import_failed: {e}'
        print(f'\n[FAIL] TensorFlow import/probe failed: {e}')
        _write_report(args.out_dir, report)
        return 10

    print('')
    print(f'tensorflow    {report["tf"]["tensorflow"]}')
    print(f'keras         {report["tf"]["keras"]}')
    print(f'cuda          {report["tf"]["cuda_version"]}')
    print(f'cudnn         {report["tf"]["cudnn_version"]}')
    print(f'physical_gpus {report["tf"]["n_physical_gpus"]}')
    for d in report['tf']['gpu_details']:
        print(f'   {d.get("name")}  {d.get("device_name","")}  '
              f'compute={d.get("compute_capability","?")}')

    if args.check_only:
        elapsed = time.time() - t0
        report['stage'] = 'check_only_complete'
        report['elapsed_s'] = elapsed
        _write_report(args.out_dir, report)
        print(f'\n[OK] check-only complete in {elapsed:.2f}s  report: {_report_path(args.out_dir)}')
        return 0

    # --- Device / distribution config ---
    import tensorflow as tf
    n_phys = report['tf']['n_physical_gpus']
    if args.gpus == -1:
        n_use = n_phys
    elif args.gpus == 0:
        n_use = 0
    else:
        n_use = min(args.gpus, n_phys)

    if args.gpus > n_phys:
        print(f'\n[WARN] requested {args.gpus} GPUs, only {n_phys} available - using {n_phys}')

    if n_use == 0:
        print('\n[cfg] CPU-only mode')
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception as e:
            print(f'   could not disable GPUs: {e}')
        strategy = tf.distribute.get_strategy()  # default
    elif n_use == 1:
        print(f'\n[cfg] single-GPU mode (1 of {n_phys})')
        for g in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
    else:
        print(f'\n[cfg] MirroredStrategy across {n_use} GPU(s)')
        for g in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        devices = [f'/GPU:{i}' for i in range(n_use)]
        strategy = tf.distribute.MirroredStrategy(devices=devices)

    report['distribution'] = {
        'strategy': strategy.__class__.__name__,
        'num_replicas_in_sync': strategy.num_replicas_in_sync,
        'n_use_gpus': n_use,
    }
    print(f'   strategy={strategy.__class__.__name__}  '
          f'num_replicas={strategy.num_replicas_in_sync}')

    # --- Mixed precision ---
    if args.mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print('[cfg] mixed_precision=mixed_float16')
            report['mixed_precision'] = 'mixed_float16'
        except Exception as e:
            print(f'[WARN] mixed precision not supported: {e}')
            report['mixed_precision'] = f'error: {e}'

    # --- Seed ---
    tf.keras.utils.set_random_seed(args.seed)

    # --- Build synthetic data ---
    lat, lon = args.grid
    report['stage'] = 'data_build'
    print(f'\n[data] synthetic  samples={args.n_samples}  '
          f'window={args.input_window}  grid={lat}x{lon}  '
          f'features={args.n_features}  horizon={args.horizon}')
    X, y = _build_synthetic(args.n_samples, args.input_window, lat, lon,
                            args.n_features, args.horizon, args.seed)
    cut = int(args.n_samples * 0.8)
    X_tr, X_va = X[:cut], X[cut:]
    y_tr, y_va = y[:cut], y[cut:]
    print(f'   X_tr={X_tr.shape}  y_tr={y_tr.shape}  X_va={X_va.shape}  y_va={y_va.shape}')

    # --- Build model inside strategy scope ---
    report['stage'] = 'model_build'
    factory = MODEL_FACTORIES[args.model]
    with strategy.scope():
        model = factory(args.n_features, lat, lon, args.horizon)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss='mse', metrics=['mae'])
    n_params = int(model.count_params())
    print(f'\n[model] {args.model}  params={n_params:,}')
    report['model'] = {'name': args.model, 'params': n_params}

    # --- Train ---
    report['stage'] = 'training'
    print(f'\n[train] epochs={args.epochs}  global_batch={args.batch_size}  '
          f'per_replica={args.batch_size // max(1, strategy.num_replicas_in_sync)}')
    t_train_start = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                     epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    t_train = time.time() - t_train_start
    report['train'] = {
        'epochs_run': len(hist.history.get('loss', [])),
        'final_train_loss': float(hist.history['loss'][-1]) if hist.history.get('loss') else None,
        'final_val_loss': float(hist.history['val_loss'][-1]) if hist.history.get('val_loss') else None,
        'wall_clock_s': t_train,
        'wall_clock_per_epoch_s': t_train / max(1, args.epochs),
    }
    print(f'   final loss={report["train"]["final_train_loss"]:.4f}  '
          f'val_loss={report["train"]["final_val_loss"]:.4f}  '
          f'time={t_train:.2f}s  ({report["train"]["wall_clock_per_epoch_s"]:.2f}s/ep)')

    # --- Save artefact ---
    report['stage'] = 'save'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.out_dir / f'smoke_{args.model}_seed{args.seed}.weights.h5'
    try:
        model.save_weights(str(weights_path))
        report['weights_path'] = str(weights_path)
        print(f'\n[save] {weights_path.name} ({weights_path.stat().st_size} bytes)')
    except Exception as e:
        print(f'[WARN] weight save failed: {e}')
        report['save_error'] = str(e)

    # --- Final report ---
    report['stage'] = 'complete'
    report['elapsed_s'] = time.time() - t0
    _write_report(args.out_dir, report)
    print(f'\n[OK] total {report["elapsed_s"]:.2f}s  report: {_report_path(args.out_dir)}')
    return 0


def _report_path(out_dir: Path) -> Path:
    return out_dir / 'smoke_report.json'


def _write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _report_path(out_dir).write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
