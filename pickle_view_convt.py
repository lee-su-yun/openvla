

# for i in range(1, 21):
#     with open(f"/sdb1/piper_subtask_data/train/pick/Pick the blue plastic cup in the center./{i}/episode.pickle", "rb") as f:
#         data = pickle.load(f)
#
#     # 전체 구조 보기 (요약)
#     # for key in data:
#     #     print(f"{key}: type={type(data[key])}, shape={getattr(data[key], 'shape', None)}")
#
#
#     outdir = f"/sdb1/piper_subtask_data/train/pick/Pick the blue plastic cup in the center./downnp/{i}"
#     os.makedirs(outdir, exist_ok=True)
#     np.savez(f"{outdir}/episode.npz", **data)

import pickle
import numpy as np
import pandas as pd
import os

import pandas as pd
import numpy as np

def to_numpy_safe(x):
    # pandas Series인 경우 → .to_numpy() 먼저
    if isinstance(x, pd.Series):
        x = x.to_numpy()

    # object dtype numpy array인 경우 → stack
    if isinstance(x, np.ndarray) and x.dtype == object:
        try:
            return np.stack(x)
        except Exception:
            # fallback: 각 원소를 np.array로 강제
            return np.array([np.array(e) for e in x])
    return x

def stack_if_object(arr):
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            return np.stack(arr)
        except Exception:
            return np.array([np.array(x) for x in arr])
    return arr

def to_numpy(x):
    return x.to_numpy() if isinstance(x, pd.Series) else np.array(x)

for i in range(1, 21):
    with open(f"/sdb1/piper_subtask_data/train/pick/Pick the blue plastic cup in the center./{i}/episode.pickle", "rb") as f:
        data = pickle.load(f)

    clean_data = {
        'timestamp': to_numpy_safe(data['timestamp']),
        'frame_index': to_numpy_safe(data['frame_index']),
        'episode_index': to_numpy_safe(data['episode_index']),
        'index': to_numpy_safe(data['index']),
        'task_index': to_numpy_safe(data['task_index']),
        'action': to_numpy_safe(data['action']),
        'observation.state': to_numpy_safe(data['observation.state']),
        'observation.images.wrist': to_numpy_safe(data['observation.images.wrist']),
        'observation.images.exo': to_numpy_safe(data['observation.images.exo']),
        'observation.images.table': to_numpy_safe(data['observation.images.table']),
    }

    for k, v in clean_data.items():
        print(f"{k}: type={type(v)}, dtype={getattr(v, 'dtype', None)}")
    exit()

    outdir = f"/sdb1/piper_subtask_data/train/pick/Pick the blue plastic cup in the center./downnp/{i}"
    os.makedirs(outdir, exist_ok=True)
    np.savez(f"{outdir}/episode.npz", **clean_data)