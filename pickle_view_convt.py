

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
        'timestamp': data['timestamp'],
        'frame_index': data['frame_index'],
        'episode_index': data['episode_index'],
        'index': data['index'],
        'task_index': data['task_index'],
        'action': stack_if_object(data['action']),
        'observation.state': stack_if_object(data['observation.state']),
        'observation.images.wrist': stack_if_object(data['observation.images.wrist']),
        'observation.images.exo': stack_if_object(data['observation.images.exo']),
        'observation.images.table': stack_if_object(data['observation.images.table']),
    }
    for k, v in clean_data.items():
        print(f"{k}: type={type(v)}, dtype={getattr(v, 'dtype', None)}")
    exit()

    outdir = f"/sdb1/piper_subtask_data/train/pick/Pick the blue plastic cup in the center./downnp/{i}"
    os.makedirs(outdir, exist_ok=True)
    np.savez(f"{outdir}/episode.npz", **clean_data)