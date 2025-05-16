

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

def to_numpy(x):
    return x.to_numpy() if isinstance(x, pd.Series) else np.array(x)

for i in range(1, 21):
    with open(f"/sdb1/piper_subtask_data/train/pick/Pick the blue plastic cup in the center./{i}/episode.pickle", "rb") as f:
        data = pickle.load(f)

    clean_data = {
        'timestamp': to_numpy(data['timestamp']),
        'frame_index': to_numpy(data['frame_index']),
        'episode_index': to_numpy(data['episode_index']),
        'index': to_numpy(data['index']),
        'task_index': to_numpy(data['task_index']),
        'action': to_numpy(data['action']),
        'observation.state': to_numpy(data['observation.state']),
        'observation.images.wrist': to_numpy(data['observation.images.wrist']),
        'observation.images.exo': to_numpy(data['observation.images.exo']),
        'observation.images.table': to_numpy(data['observation.images.table']),
    }
    for k, v in clean_data.items():
        print(f"{k}: type={type(v)}, dtype={getattr(v, 'dtype', None)}")
    exit()

    outdir = f"/sdb1/piper_subtask_data/train/pick/Pick the blue plastic cup in the center./downnp/{i}"
    os.makedirs(outdir, exist_ok=True)
    np.savez(f"{outdir}/episode.npz", **clean_data)