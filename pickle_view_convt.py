import pickle
import numpy as np
import os

for i in range(1, 21):
    with open(f"/sdb1/piper_subtask_data/train/pick/Pick the blue plastic cup in the center./{i}/episode.pickle", "rb") as f:
        data = pickle.load(f)

    # 전체 구조 보기 (요약)
    # for key in data:
    #     print(f"{key}: type={type(data[key])}, shape={getattr(data[key], 'shape', None)}")


    outdir = f"/sdb1/piper_subtask_data/train/pick/Pick the blue plastic cup in the center./downnp/{i}"
    os.makedirs(outdir, exist_ok=True)
    np.savez(f"{outdir}/episode.npz", **data)
# print(data['episode_index'][:10])
# print(data['index'][:10])
# print(data['task_index'][:10])

# 예시로 일부 확인
#print("예시 데이터:")
#print("첫 번째 이미지 shape:", data['observation.images.table'][0].shape)
#print("첫 번째 action:", data['action'][0])