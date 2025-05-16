import pickle

with open("/sdb1/piper_subtask_data/train/pick/Pick the blue plastic cup in the center./1/episode.pickle", "rb") as f:
    data = pickle.load(f)

# 전체 구조 보기 (요약)
for key in data:
    print(f"{key}: type={type(data[key])}, shape={getattr(data[key], 'shape', None)}")

print(data['episode_index'][:10])
print(data['index'][:10])
print(data['task_index'][:10])

# 예시로 일부 확인
#print("예시 데이터:")
#print("첫 번째 이미지 shape:", data['observation.images.table'][0].shape)
#print("첫 번째 action:", data['action'][0])