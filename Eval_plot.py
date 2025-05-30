import pickle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":


   #  with open("/sdb1/piper_subtask_data/eval/pick/Val_np/Pick the blue cup on the right./episode.pickle/episode.pickle", "rb") as f:
   #      data1 = pickle.load(f)
   #  with open("/sdb1/piper_5hz/validation/Align the cups/111/episode.pickle", "rb") as f:
   #      data = pickle.load(f)
   #  print(f"data type: {type(data)}\n")
   #
   #  # dict일 경우 → 키 확인
   #  if isinstance(data, dict):
   #      print(f"key: {list(data.keys())}\n")
   #      for key in data:
   #          print(
   #              f"  └ {key}: type = {type(data[key])}, len = {len(data[key]) if hasattr(data[key], '__len__') else 'N/A'}")
   #  arr = np.array(data1['action'])
   #  print(arr.shape)
   # #print(data1['action'].shape)
   #  print(data['action'][0][:][:])
   #  exit()
        #
    import os
    import pickle
    import pandas as pd

   # 경로 설정
   #  source_base = "/sdb1/piper_subtask_data/train/pick/Pick the blue cup in the center."
   #  target_base = "/sdb1/piper_subtask_data/train/pick/train_np/Pick the blue cup in the center."
   #
   # # 폴더 없으면 생성
   #  os.makedirs(target_base, exist_ok=True)
   #
   # # 1~10 에피소드 반복
   #  for i in range(1, 21):
   #     source_path = os.path.join(source_base, str(i), "episode.pickle")
   #     target_path = os.path.join(target_base, f"episode{i}.pickle")
   #
   #     # DataFrame 로드
   #     with open(source_path, "rb") as f:
   #         df = pickle.load(f)
   #
   #     # dict로 변환
   #     data_dict = df.to_dict(orient="list")
   #
   #     # 저장
   #     with open(target_path, "wb") as f:
   #         pickle.dump(data_dict, f, protocol=4)
   #
   #     print(f"Saved: {target_path}")
   #  exit()
   #  with open("/sdb1/piper_subtask_data/train/pick/train_np/Pick the blue cup in the center./episode1.pickle", "rb") as f:
    #with open("/sdb1/piper_subtask_data/train/pick/Pick the blue cup in the center./1/episode.pickle", "rb") as f:
    with open("/data/piper_subtask_data/Val/Pick the blue cup on the right./episode.pickle", "rb") as f:
        data = pickle.load(f)
    image = Image.fromarray(data['observation.images.exo'][0][0])
    state = data['observation.images.exo'][0][0]
    action = data['action'][0][0]
    print(f'state : {state}\n')
    print(f'action : {action}\n')
    plt.imshow(image)
    plt.show()

    # Load Processor & VLA
    #model_path = "/sdc1/piper_subtask/openvla/Norm/openvla-7b+piper5_hz_subtask+b16+lr-0.0005+val+lora-r32+dropout-0.0--image_aug+norm/step_10000"
    #model_path = "/sdc1/piper_subtask/openvla/Top_Norm_lora/openvla-7b+piper5_hz_subtask+b16+lr-0.0005+top+lora-r32+dropout-0.0--image_aug+norm/step_10000"
    model_path = "/ckpt/piper_subtask/openvla/Fully/openvla-7b+piper5_hz_subtask+b16+lr-0.0005+val+qlora-r32+dropout-0.0--image_aug+norm/step_2000"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        #"/sdb1/ckpt/openvla_5hz_n/openvla-7b+piper5_hz+b16+lr-0.0005+lora-r32+dropout-0.0/latest",
        #"/ckpt/openvla-7b",
        model_path,
        #attn_implementation="flash_attention_2", # [Optional] Requires `flash_attn`
        torch_dtype=dtype,
        #low_cpu_mem_usage=True,
        low_cpu_mem_usage=True,
     #   quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        trust_remote_code=True
    ).to(device)

    ###########
   # Pick

    ###########

    #with open("/sdb1/piper_subtask_data/eval/pick/Validation/Pick the blue cup on the right./episode.pickle", "rb") as f:
    #Pick the blue cup on the right.
    #Pick the white cup nearest from the robot.
    #Pick the red cup behind the purple one.
    #Pick the yellow cup.
    #with open("/sdb1/piper_subtask_data/eval/pick/Val_np/Pick the yellow cup./episode.pickle", "rb") as f:



    traj_111_latest = []
    #for i in range(50):
    #for i in range(0, 300, 6):
    for i in range(30, 294, 6):
        image = Image.fromarray(data['observation.images.exo'][i][0])

        #
        # np_img = np.array(image)
        # np_img_bgr = np_img[..., ::-1]
        # image_bgr = Image.fromarray(np_img_bgr)


        prompt = "In: What should the robot do to pick the blue cup in the center?\nOut:"
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="piper5_hz_subtask", do_sample=False)
        traj_111_latest.append(action)
    predictions_111_latest = []
    x = []
    y = []
    z = []
    rx = []
    ry = []
    rz = []
    g = []
    for i in range(44):
        x.append(traj_111_latest[i][0])
        y.append(traj_111_latest[i][1])
        z.append(traj_111_latest[i][2])
        rx.append(traj_111_latest[i][3])
        ry.append(traj_111_latest[i][4])
        rz.append(traj_111_latest[i][5])
        g.append(traj_111_latest[i][6])
    predictions_111_latest.append(x)
    predictions_111_latest.append(y)
    predictions_111_latest.append(z)
    predictions_111_latest.append(rx)
    predictions_111_latest.append(ry)
    predictions_111_latest.append(rz)
    predictions_111_latest.append(g)






    import plotly.graph_objects as go
    import numpy as np
    timesteps = np.arange(44)
    import matplotlib.pyplot as plt
    gt_111 = []
    x = []
    y = []
    z = []
    rx = []
    ry = []
    rz = []
    g = []
    for i in range(30, 294, 6):
        # x.append(data['action'][i][0][0])
        # y.append(data['action'][i][0][1])
        # z.append(data['action'][i][0][2])
        # rx.append(data['action'][i][0][3])
        # ry.append(data['action'][i][0][4])
        # rz.append(data['action'][i][0][5])
        # g.append(data['action'][i][0][6])
        x.append(data['observation.state'][6*(i+1)][0][0]-data['observation.state'][i][0][0])
        y.append(data['observation.state'][6*(i+1)][0][1]-data['observation.state'][i][0][1])
        z.append(data['observation.state'][6*(i+1)][0][2]-data['observation.state'][i][0][2])
        rx.append(data['observation.state'][6*(i+1)][0][3]-data['observation.state'][i][0][3])
        ry.append(data['observation.state'][6*(i+1)][0][4]-data['observation.state'][i][0][4])
        rz.append(data['observation.state'][6*(i+1)][0][5]-data['observation.state'][i][0][5])
        g.append(data['observation.state'][6*(i+1)][0][6]-data['observation.state'][i][0][6])
    gt_111.append(x)
    gt_111.append(y)
    gt_111.append(z)
    gt_111.append(rx)
    gt_111.append(ry)
    gt_111.append(rz)
    gt_111.append(g)
# 7개의 리스트를 50개의 (7,) 벡터로 전치
    trajectory_array = np.stack(gt_111, axis=1)  # shape: (50, 7)
    pre_trajectory_array = np.stack(predictions_111_latest, axis=1)

    # 저장
    # np.save("gt_train.npy", trajectory_array)
    # np.save("predict_train.npy", pre_trajectory_array)
    fig_111_latest = go.Figure(data=[go.Scatter3d(
        x=predictions_111_latest[0], y=predictions_111_latest[1], z=predictions_111_latest[2],
        mode='lines+markers',
        marker=dict(
            size=10,
            color=timesteps,
            symbol="circle",
            colorscale='Viridis',  # 기본 색상 스케일 (Viridis)
            colorbar=dict(title='Timestep')  # 색상 바 제목
        ),
        line=dict(
            color=timesteps,  # 선 색상도 timestep에 따라 변함
            colorscale='Viridis',  # 동일한 색상 스케일
            width=4
        ),
    ), go.Scatter3d(
        x=gt_111[0], y=gt_111[1], z=gt_111[2],
        mode='lines+markers',
        marker=dict(
            size=10,
            color=timesteps,  # 색상 배열을 적용
            symbol="square",
            colorscale='Viridis',  # 기본 색상 스케일 (Viridis)
            colorbar=dict(title='Timestep')  # 색상 바 제목
        ),
        line=dict(
            color=timesteps,  # 선 색상도 timestep에 따라 변함
            colorscale='Viridis',  # 동일한 색상 스케일
            width=4
        ),
    )])
    fig_111_latest.show()
    fig_111_latest.write_html("delta.html")
    #Pick_the_blue_cup_on_the_right
    #Pick_the_white_cup_nearest_from_the_robot
    #Pick_the_red_cup_behind_the_purple_one
    #Pick_the_yellow_cup
    for i in range(7):
        n = f'71{i + 1}'
        plt.subplot(int(n))
        plt.plot(predictions_111_latest[i], 'b--', gt_111[i], 'r')
    plt.savefig("delta.png")
    plt.show()

# #######################################################################################
# #######################################################################################
#
#     with open("/sdb1/piper_subtask_data/eval/pick/Val_np/Pick the white cup nearest from the robot./episode.pickle",
#               "rb") as f:
#         data = pickle.load(f)
#
#     traj_111_latest = []
#     # for i in range(50):
#     for i in range(0, 300, 6):
#         image = Image.fromarray(data['observation.images.table'][i][0])
#         prompt = "In: What should the robot do to pick the white cup nearest from the robot?\nOut:"
#         inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
#         action = vla.predict_action(**inputs, unnorm_key="piper5_hz", do_sample=False)
#         traj_111_latest.append(action)
#     predictions_111_latest = []
#     x = []
#     y = []
#     z = []
#     rx = []
#     ry = []
#     rz = []
#     g = []
#     for i in range(50):
#         x.append(traj_111_latest[i][0])
#         y.append(traj_111_latest[i][1])
#         z.append(traj_111_latest[i][2])
#         rx.append(traj_111_latest[i][3])
#         ry.append(traj_111_latest[i][4])
#         rz.append(traj_111_latest[i][5])
#         g.append(traj_111_latest[i][6])
#     predictions_111_latest.append(x)
#     predictions_111_latest.append(y)
#     predictions_111_latest.append(z)
#     predictions_111_latest.append(rx)
#     predictions_111_latest.append(ry)
#     predictions_111_latest.append(rz)
#     predictions_111_latest.append(g)
#
#     import plotly.graph_objects as go
#     import numpy as np
#
#     timesteps = np.arange(50)
#     import matplotlib.pyplot as plt
#
#     gt_111 = []
#     x = []
#     y = []
#     z = []
#     rx = []
#     ry = []
#     rz = []
#     g = []
#     for i in range(0, 300, 6):
#         x.append(data['action'][i][0][0])
#         y.append(data['action'][i][0][1])
#         z.append(data['action'][i][0][2])
#         rx.append(data['action'][i][0][3])
#         ry.append(data['action'][i][0][4])
#         rz.append(data['action'][i][0][5])
#         g.append(data['action'][i][0][6])
#     gt_111.append(x)
#     gt_111.append(y)
#     gt_111.append(z)
#     gt_111.append(rx)
#     gt_111.append(ry)
#     gt_111.append(rz)
#     gt_111.append(g)
#     fig_111_latest = go.Figure(data=[go.Scatter3d(
#         x=predictions_111_latest[0], y=predictions_111_latest[1], z=predictions_111_latest[2],
#         mode='lines+markers',
#         marker=dict(
#             size=10,
#             color=timesteps,
#             symbol="circle",
#             colorscale='Viridis',  # 기본 색상 스케일 (Viridis)
#             colorbar=dict(title='Timestep')  # 색상 바 제목
#         ),
#         line=dict(
#             color=timesteps,  # 선 색상도 timestep에 따라 변함
#             colorscale='Viridis',  # 동일한 색상 스케일
#             width=4
#         ),
#     ), go.Scatter3d(
#         x=gt_111[0], y=gt_111[1], z=gt_111[2],
#         mode='lines+markers',
#         marker=dict(
#             size=10,
#             color=timesteps,  # 색상 배열을 적용
#             symbol="square",
#             colorscale='Viridis',  # 기본 색상 스케일 (Viridis)
#             colorbar=dict(title='Timestep')  # 색상 바 제목
#         ),
#         line=dict(
#             color=timesteps,  # 선 색상도 timestep에 따라 변함
#             colorscale='Viridis',  # 동일한 색상 스케일
#             width=4
#         ),
#     )])
#     fig_111_latest.show()
#     fig_111_latest.write_html("Pick_the_white_cup_nearest_from_the_robot.html")
#     # Pick_the_blue_cup_on_the_right
#     # Pick_the_white_cup_nearest_from_the_robot
#     # Pick_the_red_cup_behind_the_purple_one
#     # Pick_the_yellow_cup
#     for i in range(7):
#         n = f'71{i + 1}'
#         plt.subplot(int(n))
#         plt.plot(predictions_111_latest[i], 'b--', gt_111[i], 'r')
#     plt.savefig("Pick_the_white_cup_nearest_from_the_robot.png")
#     plt.show()
#
#     with open("/sdb1/piper_subtask_data/eval/pick/Val_np/Pick the red cup behind the purple one./episode.pickle",
#               "rb") as f:
#         data = pickle.load(f)
#
#     traj_111_latest = []
#     # for i in range(50):
#     for i in range(0, 300, 6):
#         image = Image.fromarray(data['observation.images.table'][i][0])
#         prompt = "In: What should the robot do to pick the red cup behind the purple one?\nOut:"
#         inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
#         action = vla.predict_action(**inputs, unnorm_key="piper5_hz", do_sample=False)
#         traj_111_latest.append(action)
#     predictions_111_latest = []
#     x = []
#     y = []
#     z = []
#     rx = []
#     ry = []
#     rz = []
#     g = []
#     for i in range(50):
#         x.append(traj_111_latest[i][0])
#         y.append(traj_111_latest[i][1])
#         z.append(traj_111_latest[i][2])
#         rx.append(traj_111_latest[i][3])
#         ry.append(traj_111_latest[i][4])
#         rz.append(traj_111_latest[i][5])
#         g.append(traj_111_latest[i][6])
#     predictions_111_latest.append(x)
#     predictions_111_latest.append(y)
#     predictions_111_latest.append(z)
#     predictions_111_latest.append(rx)
#     predictions_111_latest.append(ry)
#     predictions_111_latest.append(rz)
#     predictions_111_latest.append(g)
#
#     import plotly.graph_objects as go
#     import numpy as np
#
#     timesteps = np.arange(50)
#     import matplotlib.pyplot as plt
#
#     gt_111 = []
#     x = []
#     y = []
#     z = []
#     rx = []
#     ry = []
#     rz = []
#     g = []
#     for i in range(0, 300, 6):
#         x.append(data['action'][i][0][0])
#         y.append(data['action'][i][0][1])
#         z.append(data['action'][i][0][2])
#         rx.append(data['action'][i][0][3])
#         ry.append(data['action'][i][0][4])
#         rz.append(data['action'][i][0][5])
#         g.append(data['action'][i][0][6])
#     gt_111.append(x)
#     gt_111.append(y)
#     gt_111.append(z)
#     gt_111.append(rx)
#     gt_111.append(ry)
#     gt_111.append(rz)
#     gt_111.append(g)
#     fig_111_latest = go.Figure(data=[go.Scatter3d(
#         x=predictions_111_latest[0], y=predictions_111_latest[1], z=predictions_111_latest[2],
#         mode='lines+markers',
#         marker=dict(
#             size=10,
#             color=timesteps,
#             symbol="circle",
#             colorscale='Viridis',  # 기본 색상 스케일 (Viridis)
#             colorbar=dict(title='Timestep')  # 색상 바 제목
#         ),
#         line=dict(
#             color=timesteps,  # 선 색상도 timestep에 따라 변함
#             colorscale='Viridis',  # 동일한 색상 스케일
#             width=4
#         ),
#     ), go.Scatter3d(
#         x=gt_111[0], y=gt_111[1], z=gt_111[2],
#         mode='lines+markers',
#         marker=dict(
#             size=10,
#             color=timesteps,  # 색상 배열을 적용
#             symbol="square",
#             colorscale='Viridis',  # 기본 색상 스케일 (Viridis)
#             colorbar=dict(title='Timestep')  # 색상 바 제목
#         ),
#         line=dict(
#             color=timesteps,  # 선 색상도 timestep에 따라 변함
#             colorscale='Viridis',  # 동일한 색상 스케일
#             width=4
#         ),
#     )])
#     fig_111_latest.show()
#     fig_111_latest.write_html("Pick_the_red_cup_behind_the_purple_one.html")
#     # Pick_the_blue_cup_on_the_right
#     # Pick_the_white_cup_nearest_from_the_robot
#     # Pick_the_red_cup_behind_the_purple_one
#     # Pick_the_yellow_cup
#     for i in range(7):
#         n = f'71{i + 1}'
#         plt.subplot(int(n))
#         plt.plot(predictions_111_latest[i], 'b--', gt_111[i], 'r')
#     plt.savefig("Pick_the_red_cup_behind_the_purple_one.png")
#     plt.show()
#
#     with open("/sdb1/piper_subtask_data/eval/pick/Val_np/Pick the yellow cup./episode.pickle",
#               "rb") as f:
#         data = pickle.load(f)
#
#     traj_111_latest = []
#     # for i in range(50):
#     for i in range(0, 300, 6):
#         image = Image.fromarray(data['observation.images.table'][i][0])
#         prompt = "In: What should the robot do to pick the yellow cup?\nOut:"
#         inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
#         action = vla.predict_action(**inputs, unnorm_key="piper5_hz", do_sample=False)
#         traj_111_latest.append(action)
#     predictions_111_latest = []
#     x = []
#     y = []
#     z = []
#     rx = []
#     ry = []
#     rz = []
#     g = []
#     for i in range(50):
#         x.append(traj_111_latest[i][0])
#         y.append(traj_111_latest[i][1])
#         z.append(traj_111_latest[i][2])
#         rx.append(traj_111_latest[i][3])
#         ry.append(traj_111_latest[i][4])
#         rz.append(traj_111_latest[i][5])
#         g.append(traj_111_latest[i][6])
#     predictions_111_latest.append(x)
#     predictions_111_latest.append(y)
#     predictions_111_latest.append(z)
#     predictions_111_latest.append(rx)
#     predictions_111_latest.append(ry)
#     predictions_111_latest.append(rz)
#     predictions_111_latest.append(g)
#
#     import plotly.graph_objects as go
#     import numpy as np
#
#     timesteps = np.arange(50)
#     import matplotlib.pyplot as plt
#
#     gt_111 = []
#     x = []
#     y = []
#     z = []
#     rx = []
#     ry = []
#     rz = []
#     g = []
#     for i in range(0, 300, 6):
#         x.append(data['action'][i][0][0])
#         y.append(data['action'][i][0][1])
#         z.append(data['action'][i][0][2])
#         rx.append(data['action'][i][0][3])
#         ry.append(data['action'][i][0][4])
#         rz.append(data['action'][i][0][5])
#         g.append(data['action'][i][0][6])
#     gt_111.append(x)
#     gt_111.append(y)
#     gt_111.append(z)
#     gt_111.append(rx)
#     gt_111.append(ry)
#     gt_111.append(rz)
#     gt_111.append(g)
#     fig_111_latest = go.Figure(data=[go.Scatter3d(
#         x=predictions_111_latest[0], y=predictions_111_latest[1], z=predictions_111_latest[2],
#         mode='lines+markers',
#         marker=dict(
#             size=10,
#             color=timesteps,
#             symbol="circle",
#             colorscale='Viridis',  # 기본 색상 스케일 (Viridis)
#             colorbar=dict(title='Timestep')  # 색상 바 제목
#         ),
#         line=dict(
#             color=timesteps,  # 선 색상도 timestep에 따라 변함
#             colorscale='Viridis',  # 동일한 색상 스케일
#             width=4
#         ),
#     ), go.Scatter3d(
#         x=gt_111[0], y=gt_111[1], z=gt_111[2],
#         mode='lines+markers',
#         marker=dict(
#             size=10,
#             color=timesteps,  # 색상 배열을 적용
#             symbol="square",
#             colorscale='Viridis',  # 기본 색상 스케일 (Viridis)
#             colorbar=dict(title='Timestep')  # 색상 바 제목
#         ),
#         line=dict(
#             color=timesteps,  # 선 색상도 timestep에 따라 변함
#             colorscale='Viridis',  # 동일한 색상 스케일
#             width=4
#         ),
#     )])
#     fig_111_latest.show()
#     fig_111_latest.write_html("Pick_the_yellow_cup.html")
#     # Pick_the_blue_cup_on_the_right
#     # Pick_the_white_cup_nearest_from_the_robot
#     # Pick_the_red_cup_behind_the_purple_one
#     # Pick_the_yellow_cup
#     for i in range(7):
#         n = f'71{i + 1}'
#         plt.subplot(int(n))
#         plt.plot(predictions_111_latest[i], 'b--', gt_111[i], 'r')
#     plt.savefig("Pick_the_yellow_cup.png")
#     plt.show()
#
#
