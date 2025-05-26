import pickle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
device = "cuda:1" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
import numpy as np


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
   #  source_base = "/sdb1/piper_subtask_data/eval/pick/Validation_add/Pick the yellow cup."
   #  target_base = "/sdb1/piper_subtask_data/eval/pick/Val_add_np/Pick the yellow cup."
   #
   # # 폴더 없으면 생성
   #  os.makedirs(target_base, exist_ok=True)
   #
   # # 1~10 에피소드 반복
   #  for i in range(1, 11):
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

    # Load Processor & VLA
    #model_path = "/sdc1/piper_subtask/openvla/openvla-7b+piper5_hz_subtask+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    model_path = "/sdc1/piper_subtask/openvla/Top_Norm/openvla-7b+piper5_hz_subtask+b16+lr-0.0005+val+qlora-r32+dropout-0.0+q-4bit--image_aug+norm/step_1000"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        #"/sdb1/ckpt/openvla_5hz_n/openvla-7b+piper5_hz+b16+lr-0.0005+lora-r32+dropout-0.0/latest",
        #"/ckpt/openvla-7b",
        model_path,
        #attn_implementation="flash_attention_2", # [Optional] Requires `flash_attn`
        torch_dtype=dtype,
        #low_cpu_mem_usage=True,
        #low_cpu_mem_usage=True,
        #quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        trust_remote_code=True
    ).to(device)


    ###########
   # Pick
    # vla.norm_stats["piper5_hz"] = {
    #     "action": {
    #         "mean": [155008.453125, 10229.4921875, 330142.46875, 31946.19140625, 51902.80859375, -11467.6015625,
    #                      19763.119140625],
    #         "std": [113648.5546875, 50988.80078125, 101760.9921875, 163935.375, 21492.873046875, 160660.078125,
    #                     24381.95703125],
    #         "max": [440180.0, 231048.0, 527437.0, 180000.0, 90000.0, 179999.0, 76076.0],
    #         "min": [-56222.0, -154621.0, 164685.0, -179992.0, 4915.0, -179995.0, -1674.0],
    #         "q01": [-26293.22, -127458.24, 166510.0, -179821.02, 14869.9, -179428.0, -1528.0],
    #         "q99": [420192.36000000004, 150692.44000000006, 495632.79000000004, 179749.08000000002, 85464.17, 179322.16, 73382.73000000001],
    #         "mask": [True, True, True, True, True, True, True]
    #
    #     },
    #     "proprio": {
    #         "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "max": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "q01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "q99": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     },
    #     "num_transitions": 10000,
    #     "num_trajectories": 200
    # }

    vla.norm_stats["piper5_hz"] = {
        "action": {
      "mean": [
        166300.484375,
        11240.7763671875,
        348090.59375,
        31871.673828125,
        49085.80859375,
        -15648.5546875,
        22036.552734375
      ],
      "std": [
        114340.0234375,
        53640.58203125,
        91002.1015625,
        164524.6875,
        20819.51953125,
        160293.09375,
        24605.005859375
      ],
      "max": [
        440180.0,
        231048.0,
        527437.0,
        180000.0,
        90000.0,
        179999.0,
        76076.0
      ],
      "min": [
        -56222.0,
        -154621.0,
        164685.0,
        -179992.0,
        4915.0,
        -179995.0,
        -1674.0
      ],
      "q01": [
        -27396.65,
        -130284.49,
        167111.61,
        -179833.01,
        14653.8,
        -179428.09,
        -1528.0
      ],
      "q99": [
        421863.1500000001,
        157140.5600000001,
        496357.83,
        179778.03,
        85821.07,
        179394.05,
        73964.0
      ],
      "mask": [
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ]
    },
    "proprio": {
      "mean": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "std": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "max": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "min": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "q01": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "q99": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    "num_transitions": 9000,
    "num_trajectories": 200
  }
   # Align the cups
   #  vla.norm_stats["piper5_hz"] = {
   #      "action": {
   #          "mean": [239851.25, -21164.9609375, 261495.8125, 11370.884765625, 37447.53515625, 28121.734375,
   #                   46690.7578125],
   #          "std": [102836.7109375, 62868.7109375, 54746.45703125, 169204.71875, 21283.03515625, 160880.140625,
   #                  24327.744140625],
   #          "max": [461165.0, 202749.0, 456069.0, 180000.0, 90000.0, 179992.0, 76512.0],
   #          "min": [31404.0, -196438.0, 142481.0, -179998.0, -5868.0, -179999.0, -2693.0],
   #          "q01": [48133.68, -159485.18, 164291.0, -179816.02, 3283.49, -179610.02, -1310.0],
   #          "q99": [433687.7, 120789.82000000004, 401451.84, 179817.02, 87156.65000000001, 179512.12, 76003.0],
   #          "mask": [True, True, True, True, True, True, True]
   #      },
   #      "proprio": {
   #          "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   #          "std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   #          "max": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   #          "min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   #          "q01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   #          "q99": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
   #      },
   #      "num_transitions": 13350,
   #      "num_trajectories": 89
   #  }

    ###########




    #with open("/sdb1/piper_subtask_data/eval/pick/Validation/Pick the blue cup on the right./episode.pickle", "rb") as f:
    #Pick the blue cup on the right.
    #Pick the white cup nearest from the robot.
    #Pick the red cup behind the purple one.
    #Pick the yellow cup.
    with open("/sdb1/piper_subtask_data/eval/pick/Val_np/Pick the yellow cup./episode.pickle", "rb") as f:
        data = pickle.load(f)


    traj_111_latest = []
    #for i in range(50):
    for i in range(30, 300, 6):
        image = Image.fromarray(data['observation.images.table'][i][0])
        prompt = "In: What should the robot do to pick the yellow cup?\nOut:"
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="piper5_hz", do_sample=False)
        traj_111_latest.append(action)
    predictions_111_latest = []
    x = []
    y = []
    z = []
    rx = []
    ry = []
    rz = []
    g = []
    for i in range(45):
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
    timesteps = np.arange(45)
    import matplotlib.pyplot as plt
    gt_111 = []
    x = []
    y = []
    z = []
    rx = []
    ry = []
    rz = []
    g = []
    for i in range(30, 300, 6):
        x.append(data['action'][i][0][0])
        y.append(data['action'][i][0][1])
        z.append(data['action'][i][0][2])
        rx.append(data['action'][i][0][3])
        ry.append(data['action'][i][0][4])
        rz.append(data['action'][i][0][5])
        g.append(data['action'][i][0][6])
    gt_111.append(x)
    gt_111.append(y)
    gt_111.append(z)
    gt_111.append(rx)
    gt_111.append(ry)
    gt_111.append(rz)
    gt_111.append(g)
# 7개의 리스트를 50개의 (7,) 벡터로 전치
    trajectory_array = np.stack(gt_111, axis=1)  # shape: (50, 7)

    # 저장
    # np.save("gt_111_latest.npy", trajectory_array)
    # exit()
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
    fig_111_latest.write_html("Pick_the_yellow_cup.html")
    #Pick_the_blue_cup_on_the_right
    #Pick_the_white_cup_nearest_from_the_robot
    #Pick_the_red_cup_behind_the_purple_one
    #Pick_the_yellow_cup
    for i in range(7):
        n = f'71{i + 1}'
        plt.subplot(int(n))
        plt.plot(predictions_111_latest[i], 'b--', gt_111[i], 'r')
    plt.savefig("Pick_the_yellow_cup.png")
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
