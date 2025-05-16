from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16


if __name__ == "__main__":

    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        #"/sdb1/ckpt/openvla_5hz_n/openvla-7b+piper5_hz+b16+lr-0.0005+lora-r32+dropout-0.0/latest",
        "/ckpt/openvla-7b",
        #attn_implementation="flash_attention_2", # [Optional] Requires `flash_attn`
        torch_dtype=dtype,
        #low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        trust_remote_code=True
    )

    with open("/sdb1/piper_5hz/validation/Align the cups/111/episode.pickle", "rb") as f:
        data = pickle.load(f)
    traj_111_latest = []
    for i in range(150):
        image = Image.fromarray(data['observation.images.table'][i])
        prompt = "In: What action should the robot take to align cups?\nOut:"
        inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)
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
    for i in range(150):
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
    timesteps = np.arange(150)
    import matplotlib.pyplot as plt
    gt_111 = []
    x = []
    y = []
    z = []
    rx = []
    ry = []
    rz = []
    g = []
    for i in range(150):
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
    fig_111_latest.write_html("fig_111_latest_b16_l32_4bit.html")
    for i in range(7):
        n = f'71{i + 1}'
        plt.subplot(int(n))
        plt.plot(predictions_111_latest[i], 'b--', gt_111[i], 'r')
    plt.savefig("fig_111_latest_b16_l32_4bit.png")
    plt.show()