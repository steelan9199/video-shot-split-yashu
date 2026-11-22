# 本机环境（绝对路径）

> 这是本技能的**唯一环境真相源**。换机器只需改本文件，`SKILL.md` 和脚本都不动。

## 解释器

```
D:\software\venvs\video-shot-split\Scripts\python.exe
```

- Python 3.12.13
- torch **2.11.0+cu128**（CUDA 12.8 运行时，`torch.cuda.is_available() == True`）
- GPU：NVIDIA GeForce RTX 4060 Ti，8188 MiB，驱动 591.86
- transnetv2-pytorch **1.0.5**
- scenedetect **0.7.3**（PySceneDetect）
- numpy 2.5.5 / opencv-python 5.1.0.95

**不要**使用 ComfyUI 的 `python_embeded\python.exe`，两者互不共享；也不要使用系统 PATH 上的 `python`（3.14，没有 torch）。

## ffmpeg

```
D:\software\ffmpeg\ffmpeg-2024-09-26-git-f43916e217-full_build\bin\ffmpeg.exe
D:\software\ffmpeg\ffmpeg-2024-09-26-git-f43916e217-full_build\bin\ffprobe.exe
```

## 模型权重

```
D:\software\venvs\video-shot-split\Lib\site-packages\transnetv2_pytorch\transnetv2-pytorch-weights.pth
```

- 30.5 MB，随 pip 包一起安装，**无需联网下载**
- `TransNetV2(device='auto')` 初始化时自动加载，不要手工 `load_state_dict`
- 备份/迁移：复制该文件即可

## 脚本

```
C:\Users\Administrator\.skills-manager\skills\video-shot-split-yashu\scripts\shot_split.py
```

## 标准调用

```powershell
& "D:\software\venvs\video-shot-split\Scripts\python.exe" `
  "C:\Users\Administrator\.skills-manager\skills\video-shot-split-yashu\scripts\shot_split.py" `
  "<视频绝对路径>" --max-sec 14 `
  --ffmpeg "D:\software\ffmpeg\ffmpeg-2024-09-26-git-f43916e217-full_build\bin\ffmpeg.exe" `
  --ffprobe "D:\software\ffmpeg\ffmpeg-2024-09-26-git-f43916e217-full_build\bin\ffprobe.exe"
```

## 验证环境是否可用

```powershell
& "D:\software\venvs\video-shot-split\Scripts\python.exe" -c "import torch, scenedetect, transnetv2_pytorch; print(torch.__version__, torch.cuda.is_available())"
```

期望输出：`2.11.0+cu128 True`

## 重装/迁移

```powershell
uv venv --python 3.12 "D:\software\venvs\video-shot-split"
uv pip install --python "D:\software\venvs\video-shot-split\Scripts\python.exe" torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "D:\software\venvs\video-shot-split\Scripts\python.exe" transnetv2-pytorch scenedetect opencv-python-headless
```

安装日志留存：`D:\software\venvs\video-shot-split\install.log`

## 本机实测（2026-11-22）

- 源片：1280×720 / 24fps / 4650 帧 / 193.794s，关键帧间隔约 5.3s
- TransNetV2 推理：约 2 秒（RTX 4060 Ti，8GB 显存占用 < 2GB）
- 结果：硬切命中 16 → 不确定带候选 0 → 合并后 15 个切点 → 真实镜头 16 段
- 按 14 秒上限均分后共 22 段（`shots.csv` 的 `boundary` 列标 `forced` 的 9 段不是真镜头边界）
- 默认重编码切完约 1 分钟；起点与源片像素差中位数 0.40，无重复
- 教训：本片 GOP 5.3s，`-c copy` 若不吸附关键帧，会把 120.958s 的切点拉回 115.625s，造成后一段整段重放上一段开头
