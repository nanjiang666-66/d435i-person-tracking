# D435i 人体跟踪与测距

2026-09-30 深度修正：当人物向左后方仰头，旧版框内偏上区域的中位数可能落到背景，导致约 0.8–0.9 m 跳到约 1.3 m。当前改用下半部中央的躯干区域、至少 20% 有效像素支持的近处深度簇，并收紧连续帧突跳阈值。见下文“深度跳变复测”；单靠屏幕录制尚不能验证原始深度图。

本目录是 ROS 2 Humble 的 `ament_python` 包 `person_vision`。它在 CPU 上使用预训练 YOLO26n 检测人，用 ByteTrack 维护画面内的临时目标 ID，并用 D435i 对齐深度图估计目标在相机光学坐标系中的位置。控制节点以后用 C++ 编写，通过 ROS 2 话题读取结果。本阶段不会发 `/cmd_vel`。

## 运行要求

- Ubuntu 22.04、ROS 2 Humble 和 Intel RealSense D435i。
- 相机应通过 USB 3 连接，并发布彩色图、相机内参及对齐深度图。

## 1 安装依赖

在 Ubuntu 执行。若 `~/d435i_vision_venv` 已存在，跳过创建虚拟环境的那一行。

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip \
  ros-humble-cv-bridge ros-humble-message-filters python3-colcon-common-extensions
python3 -m venv --system-site-packages ~/d435i_vision_venv
source /opt/ros/humble/setup.bash
source ~/d435i_vision_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install --upgrade 'numpy<2' 'opencv-python<4.12' ultralytics 'lap>=0.5.12'
```

`--system-site-packages` 用于在虚拟环境中访问 ROS 的 `rclpy` 和 `cv_bridge`。先执行 `python -c 'import sys, rclpy, cv_bridge, message_filters, ultralytics; print(sys.executable, "imports OK")'` 检查依赖；路径应位于 `~/d435i_vision_venv/bin/`。预训练权重不纳入 GitHub 仓库，需要单独下载到 `~/person_vision_ws/models/`。

ByteTrack 首次运行还需要 `lap`。如果 Ultralytics 提示自动安装成功但建议重启运行环境，退出跟踪程序后用原命令重新启动即可。

## 2 在 Ubuntu 本地构建 ROS 包

克隆项目并在 Ubuntu 本地文件系统中构建：

```bash
mkdir -p ~/person_vision_ws/src
git clone https://github.com/nanjiang666-66/d435i-person-tracking.git \
  ~/person_vision_ws/src/person_vision
mkdir -p ~/person_vision_ws/models
curl -fL --retry 3 \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt \
  -o ~/person_vision_ws/models/yolo26n.pt
cd ~/person_vision_ws
source /opt/ros/humble/setup.bash
source ~/d435i_vision_venv/bin/activate
colcon build --symlink-install --packages-select person_vision
source install/setup.bash
ros2 pkg executables person_vision
```

最后一条应列出 `person_vision person_tracker`。源码更新后，在包目录运行 `git pull` 并重新运行 `colcon build`。若权重下载失败，可手动将权重文件放到指定路径。权重来自 [Ultralytics 官方资源](https://github.com/ultralytics/assets/releases/tag/v8.4.0)。

短时恢复逻辑更新后，可在工作空间目录运行 `python -m unittest discover -s src/person_vision/tests -v`；这验证匹配规则，真实相机效果仍需按第 3、4 节现场测试。

### 从 VMware 共享目录更新已有工作空间

若 Ubuntu 工作空间已有此包，只同步运行代码和测试，不要复制整个目录里的 `.git`：

```bash
cp -a /mnt/hgfs/ubuntu-workspace/d435i_person_tracking/person_vision/. \
  ~/person_vision_ws/src/person_vision/person_vision/
cp -a /mnt/hgfs/ubuntu-workspace/d435i_person_tracking/tests/. \
  ~/person_vision_ws/src/person_vision/tests/
cd ~/person_vision_ws
source /opt/ros/humble/setup.bash
source ~/d435i_vision_venv/bin/activate
colcon build --symlink-install --packages-select person_vision
source install/setup.bash
python -m unittest discover -s src/person_vision/tests -v
```

如果代码已推送到 GitHub，也可在干净的 Ubuntu 源码仓库中使用 `git pull` 更新；不要在同一次更新中同时使用共享目录复制和 `git pull`。

## 3 启动相机和识别包

终端一启动相机；如果相机节点已经运行，无需再启动第二次。

```bash
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

相机分辨率会影响图像传输、深度对齐和处理帧率；可先用 640×480×30 作为 CPU 基线，再按实际效果调整。`image_size` 只改变模型输入尺寸，不改变相机发布的图像大小。更换相机流配置时先停止原相机节点，再重启；用 `camera_info` 的 `width`、`height` 验证实际生效的尺寸。

终端二启动识别包：

```bash
source /opt/ros/humble/setup.bash
source ~/d435i_vision_venv/bin/activate
source ~/person_vision_ws/install/setup.bash
ros2 run person_vision person_tracker --ros-args \
  -p model:=$HOME/person_vision_ws/models/yolo26n.pt
```

若启动时报 `ModuleNotFoundError: No module named 'ultralytics'`，先确认当前虚拟环境装有依赖：

```bash
source /opt/ros/humble/setup.bash
source ~/d435i_vision_venv/bin/activate
python -c 'import sys, ultralytics; print(sys.executable, ultralytics.__version__)'
head -n 1 ~/person_vision_ws/install/person_vision/lib/person_vision/person_tracker
```

如果第一条 `python -c` 就报缺模块，回到第 1 节安装依赖；如果虚拟环境能导入，但入口脚本的首行仍指向系统 Python，则更新源码并重新构建。本包的入口会将进程切换到已激活的虚拟环境 Python。也可直接用 `python -m person_vision.tracker_node` 快速验证。

程序默认模型名为 `yolo26n.pt`，并使用 `416` 像素推理输入、CPU。以上命令指定本地权重路径，避免自动下载。也可通过 `model` 参数指定其他权重：

```bash
ros2 run person_vision person_tracker --ros-args \
  -p model:=$HOME/person_vision_ws/models/yolo26n.pt -p image_size:=416
```

若画面中衣服等物体被误检为人，先观察检测框标签末尾的置信度分数。当前默认检测阈值为 `confidence:=0.25`，启动时不必再指定。需要对比时，可通过 `-p confidence:=0.10` 恢复较低阈值；若仍误报，可试 `0.35`。阈值升高会过滤更多误报，也可能使远处、侧身或遮挡的人更容易丢失。不要只看误报是否消失，同时复测选中人左右走动、第二个人交叉和遮挡物遮挡。更换阈值需要重新启动节点。

```bash
ros2 run person_vision person_tracker --ros-args \
  -p model:=$HOME/person_vision_ws/models/yolo26n.pt \
  -p image_size:=416
```

这项参数只改变检测框筛选，不会改变 YOLO26n 的预训练权重。如果误报的置信度仍很高，需要结合更合适的模型、姿态线索或后续数据微调评估，不能靠无限提高阈值解决。

画面出现后，**单击**要跟踪的人；绿色框为选中的目标。按 `c` 清除目标，按 `q` 退出。多人出现时程序不会主动改跟别人。后续在无显示器的 NUC 上可使用 `-p show_window:=false`，并通过 ROS 参数设置已知的目标 ID；远程选人界面留到机器人阶段。

显示窗口默认 `display_scale:=2.0`，把 640×480 视频等比例显示为 1280×960，鼠标点选坐标会自动换算回原图。窗口会跟随图像尺寸变化，不再通过拖大外框来放大白色区域。先单击画面使其获得键盘焦点，再按 `+`（或 `=`）放大整幅图像、按 `-` 缩小，每次变化 0.25 倍，可在 1.0–3.0 倍间调整。它只改变屏幕显示，不改变相机话题分辨率或 YOLO 推理尺寸。若虚拟机屏幕放不下，可在启动命令后加 `-p display_scale:=1.5`；设为 `1.0` 则恢复原尺寸。

遮挡恢复试验默认使用包内 `bytetrack_recovery.yaml`，把 ByteTrack 保留丢失轨迹的长度由默认 30 帧改为 75 帧。在实测约 25 个处理帧/秒时约为 3 秒；实际时长取决于处理帧率。若要与原配置对比，可用 `-p tracker:=bytetrack.yaml` 启动；用 `ros2 param get /d435i_person_tracker tracker` 核实当前配置。

此外，当前版本默认开启 `auto_reacquire:=true`：选中的人短暂失去旧 ID 后，最多在 `reacquire_timeout:=3.0` 秒内寻找新 ID。程序比较新框与旧目标的颜色分布、画面位置和深度，要求新框至少连续出现 3 帧，且不会改选在目标丢失前就已出现的其他 ID。多个候选人难以区分、深度差太大或超过 3 秒时，不自动改选。成功时终端打印 `Auto reacquired target ID ... -> ...`。这只是短时恢复的启发式方法，**不能证明两框是同一个人**；多人相似、快速交叉或完整出画面后从远处返回仍可能失败，也有误认风险。对比旧行为可加 `-p auto_reacquire:=false`；恢复时限可通过 `-p reacquire_timeout:=3.0` 调整，先不要调长。

目标丢失时画面左上角显示 `Target: <旧 ID> LOST`，`/person/visible` 发布 `false`。蓝框仍只是普通的人体检测框，不能据此判定是原来选中的人。自动恢复成功后绿框重新出现；若保持蓝框，请手动重选。人工重选时终端会记录旧 ID 丢失了多久才切换到新 ID；这条记录不同于 `Auto reacquired` 日志。选中人深度若突然出现不合理跳变，程序暂时置 `visible=false`、显示 `depth jump` 并打印 `depth jump rejected`，不会发布这个错误距离。

请分开记录三个场景：① 选中者在画面内左右走动，绿框是否持续；② 第二个人从前方交叉，绿框是否保持在原人身上；③ 用遮挡物完全挡住选中者约 0.5、1、2、3 秒，移开后是恢复绿框、仅有蓝框，还是没有检测框。同时记录终端的 `processed ... FPS`。目标完全走出画面另记为第四种场景。

新版本在选中 ID 首次丢失时打印 `Selected ID ... missing` 和当时画面中的其他 ID；同一 ID 接回时打印 `restored after ... s`；自动认回新 ID 时打印 `Auto reacquired ...`；人工重选后打印 `Target ID changed from ... to ... after ...s missing`。短视频（包含绿框、蓝框、左上角 FPS）和这些终端日志，比只看 `/person/visible` 更容易区分检测丢失与跟踪关联丢失。

## 4 用命令检查效果

### 深度跳变复测

选中目标，先坐正，再向左后方仰头，保持人与相机的实际距离基本不变。在另一个已 `source` ROS 2 环境的终端运行 `ros2 topic echo /person/target_point --field point.z`。重点观察深度是否仍从约 0.8–0.9 m 稳定跳到约 1.3 m。改为测躯干后，数值可能与原先测头部的读数略有不同；如果深度不足或突变被拒绝，程序应暂时发布 `/person/visible=false`，而不是输出错误的背景距离。此修正已通过模拟深度图测试，但真实 D435i 效果仍需复测；如果问题持续，需要检查原始对齐深度图。

另开终端并运行：

```bash
source /opt/ros/humble/setup.bash
source ~/person_vision_ws/install/setup.bash
ros2 node list
ros2 param get /d435i_person_tracker model
ros2 topic list
ros2 topic echo /person/visible
ros2 topic echo /person/target_point
ros2 topic echo /person/target_id
```

这些 `echo` 命令会持续输出，分别在不同终端运行，按 `Ctrl+C` 停止。未选择目标时 `/person/visible` 应为 `false`。点选并测到有效深度后应变为 `true`，`/person/target_point` 应持续输出。用卷尺改变人与相机的距离，观察 `point.z` 是否相应变化。暂时遮挡目标或拔掉相机后，`/person/visible` 应变为 `false`。对视频流和处理输出测频：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic hz /person/visible
```

性能排查时优先看跟踪程序终端每约 5 秒输出的 `processed ... FPS`、`color ... FPS`、`depth ... FPS`、`callback ... ms/frame`、`conversion ... ms/frame`、`track call ... ms/frame` 和 `longest frame gap ... s`。`color`、`depth` 是识别节点分别收到的两路图像帧率，`processed` 是成功配对并处理的帧率；`callback` 是一对已处理图像在程序内的完整处理时间，`conversion` 是把 ROS 图像转为数组的时间，`track call` 是模型检测和跟踪调用时间。如果两路图像都快而 `processed` 很低，重点检查时间戳和同步；如果某一路图像就很慢，重点检查其相机配置、传输和虚拟机调度。画面左上角也显示 5 秒平均处理帧率。`/person/visible` 在超过 1 秒没有同步图像时仍会由看门狗定时发布 `false`，所以它的 `ros2 topic hz` **不等同于**真实处理帧率。

查看彩色图和对齐深度图的分辨率（各输出中的 `width`、`height`）：

```bash
ros2 topic echo --once /camera/camera/color/camera_info
ros2 topic echo --once /camera/camera/aligned_depth_to_color/camera_info
```

`image_size:=320` 只缩小模型推理输入，不改变相机发布到 ROS 的原始图像分辨率或传输数据量。

如果要隔离虚拟机图形窗口的开销，可在相同模型与 `image_size` 下暂时用 `-p show_window:=false` 运行，并比较终端中的处理帧率。此模式不显示画面、无法用鼠标选人，`visible=false` 属于正常现象。测试完退出后恢复默认窗口模式。

也可以查看或手动清除目标 ID：

```bash
ros2 param get /d435i_person_tracker target_id
ros2 param set /d435i_person_tracker target_id -1
```

## 输出接口和限制

| 话题 | 类型 | 含义 |
| --- | --- | --- |
| `/person/target_point` | `geometry_msgs/msg/PointStamped` | 相机彩色光学坐标系：`x` 向右、`y` 向下、`z` 向前，单位米 |
| `/person/target_id` | `std_msgs/msg/Int32` | 跟踪器当前临时 ID |
| `/person/visible` | `std_msgs/msg/Bool` | 当前帧目标和深度均有效；相机超过约 1 秒没有同步帧也会置为 `false` |

`PointStamped.header.stamp` 来自彩色图，C++ 控制节点将来必须检查 `visible` 和消息时间戳；目标丢失或数据过期时停车。`point` 在相机光学坐标系中，不能直接当作底盘 `base_link` 坐标使用。`16UC1` 深度默认按 0.001 米每单位转换；如果相机的深度单位不同，用 ROS 参数 `depth_scale` 调整。此包只跟踪画面中的目标，不判断其真实身份，也不识别手势。

## 学习资料

- [RealSense ROS 驱动与深度对齐](https://github.com/realsenseai/realsense-ros)
- [ROS 2 Humble 包和工作空间](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html)
- [YOLO26 预训练检测模型](https://docs.ultralytics.com/models/yolo26)
- [Ultralytics 视频跟踪](https://docs.ultralytics.com/modes/track)
- [OpenVINO CPU 部署](https://docs.ultralytics.com/integrations/openvino)
