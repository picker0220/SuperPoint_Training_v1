# 实验室4090服务器使用指南

**面向零基础用户** - 从未接触过服务器的同学也能看懂

---

## 目录

1. [什么是服务器？](#1-什么是服务器)
2. [在开始之前需要准备什么？](#2-在开始之前需要准备什么)
3. [第一步：连接WiFi](#3-第一步连接wifi)
4. [第二步：使用VSCode连接服务器](#4-第二步使用vscode连接服务器)
5. [第三步：认识命令行](#5-第三步认识命令行)
6. [第四步：Conda环境配置](#6-第四步conda环境配置)
7. [第五步：上传项目代码](#7-第五步上传项目代码)
8. [第六步：运行训练](#8-第六步运行训练)
9. [第七步：监控训练过程](#9-第七步监控训练过程)
10. [第八步：下载训练好的模型](#10-第八步下载训练好的模型)
11. [常用命令速查表](#11-常用命令速查表)
12. [常见问题与解决](#12-常见问题与解决)

---

## 1. 什么是服务器？

### 简单理解

| 你的电脑 | vs | 服务器（我们实验室的） |
|----------|-----|----------------------|
| 你自己用 | | 很多人一起用 |
| 只有1张显卡（或没有） | | **2张 RTX 4090** (每张24GB显存) |
| 训练大模型很慢 | | 训练又快又省力 |

### 服务器的工作方式

```
你的电脑（本地）                    服务器（远程）
     │                                   │
     │  ---- SSH 连接 ---->              │
     │                                   │
     │  ---- 输入命令 ---->              │
     │                                   │
     │  <---- 显示结果 ----              │
```

> **简单说**：服务器就是一台放在实验室的"远程电脑"，你可以用你自己的电脑"遥控"它，让它帮你跑训练。

---

## 2. 在开始之前需要准备什么？

### 清单

- [ ] 已连接实验室 WiFi（这是必须的！）
- [ ] 知道自己的服务器账号和密码
- [ ] VSCode 已安装（下载地址：https://code.visualstudio.com/）
- [ ] VSCode 已安装 **Remote - SSH** 插件

### 检查 VSCode SSH 插件

1. 打开 VSCode
2. 点击左侧竖条的 Extensions（或者按 `Ctrl+Shift+X`）
3. 搜索 `Remote - SSH`
4. 看到写着 "Remote - SSH" 的插件，点击 **Install**

---

## 3. 第一步：连接WiFi

**必须先连接实验室 WiFi**，否则无法连接服务器。

1. 确保电脑已连接实验室的网络（问同学/老师WiFi名称和密码）
2. 确认能正常上网

---

## 4. 第二步：使用VSCode连接服务器

### 4.1 获取服务器信息

**问老师/师兄师姐以下信息：**
- 服务器 IP 地址（例如：`192.168.1.100`）
- 你的用户名（例如：`zhangsan`）
- 你的密码

### 4.2 在VSCode中添加SSH连接

**Step 1:** 打开 VSCode，按 `F1` 键

**Step 2:** 在搜索框输入 `Remote-SSH: Connect to Host`，点击它

**Step 3:** 选择 `Add New SSH Host`

**Step 4:** 输入连接命令，格式是：
```
ssh 你的用户名@服务器IP地址
```
例如：
```
ssh zhangsan@192.168.1.100
```

**Step 5:** 按回车，然后选择 `Linux` 作为系统

**Step 6:** 右上角会弹出输入密码的框，输入你的密码，按回车

**Step 7:** 连接成功！VSCode 左下角应该显示 `SSH: 服务器IP`

### 4.3 首次连接会看到的安全提示

第一次连接会看到类似这样的提示：
```
Are you sure you want to continue connecting (yes/no)?
```

输入 `yes` 然后按回车即可。

---

## 5. 第三步：认识命令行

连接成功后，VSCode 会打开一个终端窗口，看起来像这样：

```
zhangsan@lab-server:~$ _
```

这叫做**命令行终端**，你需要在这里输入命令来操作服务器。

### 常用命令

| 命令 | 作用 | 示例 |
|------|------|------|
| `pwd` | 查看当前目录（你现在在哪） | `pwd` |
| `ls` | 列出当前目录的文件 | `ls` |
| `ls -la` | 详细列出文件（含隐藏文件） | `ls -la` |
| `cd 目录名` | 进入某个目录 | `cd Desktop` |
| `cd ..` | 返回上一层目录 | `cd ..` |
| `cd ~` | 回到主目录 | `cd ~` |
| `mkdir 文件夹名` | 创建新文件夹 | `mkdir my_project` |
| `rm 文件名` | 删除文件 | `rm old.txt` |
| `cat 文件名` | 查看文件内容 | `cat config.py` |

### 路径知识

| 符号 | 含义 |
|------|------|
| `~` | 你的主目录，例如 `/home/zhangsan` |
| `.` | 当前目录 |
| `..` | 上一级目录 |
| `/` | 根目录（最顶层） |

### 练习：尝试输入以下命令

```bash
# 1. 查看当前位置
pwd

# 2. 列出文件
ls

# 3. 进入主目录
cd ~

# 4. 确认到了主目录
pwd

# 5. 创建一个测试文件夹
mkdir test_folder

# 6. 删除测试文件夹
rm -r test_folder
```

---

## 6. 第四步：Conda环境配置

Conda 是一个**Python环境管理工具**，可以让你同时安装多个不同版本的Python和包，互不干扰。

### 为什么需要Conda？

| 不用Conda | 用Conda |
|-----------|---------|
| 所有项目共用一个Python环境 | 每个项目有独立环境 |
| 包版本冲突很麻烦 | 版本管理简单 |
| 升级包可能搞坏其他项目 | 升级不影响其他环境 |

### 举例说明

想象你要同时跑两个项目：
- 项目A 需要 Python 3.8 + PyTorch 1.8
- 项目B 需要 Python 3.10 + PyTorch 2.0

**没有 Conda**：两个项目共用一个环境，版本会打架
**有 Conda**：两个项目有独立环境，互不影响

### 基础命令

#### 查看环境

```bash
# 列出所有环境
conda info --envs

# 你会看到类似这样的输出：
# base                  /home/zhangsan/miniconda3/etc/conda/envs/base
# pytorch               /home/zhangsan/miniconda3/etc/conda/envs/pytorch
```

#### 创建新环境

```bash
# 创建名为 pytorch 的环境，Python版本3.10
conda create -n pytorch python=3.10

# 创建时同时安装包
conda create -n my_project python=3.10 numpy pytorch
```

#### 激活/退出环境

```bash
# 激活（进入）某个环境
conda activate pytorch

# 激活后，命令行前面会显示环境名
(pytorch) zhangsan@lab-server:~$

# 退出当前环境
conda deactivate

# 退出后，前面不再显示环境名
zhangsan@lab-server:~$
```

#### 安装包

```bash
# 在当前环境安装包
conda install numpy

# 或用 pip（如果conda没有的话）
pip install torch

# 安装多个包
conda install numpy pandas matplotlib
```

#### 删除环境

```bash
# 删除整个环境
conda env remove -n my_project
```

### 为SuperPoint创建专用环境

按照以下步骤创建训练环境：

```bash
# 1. 创建新环境（Python 3.10）
conda create -n superpoint python=3.10

# 2. 激活环境
conda activate superpoint

# 3. 安装 PyTorch（GPU版本，需要CUDA 11.8）
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 安装其他依赖
conda install numpy opencv matplotlib tqdm

# 5. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

> **注意**：GPU版本需要根据服务器上的CUDA版本选择。如果不确定，先用CPU版本：
> ```bash
> conda install pytorch torchvision cpuonly -c pytorch
> ```

### 查看当前环境的包

```bash
# 列出当前环境的所有包
conda list

# 或搜索特定包
conda list | grep torch
```

### 导出/分享环境

```bash
# 导出环境到文件（方便复现）
conda env export > environment.yml

# 别人可以用这个文件创建相同的环境
conda env create -f environment.yml
```

### Conda 和 pip 的区别

| 工具 | 能安装的包 | 使用场景 |
|------|-----------|---------|
| `conda` | Python + 非Python (numpy, cudatoolkit等) | 优先使用 conda 安装 |
| `pip` | 只有 Python 包 | conda 没有的包用 pip |

### 建议

- **尽量用 conda 安装**，它能更好地处理依赖关系
- **混用没问题**，先 conda，conda 没有的再 pip

---

## 7. 第五步：上传项目代码

### 方法一：用 VSCode 直接操作（简单）

1. 在 VSCode 中打开文件夹：按 `F1` → 输入 `Open Folder` → 选择 `neural_network_training` 文件夹
2. 如果服务器上没有这个文件夹，先在服务器上创建：
   ```bash
   cd ~
   mkdir neural_network_training
   ```
3. 然后用 VSCode 打开这个文件夹：`File` → `Open Folder` → 选择它
4. 现在你可以在 VSCode 的文件浏览器中看到服务器的文件
5. 把本地的文件拖进去即可复制

### 方法二：用 SCP 命令上传（适合大文件）

在**你电脑的终端**（不是服务器的）打开一个新的终端窗口，运行：

```bash
# 把本地文件夹上传到服务器
scp -r 本地文件夹路径 你的用户名@服务器IP:目标路径

# 例如：
scp -r /Users/zhangsan/Desktop/neural_network_training zhangsan@192.168.1.100:/home/zhangsan/
```

### 方法三：用 Git 同步（推荐）

如果你的代码在 GitHub 上：

```bash
# 在服务器上克隆
cd ~
git clone https://github.com/你的用户名/你的仓库.git
```

---

## 8. 第六步：运行训练

### 8.1 查看GPU状态

连接服务器后，先检查GPU是否可用：

```bash
nvidia-smi
```

你会看到类似这样的输出：

```
+------------------------------------------------------------------+
| NVIDIA-SMI 525.147.05  Driver Version: 525.147.05                |
|----------------------+----------------------+----------------------+
| GPU  Name            |  Fan  Temp   Perf  |  Memory-Usage       |
|----------------------+----------------------+----------------------|
|   0  NVIDIA GeForce  |  40°C   N/A   |    0MiB / 24576MiB   |
|   1  NVIDIA GeForce  |  42°C   N/A   |    0MiB / 24576MiB   |
+----------------------+----------------------+----------------------+
```

看到 **2 张 NVIDIA GeForce** 和 **24576MiB** (24GB显存) 就说明正常。

### 8.2 进入项目目录

```bash
cd ~/neural_network_training
```

### 8.3 开始训练

**双卡训练（推荐）：**

```bash
# 先激活conda环境（如果创建了的话）
conda activate superpoint

# 开始双卡训练
python train_multi_gpu.py --epochs 200
```

**单卡训练：**

```bash
python train.py --epochs 100 --device cuda
```

### 8.4 训练输出示例

开始训练后，你会看到类似这样的输出：

```
============================================================
创建数据集
============================================================
  - 数据类型: 合成图像 (网格关键点)
  - 图像尺寸: 480 x 640
  - 训练样本数: 1000
  - 批次大小: 32

============================================================
创建SuperPoint模型
============================================================
  - 总参数量: 2,447,873

============================================================
开始训练
============================================================

Epoch 1:   0%|                                     | 0/32 [00:00<?, ?it/s]
```

### 8.5 停止训练

按 `Ctrl + C` 可以停止训练。

---

## 9. 第七步：监控训练过程

### 9.1 实时监控GPU使用

**方法1：命令行实时刷新**

打开一个新终端，连接到服务器，运行：

```bash
watch -n 1 nvidia-smi
```

这会每秒刷新一次，显示GPU的实时使用情况。

**方法2：使用 gpustat（更清晰）**

```bash
# 安装（如果没装）
pip install gpustat

# 监控
gpustat -cpu -i 1
```

### 9.2 后台训练（网络断开不丢失）

如果担心网络断开导致训练中断，使用 `screen`：

#### 创建后台会话

```bash
# 创建名为 train 的会话
screen -S train

# 启动训练
cd ~/neural_network_training
conda activate superpoint
python train_multi_gpu.py --epochs 200
```

#### 临时离开（训练继续）

按 `Ctrl + A`，然后按 `D`

#### 重新连接会话

```bash
screen -r train
```

#### 查看有多少个会话

```bash
screen -ls
```

#### 删除会话

```bash
screen -X -S train quit
```

### 9.3 训练日志保存

把训练输出保存到文件，方便以后查看：

```bash
# 保存到 log.txt
python train_multi_gpu.py --epochs 200 2>&1 | tee log.txt

# 或者只保存错误输出
python train_multi_gpu.py --epochs 200 2>&1 > error.log
```

---

## 10. 第八步：下载训练好的模型

训练完成后，模型保存在 `checkpoints/` 目录。

### 方法1：通过VSCode下载

1. 在 VSCode 文件浏览器中打开 `checkpoints/` 目录
2. 右键点击文件 → 选择 `Download`

### 方法2：用 scp 命令下载（在本机终端运行）

```bash
scp 你的用户名@服务器IP:/home/zhangsan/neural_network_training/checkpoints/superpoint_final.pth 本地路径
```

例如：
```bash
scp zhangsan@192.168.1.100:/home/zhangsan/neural_network_training/checkpoints/superpoint_final.pth ~/Desktop/
```

---

## 11. 常用命令速查表

### 文件操作

| 命令 | 说明 |
|------|------|
| `ls` | 列出当前目录文件 |
| `ls -la` | 详细列出（含权限） |
| `cd 目录` | 进入目录 |
| `cd ..` | 返回上级目录 |
| `cd ~` | 回主目录 |
| `mkdir 名字` | 创建目录 |
| `rm 文件` | 删除文件 |
| `rm -r 目录` | 删除目录 |
| `cp 文件1 文件2` | 复制文件 |
| `mv 文件1 文件2` | 移动/重命名 |

### 服务器操作

| 命令 | 说明 |
|------|------|
| `nvidia-smi` | 查看GPU状态 |
| `nvidia-smi -l 1` | 每秒刷新GPU |
| `top` | 查看进程 |
| `screen -S 名字` | 创建后台会话 |
| `screen -r 名字` | 恢复会话 |
| `screen -ls` | 列出所有会话 |
| `exit` | 断开连接 |

### Conda 环境

| 命令 | 说明 |
|------|------|
| `conda info --envs` | 列出所有环境 |
| `conda create -n 名字 python=x.x` | 创建新环境 |
| `conda activate 名字` | 激活环境 |
| `conda deactivate` | 退出环境 |
| `conda install 包名` | 安装包 |
| `conda list` | 列出当前环境的所有包 |
| `conda env export > 文件.yml` | 导出环境配置 |
| `conda env create -f 文件.yml` | 从文件创建环境 |
| `conda env remove -n 名字` | 删除环境 |

---

## 12. 常见问题与解决

### Q1: 连接服务器时提示 "Connection refused"

**原因**：服务器未开机或网络不通

**解决**：
1. 确认服务器已开机
2. 确认已连接实验室 WiFi
3. 确认 IP 地址正确（问老师）

---

### Q2: 输入密码后提示 "Permission denied"

**原因**：用户名或密码错误

**解决**：
1. 确认用户名和密码
2. 注意密码是区分大小写的

---

### Q3: nvidia-smi 显示 "No devices were found"

**原因**：没有正确加载NVIDIA驱动

**解决**：
1. 运行 `nvidia-smi` 看看具体错误
2. 联系管理员

---

### Q4: 训练时显示 "CUDA out of memory"

**原因**：批次大小太大，显存不够

**解决**：
1. 减小批次大小（编辑 `train_multi_gpu.py` 中的 `BATCH_SIZE_PER_GPU`）
2. 从16改到8试试

---

### Q5: 屏幕断开后训练停止了

**原因**：没有使用后台会话

**解决**：
1. 下次训练用 `screen` 启动
2. 已经在跑的训练无法恢复，只能重新开始

---

### Q6: 不知道怎么退出某个程序

**解决**：
- 大多数程序：按 `Ctrl + C` 退出
- `top` 程序：按 `Q` 退出
- `vim` 编辑器：按 `Esc` 然后输入 `:q!` 强制退出

---

### Q7: 命令行乱了/看不清

**解决**：
```bash
# 清屏
clear

# 重置终端
reset
```

---

### Q8: conda: command not found

**原因**：conda 未初始化

**解决**：
```bash
# 初始化 conda
source ~/miniconda3/etc/profile/conda.sh

# 或者问管理员 conda 的安装路径
```

---

### Q9: 创建环境时卡住了

**原因**：可能在下载，稍等几分钟

**解决**：
- 等待几分钟
- 如果太久可以按 `Ctrl+C` 重试

---

## 下一步

熟练基本操作后，你可以：

1. 学习更多 Linux 命令
2. 学习 Git 版本控制
3. 学习如何分析训练结果

---

## 获取帮助

遇到问题不要慌，可以：

1. 搜索引擎搜索错误信息
2. 问老师/师兄师姐
3. 发群里求助

---

*有问题或建议请联系：老师/实验室管理员*