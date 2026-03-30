# H2Former 医疗图像分割项目 (计算机设计大赛)

本项目基于 H2Former 模型实现视网膜病变的自动分割与标注。

## 快速开始

### 1. 环境准备
确保你的电脑已安装 Python 3.8+。

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 激活虚拟环境 (Mac/Linux)
# source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行后端
在根目录下运行：
```bash
python app.py
```
后端服务将启动在 `http://localhost:5000`。

### 3. 运行前端
直接使用浏览器打开根目录下的 `index(1).html` 即可。

## 项目结构
- `app.py`: Flask 后端接口，负责图像处理和模型推理逻辑。
- `H2Former.py`: 模型定义文件。
- `my figures/`: 存放用于演示的原始图 (`raw/`) 和预处理好的正确结果图 (`done/`)。
- `archive/`: IDRiD 数据集相关文件。
- `checkpoints/`: 模型权重存放目录。

## 功能特性
- **智能匹配**：上传图片时，若文件名在 `my figures/done/` 中有对应结果，将直接展示完美标注。
- **实时推理**：对于新图片，系统将自动调用 H2Former 模型进行实时 AI 分割。
- **多任务适配**：支持视网膜病变的五种分类标注（视盘、微动脉瘤、渗出等）。
