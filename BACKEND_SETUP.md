# H2Former 后端服务搭建指南

## 文件清单

### 后端文件（Flask服务）
- `app.py` - Flask后端主程序
- `requirements.txt` - Python依赖列表
- `start_service.bat` - Windows一键启动脚本

### 前端文件（需要提供给队友）
- `web/index.html` - 已集成后端调用功能

### 模型文件（需要下载）
- `checkpoints/h2former_*.pth` - 预训练模型权重

---

## 快速启动步骤

### 方法1：一键启动（Windows）
```bash
双击运行 start_service.bat
```

### 方法2：手动启动
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python app.py
```

### 方法3：使用虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖并启动
pip install -r requirements.txt
python app.py
```

---

## API接口文档

### 1. 健康检查
**GET** `http://localhost:5000/api/health`

**响应：**
```json
{
  "success": true,
  "status": "healthy",
  "model_loaded": true,
  "models_status": {
    "fundus": true,
    "skin": true,
    "polyp": true,
    "cardiac": true
  },
  "device": "cuda"
}
```

### 2. 模型推理
**POST** `http://localhost:5000/api/infer`

**请求参数：**
- `image`: 图像文件（通过FormData上传）
- `task_type`: 任务类型（fundus/skin/polyp/cardiac）

**响应：**
```json
{
  "success": true,
  "result": {
    "overlay": "base64_encoded_image",
    "metrics": {
      "dice": "91.2",
      "iou": "84.0",
      "acc": "95.1",
      "time": "38"
    },
    "task_type": "skin",
    "model_loaded": true
  }
}
```

### 3. 模型信息
**GET** `http://localhost:5000/api/model-info`

**响应：**
```json
{
  "success": true,
  "models": {
    "fundus": {
      "loaded": true,
      "num_classes": 5,
      "device": "cuda"
    }
  }
}
```

---

## 前端集成说明

### 切换推理模式
在浏览器控制台中设置：

```javascript
// 使用后端推理（需要启动app.py）
localStorage.setItem('h2former-inference-mode', 'backend');

// 使用模拟推理（无需后端）
localStorage.setItem('h2former-inference-mode', 'mock');
```

### 前端文件说明
`web/index.html` 已包含：
- `runSegmentation()` - 主推理函数，自动根据模式选择
- `runBackendInference()` - 后端API调用函数
- `runMockSegmentation()` - 模拟推理函数
- `checkBackendService()` - 自动检测后端状态

---

## 模型权重下载

将预训练权重下载到 `checkpoints/` 目录：

```bash
# 创建checkpoints目录
mkdir -p checkpoints

# 下载模型权重（从官方仓库或HuggingFace）
# 例如：
# wget https://huggingface.co/NKUhealong/H2Former/resolve/main/h2former_fundus.pth -P checkpoints/
# wget https://huggingface.co/NKUhealong/H2Former/resolve/main/h2former_skin.pth -P checkpoints/
# wget https://huggingface.co/NKUhealong/H2Former/resolve/main/h2former_polyp.pth -P checkpoints/
# wget https://huggingface.co/NKUhealong/H2Former/resolve/main/h2former_cardiac.pth -P checkpoints/
```

**注意：** 如果没有模型权重，服务会提示 warning 但仍可启动，前端会自动回退到模拟模式。

---

## 故障排查

### 问题1：端口被占用
```bash
# 修改app.py中的端口
app.run(host='0.0.0.0', port=5001)  # 改为其他端口
```

### 问题2：CUDA内存不足
- 在CPU上运行：无需修改，代码自动回退到CPU
- 或减小batch_size（当前为1）

### 问题3：模型加载失败
- 检查模型文件路径：`./checkpoints/h2former_*.pth`
- 检查文件权限
- 查看日志输出详细错误信息

### 问题4：CORS跨域问题
- 已配置Flask-CORS，允许所有来源
- 如需限制，修改app.py中的CORS配置

---

## 性能参考

| 设备 | 推理时间 | 内存占用 |
|------|---------|---------|
| RTX 3090 GPU | ~38ms | ~2GB |
| CPU (i9-12900K) | ~800ms | ~4GB |
| CPU (普通) | ~2000ms | ~4GB |

---

## 队友需要的前端代码

已将后端调用功能集成到 `web/index.html` 中，主要包括：

1. **runBackendInference()** - 调用 `/api/infer` 接口
2. **checkBackendService()** - 检查 `/api/health` 接口  
3. **runSegmentation()** - 根据模式选择推理方式

**给队友的文件：**
- `web/index.html`（完整文件）
- `requirements.txt`
- `app.py`
- `start_service.bat`（Windows）

---

## 联系人
如有问题，请查看日志输出或联系前端开发者。
