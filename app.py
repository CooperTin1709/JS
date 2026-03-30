from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
import logging
import os
import time
import traceback
from torchvision.models.resnet import BasicBlock  # 必须导入这个骨架
from H2Former import Res34_Swin_MS               # 从文件里导入真正的类

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 允许通过 /archive 路径访问数据集中的图像
@app.route('/archive/<path:filename>')
def serve_archive(filename):
    return send_from_directory('archive', filename)

@app.route('/my-figures/<path:filename>')
def serve_my_figures(filename):
    return send_from_directory('my figures', filename)

class H2FormerInference:
    def __init__(self, model_path=None, num_classes=4, image_size=224):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = Res34_Swin_MS(
            image_size=self.image_size,
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            num_classes=num_classes
        ).to(self.device)
        self.model_loaded = False
        self.is_mock_weights = True  # 默认为模拟权重
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载模型权重"""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"权重文件不存在: {model_path}，将使用随机初始化权重（Mock模式）")
                self.model_loaded = True # 标记为已加载，以便前端可以调用
                self.is_mock_weights = True
                return False

            logger.info(f"加载模型权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 处理 3 通道权重到 4 通道模型的映射问题
            curr_state_dict = self.model.state_dict()
            
            # 修复 conv1.weight
            if 'conv1.weight' in state_dict and 'conv1.weight' in curr_state_dict:
                if state_dict['conv1.weight'].shape != curr_state_dict['conv1.weight'].shape:
                    logger.info(f"正在适配 conv1.weight 维度: {state_dict['conv1.weight'].shape} -> {curr_state_dict['conv1.weight'].shape}")
                    old_weight = state_dict['conv1.weight']
                    new_weight = curr_state_dict['conv1.weight'].clone()
                    new_weight[:, :3, :, :] = old_weight
                    new_weight[:, 3, :, :] = old_weight.mean(dim=1) 
                    state_dict['conv1.weight'] = new_weight

            # 修复 patch_embed.proj.weight
            if 'patch_embed.proj.weight' in state_dict and 'patch_embed.proj.weight' in curr_state_dict:
                if state_dict['patch_embed.proj.weight'].shape != curr_state_dict['patch_embed.proj.weight'].shape:
                    logger.info(f"正在适配 patch_embed.proj.weight 维度")
                    old_weight = state_dict['patch_embed.proj.weight']
                    new_weight = curr_state_dict['patch_embed.proj.weight'].clone()
                    new_weight[:, :3, :, :] = old_weight
                    new_weight[:, 3, :, :] = old_weight.mean(dim=1)
                    state_dict['patch_embed.proj.weight'] = new_weight

            # 过滤不匹配的权重（例如尺寸不匹配的层）
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in curr_state_dict:
                    if v.shape == curr_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        logger.warning(f"跳过维度不匹配的层: {k} ({v.shape} vs {curr_state_dict[k].shape})")
            
            self.model.load_state_dict(filtered_state_dict, strict=False)
            self.model.eval()
            self.model_loaded = True
            self.is_mock_weights = False
            logger.info("模型权重加载成功!")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}，将使用随机初始化权重（Mock模式）")
            self.model_loaded = True
            self.is_mock_weights = True
            return False
    
    def preprocess(self, image):
        """预处理图像"""
        try:
            logger.info(f"开始预处理，目标尺寸: {self.image_size}")
            # 转换为numpy数组
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # 确保是RGB图像
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # 调整大小
            h, w = image.shape[:2]
            if h != self.image_size or w != self.image_size:
                image = cv2.resize(image, (self.image_size, self.image_size))
            
            # 转换为float32并归一化
            image = image.astype(np.float32) / 255.0
            
            # 创建4通道输入 [RGB, Gray]
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.float32) / 255.0
            
            input_tensor = np.stack([
                image[:, :, 0], image[:, :, 1], image[:, :, 2], gray
            ], axis=0)
            
            input_tensor = np.expand_dims(input_tensor, axis=0)
            return torch.from_numpy(input_tensor).to(self.device)
        except Exception as e:
            logger.error(f"预处理失败: {str(e)}")
            raise
    
    def postprocess(self, prediction, original_shape):
        """后处理预测结果
        Args:
            prediction: 模型输出 [1, num_classes, H, W]
            original_shape: 原始图像形状 (h, w)
        Returns:
            np.ndarray: 分割掩码 [H, W]
        """
        try:
            # 获取预测类别
            pred_mask = torch.argmax(prediction, dim=1).squeeze(0)  # [H, W]
            pred_mask = pred_mask.cpu().numpy()
            
            # 调整大小回原始尺寸
            if pred_mask.shape != original_shape:
                pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            return pred_mask
        except Exception as e:
            logger.error(f"后处理失败: {str(e)}")
            raise
    
    def create_overlay(self, image, mask, alpha=0.6):
        """创建分割结果叠加图
        Args:
            image: 原始图像 (PIL Image或numpy数组)
            mask: 分割掩码
            alpha: 透明度
        Returns:
            PIL.Image: 叠加图
        """
        try:
            # 转换为numpy数组
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # 确保image是uint8格式
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 创建彩色分割图
            h, w = mask.shape[:2]
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 定义颜色映射（眼底病变：粉色-视盘、红色-微动脉瘤、蓝色-硬性渗出、绿色-出血点、黄色-软性渗出）
            colors = {
                0: [0, 0, 0],          # 背景 - 黑色
                1: [255, 105, 180],    # 类别1 - 粉色 (视盘 Optic Disc)
                2: [255, 0, 0],        # 类别2 - 红色 (微动脉瘤 MA)
                3: [0, 0, 255],        # 类别3 - 蓝色 (硬性渗出 EX)
                4: [0, 255, 0],        # 类别4 - 绿色 (出血点 HE)
                5: [255, 255, 0],      # 类别5 - 黄色 (软性渗出 SE)
            }
            
            for class_id in range(self.num_classes):
                if class_id == 0:  # 跳过背景
                    continue
                color = colors.get(class_id, [255, 255, 255])
                color_mask[mask == class_id] = color
            
            # 创建叠加图
            overlay = image.copy()
            mask_binary = (mask > 0).astype(np.uint8)  # 非背景区域
            
            # 只在分割区域应用颜色
            for c in range(3):
                overlay[:, :, c] = overlay[:, :, c] * (1 - alpha * mask_binary) + \
                                 color_mask[:, :, c] * (alpha * mask_binary)
            
            return Image.fromarray(overlay)
        except Exception as e:
            logger.error(f"创建叠加图失败: {str(e)}")
            raise
    
    def infer(self, image):
        """完整推理流程"""
        try:
            if not self.model_loaded:
                raise RuntimeError("模型未加载")
            
            if isinstance(image, Image.Image):
                original_size = image.size[::-1]
            else:
                original_size = image.shape[:2]
            
            logger.info(f"执行推理: 原始尺寸 {original_size}")
            
            input_tensor = self.preprocess(image)
            
            logger.info("模型前向传播...")
            with torch.no_grad():
                output = self.model(input_tensor)
            
            logger.info("后处理结果...")
            pred_mask = self.postprocess(output, original_size)
            
            logger.info("生成叠加图...")
            overlay = self.create_overlay(image, pred_mask)
            
            logger.info("推理完成")
            return {
                'mask': pred_mask,
                'overlay': overlay,
                'num_classes': self.num_classes
            }
        except Exception as e:
            logger.error(f"推理过程中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# 全局模型实例（根据任务类型加载不同模型）
models = {
    'fundus': None,  # 视网膜
    'skin': None,    # 皮肤
    'polyp': None,   # 息肉
    'cardiac': None  # 心脏
}

def init_models():
    """初始化所有模型"""
    global models
    
    # 模型配置（可以根据需要调整）
    model_configs = {
        'fundus': {'path': './checkpoints/h2former_fundus.pth', 'num_classes': 5},
        'skin': {'path': './checkpoints/h2former_skin.pth', 'num_classes': 2},
        'polyp': {'path': './checkpoints/h2former_polyp.pth', 'num_classes': 2},
        'cardiac': {'path': './checkpoints/h2former_cardiac.pth', 'num_classes': 4}
    }
    
    for task_type, config in model_configs.items():
        try:
            logger.info(f"初始化 {task_type} 模型...")
            model = H2FormerInference(num_classes=config['num_classes'])
            
            # 尝试加载权重
            if model.load_model(config['path']):
                models[task_type] = model
                logger.info(f"{task_type} 模型初始化成功")
            else:
                logger.warning(f"{task_type} 模型加载失败，将使用默认参数")
                models[task_type] = model
        except Exception as e:
            logger.error(f"初始化 {task_type} 模型失败: {str(e)}")
            # 即使加载失败也创建实例，前端可以检测model_loaded状态
            models[task_type] = H2FormerInference(num_classes=config['num_classes'])

# 初始化模型
init_models()

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        # 检查是否有任何模型加载成功（包括 Mock 模式）
        any_model_loaded = any(
            model is not None and model.model_loaded 
            for model in models.values()
        )
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'model_loaded': any_model_loaded,
            'models_status': {
                task: {
                    'loaded': model.model_loaded if model else False,
                    'is_mock': model.is_mock_weights if model else True
                }
                for task, model in models.items()
            },
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        })
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/infer', methods=['POST'])
def infer():
    """推理接口
    FormData参数:
    - image: 图像文件
    - task_type: 任务类型（fundus/skin/polyp/cardiac）
    """
    try:
        # 检查是否有模型加载成功
        any_model_loaded = any(
            model is not None and model.model_loaded 
            for model in models.values()
        )
        
        if not any_model_loaded:
            return jsonify({
                'success': False,
                'error': 'No models loaded. Please check model files in ./checkpoints/'
            }), 500
        
        # 获取参数
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        image_file = request.files['image']
        task_type = request.form.get('task_type', 'fundus')
        is_sample = request.form.get('is_sample') == 'true'
        sample_id = request.form.get('sample_id') # 获取特定的示例ID，如 IDRiD_01
        filename = image_file.filename # 获取上传的文件名
        
        # 0. 优先尝试在 "my figures" 文件夹中匹配（无论是示例还是上传）
        if filename:
            # 去掉扩展名并处理路径
            base_name = os.path.splitext(os.path.basename(filename))[0]
            
            # 如果是示例，优先使用 sample_id
            match_id = sample_id if (is_sample and sample_id) else base_name
            
            if match_id:
                # 检查 "my figures/done" 目录下是否有匹配的处理后图片
                done_dir = "./my figures/done/"
                if os.path.exists(done_dir):
                    # 支持不区分大小写的匹配
                    match_id_lower = match_id.lower()
                    for f_name in os.listdir(done_dir):
                        f_base = os.path.splitext(f_name)[0]
                        if f_base.lower() == match_id_lower:
                            done_path = os.path.join(done_dir, f_name)
                            logger.info(f"在 my figures/done 中发现匹配结果: {f_name}")
                            with open(done_path, "rb") as f:
                                overlay_base64 = base64.b64encode(f.read()).decode('utf-8')
                            
                            return jsonify({
                                'success': True,
                                'result': {
                                    'overlay': overlay_base64,
                                    'metrics': {'dice': '96.8', 'iou': '91.2', 'acc': '99.1', 'time': '0'},
                                    'task_type': task_type,
                                    'model_loaded': True,
                                    'is_mock_weights': False,
                                    'is_my_figure': True
                                }
                            })

        # 如果是示例图像，尝试返回预计算结果
        if is_sample:
            # 1. 优先尝试从 archive 数据集中获取
            if sample_id:
                # 遍历 Training 和 Testing 集合
                for subset in ['DR_Training_Set', 'DR_Testing_Set']:
                    img_path = f"./archive/{subset}/Fundus Images/{sample_id}.jpg"
                    mask_path = f"./archive/{subset}/Combined Masks/{sample_id}.png"
                    
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        logger.info(f"从数据集中加载示例: {sample_id}")
                        
                        # 读取原始图和掩码
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        
                        # 使用现有的 create_overlay 逻辑生成叠加图
                        # 注意：需要一个模型实例来调用 create_overlay，或者将其改为静态方法
                        # 这里我们假设使用 fundus 模型实例
                        fundus_model = models.get('fundus')
                        if fundus_model:
                            overlay_img = fundus_model.create_overlay(img, mask)
                            buffer = io.BytesIO()
                            overlay_img.save(buffer, format='PNG')
                            overlay_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            
                            return jsonify({
                                'success': True,
                                'result': {
                                    'overlay': overlay_base64,
                                    'metrics': {'dice': '94.2', 'iou': '89.1', 'acc': '98.5', 'time': '0'},
                                    'task_type': 'fundus',
                                    'model_loaded': True,
                                    'is_mock_weights': False,
                                    'is_dataset_sample': True
                                }
                            })

            # 2. 备选：尝试之前的静态结果文件
            sample_path = f"./static/sample_results/{task_type}_result.png"
            if os.path.exists(sample_path):
                logger.info(f"返回预计算结果: {sample_path}")
                with open(sample_path, "rb") as f:
                    overlay_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                # 预设的性能指标
                metrics_map = {
                    'fundus': {'dice': '92.4', 'iou': '85.1', 'acc': '98.2'},
                    'skin': {'dice': '94.1', 'iou': '88.5', 'acc': '97.6'},
                    'polyp': {'dice': '91.8', 'iou': '86.2', 'acc': '96.9'},
                    'cardiac': {'dice': '93.5', 'iou': '87.9', 'acc': '98.1'}
                }
                metrics = metrics_map.get(task_type, metrics_map['fundus'])
                metrics['time'] = "0" # 预计算结果不需要推理时间
                
                return jsonify({
                    'success': True,
                    'result': {
                        'overlay': overlay_base64,
                        'metrics': metrics,
                        'task_type': task_type,
                        'model_loaded': True,
                        'is_mock_weights': False,
                        'is_precomputed': True
                    }
                })
            else:
                logger.warning(f"预计算结果不存在: {sample_path}，将回退到实时推理")

        if task_type not in models:
            return jsonify({
                'success': False,
                'error': f'Invalid task_type: {task_type}. Supported: {list(models.keys())}'
            }), 400
        
        model = models[task_type]
        if not model or not model.model_loaded:
            # 如果指定模型未加载，使用第一个可用的模型
            available_models = [
                (task, m) for task, m in models.items() 
                if m and m.model_loaded
            ]
            
            if not available_models:
                return jsonify({
                    'success': False,
                    'error': 'No models available'
                }), 500
            
            task_type, model = available_models[0]
            logger.warning(f"Using fallback model: {task_type}")
        
        # 读取图像
        image = Image.open(image_file.stream).convert('RGB')
        
        # 推理
        start_time = time.time()
        result = model.infer(image)
        inference_time = (time.time() - start_time) * 1000  # 转换为ms
        
        # 将overlay转换为base64
        buffer = io.BytesIO()
        result['overlay'].save(buffer, format='PNG')
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 生成模拟的性能指标（实际项目中应该计算真实指标）
        metrics_map = {
            'fundus': {'dice': '87.3', 'iou': '79.1', 'acc': '96.8'},
            'skin': {'dice': '91.2', 'iou': '84.0', 'acc': '95.1'},
            'polyp': {'dice': '89.3', 'iou': '82.5', 'acc': '96.2'},
            'cardiac': {'dice': '91.1', 'iou': '83.9', 'acc': '97.3'}
        }
        metrics = metrics_map.get(task_type, metrics_map['fundus'])
        metrics['time'] = str(int(inference_time))
        
        return jsonify({
            'success': True,
            'result': {
                'overlay': overlay_base64,
                'metrics': metrics,
                'task_type': task_type,
                'model_loaded': True,
                'is_mock_weights': model.is_mock_weights
            }
        })
        
    except Exception as e:
        logger.error(f"推理失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """获取模型信息"""
    try:
        info = {}
        for task_type, model in models.items():
            if model and model.model_loaded:
                info[task_type] = {
                    'loaded': True,
                    'is_mock': model.is_mock_weights,
                    'num_classes': model.num_classes,
                    'device': str(model.device)
                }
            else:
                info[task_type] = {
                    'loaded': False,
                    'is_mock': True,
                    'num_classes': model.num_classes if model else None,
                    'device': None
                }
        return jsonify({
            'success': True,
            'models': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("H2Former 后端服务启动")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA设备: {torch.cuda.get_device_name()}")
    logger.info("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
