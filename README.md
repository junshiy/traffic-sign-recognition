# 交通标识识别与智能交互系统

基于 PyTorch 的交通标识识别系统：在 GTSRB 数据集（43 类、约 3.9 万张训练图像）上训练 CNN 分类网络，并通过 Gradio 提供交互式识别界面，测试集准确率 **92.2%**。

## 功能

- **模型训练**（`train.py`）：自定义 Dataset / DataLoader 加载 GTSRB，三段卷积 + 全连接的 CNN 分类网络，随机旋转 / 翻转数据增强 + Dropout 正则，12 轮训练并绘制 Loss / 准确率曲线
- **交互识别**（`main.py`）：基于 Gradio 的三种交互方式——图片上传识别、样本选择识别、随机抽取 5 张测试图批量测评，输出 43 类交通标识的中文语义解释（如限速、让行、注意行人）
- **模型持久化**：训练 / 推理代码分离，最优模型保存为 `traffic_43cls.pth`

## 结果

| 指标 | 数值 |
|---|---|
| 测试集准确率 | 92.2% |
| 训练轮数 | 12 epochs |

训练曲线见 `训练结果.png`。

## 运行

```bash
pip install torch torchvision gradio pandas matplotlib
python main.py   # 启动 Gradio 交互界面
python train.py  # 重新训练
```

## 技术栈

Python · PyTorch · torchvision · Gradio · Pandas · matplotlib
