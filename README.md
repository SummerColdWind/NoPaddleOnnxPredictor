# NoPaddleOnnxPredictor

不依赖paddlepaddle的PaddleOCR转ONNX模型后的文字识别推理工具

---
```
PaddleOCR是一个强大的OCR工具库，当我们使用PaddleOCR训练出模型后，
我们通常将其导出为推理模型后，进一步转换为ONNX模型。
然而，调用此ONNX模型进行推理，仍需要依赖paddlepaddle这一庞大的框架，
这是我们不希望看到的。倘若使用onnxruntime引擎推理，我们又难以进行图像的预处理和后处理。
本项目作为对PaddleOCR项目的二次封装，旨在提供一套解决方案，
使得开发者能够脱离paddlepaddle的依赖，轻盈地调用模型进行推理。

本项目基于PaddleOCR-release-2.7
```

## 快速开始
当你已经转换为ONNX模型后，你唯一需要做的就是导入NoPaddleOnnxPredictor，然后调用它
```python
from nopaddle import NoPaddleOnnxPredictor

predictor = NoPaddleOnnxPredictor(
    onnx='<your onnx model path>',
    dict_='<your dict txt path>', 
    use_gpu=True,  # you can choose it according to your situation.
)
res = predictor('<image path for predicted>')
print(res)
```
使用示例数据
```python
from nopaddle import NoPaddleOnnxPredictor

ONNX_MODEL_PATH = './data/model.onnx'
DICT_PATH = './data/dict.txt'
IMAGE_PATH = './data/test.png'

predictor = NoPaddleOnnxPredictor(ONNX_MODEL_PATH, DICT_PATH)
res = predictor(IMAGE_PATH)
print(res)
```
在我的机器上，推理耗时约20ms
```
Output: ypgu
```

