# ML_learning
<p align="center">
<a href="https://www.python.org">
  <img src="https://img.shields.io/github/languages/top/angjustinl/ML_learning" alt="languages">
</a> 
<img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="python">
<img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="black">
<img src="https://img.shields.io/github/last-commit/ANGJustinl/ML_learning.svg?label=Updated&logo=github&cacheSeconds=600" alt="">
</p>


From https://gitee.com/xforcevesa/loongchip/blob/master/docs/presentation/exam.md

### 附：
# 系统能力大赛与人工智能组考核方案

- 旧有考核方案见[此处](./exam.old.md)，亦可供参考
- 考核时提问：指针对任务所实现的代码、运行过程、实现原理或某些技术、知识点的提问
- 所有链接资料仅供参考
- 新考核方案如下：

## 人工智能考核内容

以下内容可七选四完成：

- [ ] 1. 使用C/C++实现一个简单的机器学习算法
    - 具体：实现梯度下降算法，进行线性拟合等任务，并实时打印loss值
    - 要求：算法实现代码需包含注释，并附上算法的基本原理和推导过程
    - 展示方式：当场展示（可线上）+可复现源码及其文档
    - 资料：
        1. [梯度下降算法原理](https://dsfftp.readthedocs.io/zh-cn/latest/Linear-Regression/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95%E7%9A%84%E6%95%B0%E5%AD%A6%E5%8E%9F%E7%90%86.html)
        2. [梯度下降算法实现demo](https://blog.csdn.net/AbBaCl/article/details/78817775)
- [x] 2. [实现一个基于深度学习的图像分类算法](./MNIST_CNN/)
    - 具体：使用TensorFlow或PyTorch实现一个卷积神经网络，并训练模型进行图像分类
    - 要求：算法实现代码需包含注释，并附上算法的基本原理和推导过程
    - 展示方式：当场展示（可线上）+考核时提问+可复现源码及其文档
    - 备注：采用开源数据集，展示时请放出训练loss曲线
    - 资料：
        1. [卷积神经网络原理](https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html)
        2. [MNIST数据集图像分类](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [ ] 3. 实现一个基于深度学习的序列预测与生成模型
    - 具体：使用TensorFlow或PyTorch实现一个LSTM/GRU/GPT等模型选其一，并训练模型进行序列预测与生成
    - 要求：算法实现代码需包含注释，并附上算法的基本原理和推导过程
    - 展示方式：当场展示（可线上）+考核时提问+可复现源码及其文档
    - 备注：此处序列可以是时序数据。展示时请放出训练loss曲线
    - 资料：
        1. [LSTM/GRU原理](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
        2. [Transformer for Time Series](https://medium.com/intel-tech/how-to-apply-transformers-to-time-series-models-spacetimeformer-e452f2825d2e)
- [x] 4. [实现一个图像物体识别算法与模型](./Vision/)
    - 具体：使用Python或其他编程语言实现一个基于规则或深度学习的图像物体识别算法，并训练模型进行图像分类
    - 要求：算法实现代码需包含注释，并附上算法的基本原理和推导过程
    - 展示方式：当场展示（可线上）+考核时提问+可复现源码及其文档
    - 备注：可选开源数据集如COCO、VOC等，展示时请放出训练loss曲线
    - 资料：
        1. [Ultralytics](https://www.ultralytics.com/)
        2. [OpenCV Tutorial](https://opencv-python-tutorials.readthedocs.io/)
- [ ] 5. 实现一个初等的自然语言处理模型
    - 具体：使用Python或其他编程语言实现一个基于规则或深度学习的自然语言处理模型，并训练模型进行文本分类或生成
    - 要求：算法实现代码需包含注释，并附上算法的基本原理和推导过程
    - 展示方式：当场展示（可线上）+考核时提问+可复现源码及其文档
    - 备注：请实现Transformer、Mamba、RWKV及其衍生模型
    - 资料：
        1. [Transformer原理](https://blogs.nvidia.com/blog/what-is-a-transformer-model/)
        2. [LLM Comprehensive View](https://arxiv.org/abs/2401.02038)
        3. [nanoGPT](https://github.com/karpathy/nanoGPT)
- [ ] 6. 实现一个自定功能的AI智能体模型
    - 具体：使用Python或其他编程语言实现一个基于规则或深度学习的智能体模型，并训练模型进行智能决策
    - 要求：算法实现代码需包含注释，并附上算法的基本原理和推导过程
    - 展示方式：当场展示（可线上）+考核时提问+可复现源码及其文档
    - 备注：AI智能体的功能可自行DIY
    - 资料：
        1. [MoFA框架](https://github.com/moxin-org/mofa/)
        2. [Awesome AI Agents](https://github.com/e2b-dev/awesome-ai-agents)
- [x] 7. [实现一个大模型训练推理优化的方法或工具](./Speed_optimization/)
    - 具体：使用Python与C++、CUDA等，实现一个大模型训练推理优化的方法或工具，并分析其优缺点
    - 要求：算法实现代码需包含注释，并附上算法的基本原理和推导过程
    - 展示方式：当场展示（可线上）+考核时提问+可复现源码及其文档
    - 备注：可选大模型如LlaMA-2-7B等，展示时请放出优化前后推理或训练速度对比
    - 资料：
        1. [LlaMA模型](https://github.com/meta-llama/llama)
        2. [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
        3. [onnxruntime](https://github.com/microsoft/onnxruntime)
        4. [TVM](https://github.com/apache/tvm)
        5. [XLA](https://github.com/openxla/xla)

相关链接：
   1. https://paperswithcode.com/
   2. https://huggingface.co/
   3. https://www.kaggle.com/
   4. https://roboflow.com/
