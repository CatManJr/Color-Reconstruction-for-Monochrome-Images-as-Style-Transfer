# Color-Reconstruction-for-Monochrome-Images-as-Style-Transfer
## ***`这是数字图像处理课程论文的代码仓库。`***

##  摘要：

我探讨了仿照风格迁移的模式进行黑白图片色彩重建的应用可能性，在模型架构和推理上提出对该领域未来发展方向的思考。首先探索卷积神经网络（CNN）和ViT（视觉Transformers），如VGG-19，以及更先进的架构，ResNet和Swin Transformer，在风格迁移任务中的表现，然后进行更复杂的色彩重建实验。ResNet和Swin Transformer等图像分类SOTA网络在风格迁移和色彩重建任务中存在局限性，这些模型在提取特征时丢失了具象的风格信息，导致无法有效迁移艺术风格。此外，我们发现传统的卷积网络如VGG-19不仅能完成基本风格迁移任务，对简单物体黑白图片的色彩重建也令人满意，但不能很好地重建复杂图片的色彩。因此我们补充了多模态生成式模型的相关实验。在Stable Diffusion v1.4上实验表明生成式模型在色彩和内容质量上取得了显著突破，但在保持原图内容约束方面和生产强度的平衡问题上仍有优化空间。我们还搭建了基于VGG-19和SD v1.4的风格迁移/色彩重建用户端网页GUI应用。本研究的贡献在于探索了将色彩信息视为一种图片风格进行零样本风格迁移的可能性，并比较了深度特征提取网络与生成式方法在这一任务上的机制和性能差异。我们的工作不仅为黑白图像的色彩重建提供了新的视角，也为未来的风格迁移和图像转换技术的发展提供了有价值的参考。

## 仓库内容

VGG-19，ResNet，和Swin Transformer的风格迁移和色彩重建实验的jupyter notebook笔记本位于`experiments`路径下，展示了我如何参考[Dive into Deep Learning](https://d2l.ai/)完成预训练网络风格迁移和色彩重建任务。<br>

为VGG-19搭建的Gradio网页GUI应用位于`VGG19WebUI`路径下，我对Jupyter notebook进行了重构和类封装。

Stable Diffusion v1.4的全部内容位于`Stable Diffusion experiments&WebUI`路径下，其中`model`定义了VAE、CLIP词嵌入器、U-net，`sampler`定义了DDPM和DDIM采样器，`latent_diffusion.py`整合了这些子结构，`util.py`实现了一些实用函数比如图片读取和保存。这些代码直接fork自[labmlai的相关工作](https://github.com/labmlai/annotated_deep_learning_paper_implementations)。向他们表示我最真挚的敬意和感谢。其中`util.py`中的`load_img`函数应该resize成64的倍数，我已进行修改并提交PR，希望能为这个开源项目做出贡献。`webpage.py`则是一个Gradio网页GUI应用，我在这个应用上完成了论文中的全部相关实验。

最后，`test image`中上传了我用于实验的图片。

## 补充内容
[一篇有关DDPM正向和逆向过程的推导](https://github.com/CatManJr/How-to-Teach-Your-Cat-DDPM)

