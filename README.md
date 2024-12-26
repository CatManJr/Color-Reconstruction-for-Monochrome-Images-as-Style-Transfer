# Color-Reconstruction-for-Monochrome-Images-as-Style-Transfer

<details>
<summary>中文</summary>

## **`这是数字图像处理课程论文的代码仓库。`**

## 摘要：
我探讨了仿照风格迁移的模式进行黑白图片色彩重建的应用可能性，在模型架构和推理上提出对该领域未来发展方��的思考。首先探索卷积神经网络（CNN）和ViT（视觉Transformers），如VGG-19，以及更先进的架构，ResNet和Swin Transformer，在风格迁移任务中的表现，然后进行更复杂的色彩重建实验。ResNet和Swin Transformer等图像分类SOTA网络在风格迁移和色彩重建任务中存在局限性，这些模型在提取特征时丢失了具象的风格信息，导致无法有效迁移艺术风格。此外，我们发现传统的卷积网络如VGG-19不仅能完成基本风格迁移任务，对简单物体黑白图片的色彩重建也令人满意，但不能很好地重建复杂图片的色彩。因此我们补充了多模态生成式模型的相关实验。在Stable Diffusion v1.4上实验表明生成式模型在色彩和内容质量上取得了显著突破，但在保持原图内容约束方面和生产强度的平衡问题上仍有优化空间。我们还搭建了基于VGG-19和SD v1.4的风格迁移/色彩重建用户端网页GUI应用。本研究的贡献在于探索了将色彩信息视为一种图片风格进行零样本风格迁移的可能性，并比较了深度特征提取网络与生成式方法在这一任务上的机制和性能差��。我们的工作不仅为黑白图像的色彩重建提供了新的视角，也为未来的风格迁移和图像转换技术的发展提供了有价值的参考。

## 仓库内容
VGG-19，ResNet，和Swin Transformer的风格迁移和色彩重建实验的 jupyter notebook 笔记本位于 `experiments` 路径下，展示了如何参考 [Dive into Deep Learning](https://d2l.ai/) 完成预训练网络风格迁移和色彩重建任务。<br>
为 VGG-19 搭建的 Gradio 网页 GUI 应用位于 `VGG19WebUI` 路径下，对 Jupyter notebook 进行了重构和类封装。  
Stable Diffusion v1.4 的全部内容位于 `Stable Diffusion experiments&WebUI` 路径下，其中 `model` 定义了 VAE、CLIP 词嵌入器、U-net，`sampler` 定义了 DDPM 和 DDIM 采样器，`latent_diffusion.py` 整合了这些子结构，`util.py` 实现了一些实用函数比如图片读取和保存。这些代码直接 fork 自 [labmlai 的相关工作](https://github.com/labmlai/annotated_deep_learning_paper_implementations)。向他们表示最真挚的敬意和感谢。其中 `util.py` 中的 `load_img` 函数应该 resize 成 64 的倍数，已进行修改并提交 PR。`webpage.py` 为 Gradio 网页 GUI 应用，并在此应用上完成了论文中的全部相关实验。  
最后，`test image` 中上传了用于实验的图片。  

## 补充内容
[一篇有关 DDPM 正向和逆向过程的推导](https://github.com/CatManJr/How-to-Teach-Your-Cat-DDPM)

</details>

<details open>
<summary>English</summary>

## **`This is the code repository for the Digital Image Processing course paper.`**

## Abstract:
I explored the feasibility of colorizing black-and-white images by emulating style transfer and provided insights into possible directions for future research in model architecture and reasoning for this field. First, I investigated the performance of convolutional neural networks (CNNs) such as VGG-19 and more advanced architectures including ResNet and Swin Transformer in style transfer tasks, followed by more complex color reconstruction experiments. We found that state-of-the-art networks like ResNet and Swin Transformer show limitations in style transfer and colorization, because these models lose concrete style details while extracting features. Additionally, traditional convolutional neural networks such as VGG-19 not only complete basic style transfer tasks on black-and-white images with simple objects but also achieve decent color reconstruction, although they fail to handle more complicated images. We then supplemented our experiments with multimodal generative models. Experiments on Stable Diffusion v1.4 demonstrate significant breakthroughs in both colorization and content quality, but optimizing the balance between original content constraints and generation strength remains a challenge. We also built a user-facing web GUI application for style transfer/colorization based on VGG-19 and SD v1.4. Our contribution is exploring the possibility of treating color information as a style for zero-sample style transfer and comparing the mechanisms and performance of deep feature extraction networks and generative methods in this task. Our findings offer a new perspective for colorization of black-and-white images and provide valuable reference points for future development of style transfer and image transformation techniques.

## Repository Contents
Jupyter notebooks for style transfer and colorization using VGG-19, ResNet, and Swin Transformer are in the `experiments` folder, demonstrating how to perform pretrained style transfer and colorization tasks by referencing [Dive into Deep Learning](https://d2l.ai/).  
The Gradio web GUI application built for VGG-19 is in the `VGG19WebUI` folder, where the Jupyter notebook code has been restructured and wrapped into classes.  
All content related to Stable Diffusion v1.4 is in the `Stable Diffusion experiments&WebUI` folder. The `model` folder defines the VAE, CLIP text encoder, and U-net; `sampler` defines the DDPM and DDIM samplers; `latent_diffusion.py` integrates these sub-modules; and `util.py` implements utility functions such as image loading/saving. These codes were directly forked from [labmlai’s project](https://github.com/labmlai/annotated_deep_learning_paper_implementations), to whom we extend our sincere gratitude. We modified the `load_img` function in `util.py` to make it resize to multiples of 64 and submitted a PR. The `webpage.py` is a Gradio web GUI where all the experiments in the paper were conducted.  
Finally, the `test image` folder contains the images used in our experiments.

## Additional Content
[An article on the derivation of forward and reverse processes in DDPM](https://github.com/CatManJr/How-to-Teach-Your-Cat-DDPM)

</details>

