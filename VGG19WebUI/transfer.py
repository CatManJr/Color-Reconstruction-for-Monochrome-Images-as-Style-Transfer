from d2l import torch as d2l
import gradio as gr
import torch
from PIL import Image
from utils import bilateral_filter
from vgg19 import VGG19
import matplotlib.pyplot as plt
import io

def style_transfer(content_img, style_img, lr=0.3, num_epochs=50, lr_decay_epoch=50, content_weight=1, style_weight=1e3, tv_weight=10, update_callback=None):
    # 检查图像是否为None
    if content_img is None or style_img is None:
        raise ValueError("Content image and style image must be provided.")
    
    # 设置设备和图像大小
    device, image_shape = d2l.try_gpu(), content_img.size

    # 创建VGG19实例并移动到设备
    vgg19 = VGG19(device)
    vgg19.content_weight = content_weight
    vgg19.style_weight = style_weight
    vgg19.tv_weight = tv_weight

    # 获取内容和风格特征
    content_X, contents_Y = vgg19.get_contents(content_img, image_shape, device)
    _, styles_Y = vgg19.get_styles(style_img, image_shape, device)

    # 训练生成图像
    output = vgg19.train(content_X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch, update_callback)

    # 后处理并保存生成的图像
    image = vgg19.postprocess(output)
    filtered_image = bilateral_filter(image)
    return image, filtered_image

iface = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(type="pil", label="Content Image"),
        gr.Image(type="pil", label="Style Image"),
        gr.Slider(0.0, 1.0, value=0.3, label="Learning Rate"),
        gr.Slider(1, 100, value=50, label="Number of Epochs"),
        gr.Slider(1, 100, value=50, label="Learning Rate Decay Epoch"),
        gr.Slider(0.0, 10.0, value=1, label="Content Weight"),
        gr.Slider(0.0, 1e4, value=1e3, label="Style Weight"),
        gr.Slider(0.0, 100.0, value=10, label="Total Variation Weight")
    ],
    outputs=[
        gr.Image(type="pil", label="Output Image"),
        gr.Image(type="pil", label="Filtered Image")
    ],
    live=True,
    title="Style Transfer with VGG19",
    description="Upload a content image and a style image to perform style transfer using VGG19."
)

if __name__ == "__main__":
    iface.launch(share=True)