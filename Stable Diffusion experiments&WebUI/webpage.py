"""
---
title: Generate images using stable diffusion with a prompt from a given image on WebUI
summary: >
 Generate images using stable diffusion with a prompt from a given image in a web page built by Gradio.
---
Edit by Catman Jr. December 2, 2024
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import gradio as gr
import numpy as np
from torch.cuda.amp import autocast

from labml import lab, monit
from sampler.ddim import DDIMSampler
from util import load_model, load_img, save_images, set_seed


class Img2Img:
    """
    ### Image to image class
    """

    def __init__(self, *, checkpoint_path: Path,
                 ddim_steps: int = 50,
                 ddim_eta: float = 0.0):
        """
        :param checkpoint_path: is the path of the checkpoint
        :param ddim_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        self.ddim_steps = ddim_steps
        
        # Load [latent diffusion model](../latent_diffusion.html)
        self.model = load_model(checkpoint_path)
        # Get device
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # Move the model to device
        self.model.to(self.device)

        # Initialize [DDIM sampler](../sampler/ddim.html)
        self.sampler = DDIMSampler(self.model,
                                   n_steps=ddim_steps,
                                   ddim_eta=ddim_eta)

    @torch.no_grad()
    def __call__(self, *,
                 dest_path: str,
                 orig_img,
                 strength: float,
                 batch_size: int = 3,
                 prompt: str,
                 uncond_scale: float = 5.0,
                 ):
        """
        :param dest_path: is the path to store the generated images
        :param orig_img: is the image to transform
        :param strength: specifies how much of the original image should not be preserved
        :param batch_size: is the number of images to generate in a batch
        :param prompt: is the prompt to generate images with
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        """
        # 如果 orig_img 是路径，加载图像；如果是 PIL.Image.Image 对象，直接使用
        if isinstance(orig_img, str):
            image = Image.open(orig_img).convert("RGB")
        else:
            image = orig_img.convert("RGB")

        # 调整图像大小
        w, h = image.size
        w, h = map(lambda x: x - x % 64, (w, h))
        image = image.resize((w, h), Image.Resampling.LANCZOS)

        # 转换为张量并移动到设备
        image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        orig_image = 2. * image - 1.

        # 编码图像并复制 batch_size 次
        orig = self.model.autoencoder_encode(orig_image).repeat(batch_size, 1, 1, 1)

        # Make a batch of prompts
        prompts = batch_size * [prompt]
        # Get the number of steps to diffuse the original
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_index = int(strength * self.ddim_steps)


        with autocast(enabled=torch.cuda.is_available()):
            # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
            if uncond_scale != 1.0:
                un_cond = self.model.get_text_conditioning(batch_size * [""])
            else:
                un_cond = None
            # Get the prompt embeddings
            cond = self.model.get_text_conditioning(prompts)
            # Add noise to the original image
            x = self.sampler.q_sample(orig, t_index)
            # Reconstruct from the noisy image
            x = self.sampler.paint(x, cond, t_index,
                                   uncond_scale=uncond_scale,
                                   uncond_cond=un_cond)
            # Decode the image from the [autoencoder](../model/autoencoder.html)
            images = self.model.autoencoder_decode(x)

        # Decode the images from the autoencoder
        images = self.model.autoencoder_decode(x)

        # 将生成的图像从 [-1, 1] 范围转换到 [0, 1]
        images = (images + 1) / 2

        # 将图像移动到 CPU，并转换为 NumPy 数组
        images = images.cpu().numpy()  # Shape: (batch_size, channels, height, width)

        # 交换维度，将图像从 (batch_size, channels, height, width) 转换为 (batch_size, height, width, channels)
        images = np.transpose(images, (0, 2, 3, 1))  # Shape: (batch_size, height, width, channels)

        # 将图像像素值从 [0, 1] 缩放到 [0, 255]，并转换为无符号8位整数
        images = (images * 255).astype(np.uint8)

        # 将 NumPy 数组转换为 PIL.Image 对象
        pil_images = [Image.fromarray(image) for image in images]

        # 返回 PIL.Image 列表
        return pil_images
        
# 接口函数返回所有生成的图像
def img2img_interface(orig_img, prompt, strength, scale, steps, batch_size):
    # set_seed(42)
    img2img = Img2Img(
        checkpoint_path=lab.get_data_path() / 'stable-diffusion' / 'sd-v1-4.ckpt',
        ddim_steps=int(steps)
    )
    with monit.section('Generate'):
        images = img2img(
            dest_path='outputs',
            orig_img=orig_img,
            strength=strength,
            batch_size=int(batch_size),
            prompt=prompt,
            uncond_scale=scale
        )
    # 返回 PIL.Image 列表
    return images

# Gradio 接口的输出为 Gallery
iface = gr.Interface(
    fn=img2img_interface,
    inputs=[
        gr.Image(type="pil", label="输入图像"),
        gr.Textbox(lines=1, placeholder="在此输入提示词...", label="提示词", value="a painting of a cute monkey playing guitar"),
        gr.Slider(0.0, 1.0, value=0.9, label="强度"),
        gr.Slider(1.0, 20.0, value=8.0, label="无条件指导缩放"),
        gr.Slider(1, 100, step=1, value=50, label="步数"),
        gr.Slider(1, 10, step=1, value=1, label="批次大小"),
    ],
    outputs=gr.Gallery(label="生成的图像"),
    title="Stable Diffusion V1.4 Image to Image",
    description="使用 Stable Diffusion 根据给定图像和提示词生成新图像。"
)

if __name__ == "__main__":
    iface.launch()
