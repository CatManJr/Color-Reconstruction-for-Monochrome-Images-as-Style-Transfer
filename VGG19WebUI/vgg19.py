import torch
from torch import nn
import torchvision
from PIL import Image
from d2l import torch as d2l

class VGG19(nn.Module):
    def __init__(self, device):
        super(VGG19, self).__init__()
        self.device = device
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.pretrained_net = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).to(device)
        self.style_layers = [0, 3, 8, 15, 22]
        self.content_layers = [15]
        self.net = nn.Sequential(*[self.pretrained_net.features[i] for i in
                                   range(max(self.content_layers + self.style_layers) + 1)]).to(device)
        self.content_weight = 1
        self.style_weight = 1e3
        self.tv_weight = 10

    def preprocess(self, img, target_shape):
        original_width, original_height = img.size
        target_width, target_height = target_shape
        aspect_ratio = original_width / original_height
        if target_width / target_height > aspect_ratio:
            target_width = int(target_height * aspect_ratio)
        else:
            target_height = int(target_width / aspect_ratio)
        
        img = img.resize((target_width, target_height), Image.LANCZOS)
        
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        return transforms(img).unsqueeze(0).to(self.device)

    def postprocess(self, img):
        img = img[0].to(self.rgb_std.device)
        img = torch.clamp(img.permute(1, 2, 0) * self.rgb_std + self.rgb_mean, 0, 1)
        return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

    def extract_features(self, X):
        contents = []
        styles = []
        for i in range(len(self.net)):
            X = self.net[i](X)
            if i in self.style_layers:
                styles.append(X)
            if i in self.content_layers:
                contents.append(X)
        return contents, styles

    def get_contents(self, content_img, image_shape, device):
        content_X = self.preprocess(content_img, image_shape).to(device)
        contents_Y, _ = self.extract_features(content_X)
        return content_X, contents_Y

    def get_styles(self, style_img, image_shape, device):
        style_X = self.preprocess(style_img, image_shape).to(device)
        _, styles_Y = self.extract_features(style_X)
        return style_X, styles_Y

    def content_loss(self, Y_hat, Y):
        return torch.square(Y_hat - Y.detach()).mean()

    def gram(self, X):
        num_channels, n = X.shape[1], X.numel() // X.shape[1]
        X = X.reshape((num_channels, n))
        return torch.matmul(X, X.T) / (num_channels * n)

    def style_loss(self, Y_hat, gram_Y):
        return torch.square(self.gram(Y_hat) - gram_Y.detach()).mean()

    def tv_loss(self, Y_hat):
        return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                      torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

    def compute_loss(self, X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
        contents_l = [self.content_loss(Y_hat, Y) * self.content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
        styles_l = [self.style_loss(Y_hat, Y) * self.style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
        tv_l = self.tv_loss(X) * self.tv_weight
        l = sum(10 * styles_l + contents_l + [tv_l])
        return contents_l, styles_l, tv_l, l

    def get_inits(self, X, device, lr, styles_Y):
        gen_img = SynthesizedImage(X.shape).to(device)
        gen_img.weight.data.copy_(X.data)
        trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
        styles_Y_gram = [self.gram(Y) for Y in styles_Y]
        return gen_img(), styles_Y_gram, trainer

    def train(self, X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch, update_callback=None):
        X, styles_Y_gram, trainer = self.get_inits(X, device, lr, styles_Y)
        scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
        '''
        animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                                xlim=[10, num_epochs],
                                legend=['content', 'style', 'TV'],
                                ncols=2, figsize=(7, 2.5))
        '''
        for epoch in range(num_epochs):
            trainer.zero_grad()
            contents_Y_hat, styles_Y_hat = self.extract_features(X)
            contents_l, styles_l, tv_l, l = self.compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
            l.backward()
            trainer.step()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                img = self.postprocess(X)
                #animator.axes[1].imshow(img)
                #animator.add(epoch + 1, [float(sum(contents_l)),
                #                         float(sum(styles_l)), float(tv_l)])
                if update_callback:
                    update_callback(img, epoch + 1, float(sum(contents_l)), float(sum(styles_l)), float(tv_l))
        return X

class SynthesizedImage(nn.Module):
    def __init__(self, shape):
        super(SynthesizedImage, self).__init__()
        self.weight = nn.Parameter(torch.rand(*shape))

    def forward(self):
        return self.weight