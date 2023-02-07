from pathlib import Path
import torch.nn as nn
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
from torchmetrics import TotalVariation
from torchmetrics.image.fid import FrechetInceptionDistance



from function import adaptive_instance_normalization as featMod
from function import calc_mean_std

# Custom loss functions
tv_loss = TotalVariation().to(device="cuda")
frechet_loss = FrechetInceptionDistance(feature=64, normalize=True).to(device="cuda")



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groupnum):
        super(ConvLayer, self).__init__()
        # Padding Layer
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groupnum)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        return x

class ConvTranposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, padding=0, output_padding=0):
        super(ConvTranposeLayer, self).__init__()
        # Padding Layer
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(padding)

        # Transpose Convolution Layer
        self.transp_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, groups=groups, output_padding=output_padding)

    def forward(self, x):
        if(self.padding > 0):
          x = self.reflection_pad(x)
        x = self.transp_conv_layer(x)
        return x

class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, groupnum=1):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1, groupnum=groupnum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1, groupnum=groupnum)

    def forward(self, x, weight=None, bias=None, filterMod=False):
        if filterMod:
            x1 = self.conv1(x)
            x2 = weight * x1 + bias * x
            
            x3 = self.relu(x2)
            x4 = self.conv2(x3)
            x5 = weight * x4 + bias * x3
            return x + x5
        else: 
            return x + self.conv2(self.relu(self.conv1(x)))

# Control the number of channels
slim_factor = 1

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc1 = nn.Sequential(
            ConvLayer(3, int(16*slim_factor), kernel_size=9, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), int(32*slim_factor), kernel_size=3, stride=2, groupnum=int(16*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32*slim_factor), int(32*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(32*slim_factor), int(64*slim_factor), kernel_size=3, stride=2, groupnum=int(32*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ResidualLayer(int(64*slim_factor), kernel_size=3),
            )
        self.enc2 = nn.Sequential(
            ResidualLayer(int(64*slim_factor), kernel_size=3)
            )
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        out = [x1, x2]
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec1 = ResidualLayer(int(64*slim_factor), kernel_size=3)
        self.dec2 = ResidualLayer(int(64*slim_factor), kernel_size=3)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(int(64*slim_factor), int(32*slim_factor), kernel_size=3, stride=1, groupnum=int(32*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32*slim_factor), int(32*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(int(32*slim_factor), int(16*slim_factor), kernel_size=3, stride=1, groupnum=int(16*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), int(16*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), 3, kernel_size=9, stride=1, groupnum=1)
            )
        
    def forward(self, x, s, w, b, alpha):
        x1 = featMod(x[1], s[1])
        x1 = alpha * x1 + (1-alpha) * x[1]

        x2 = self.dec1(x1, w[1], b[1], filterMod=True)
        
        x3 = featMod(x2, s[0])
        x3 = alpha * x3 + (1-alpha) * x2
        
        x4 = self.dec2(x3, w[0], b[0], filterMod=True)

        out = self.dec3(x4)
        return out

class DecoderConvTranspose(nn.Module):
    def __init__(self):
        super(DecoderConvTranspose, self).__init__()
        self.dec1 = ResidualLayer(int(64*slim_factor), kernel_size=3)
        self.dec2 = ResidualLayer(int(64*slim_factor), kernel_size=3)
        self.dec3 = nn.Sequential(
            # Upsample
            ConvTranposeLayer(int(64*slim_factor), int(16*slim_factor), kernel_size=4, stride=2, groups=int(16*slim_factor), padding=0, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(16*slim_factor), int(16*slim_factor), kernel_size=1, stride=1, groups=1, padding=0, output_padding=0),
            nn.ReLU(inplace=True),

            # Upsample
            ConvTranposeLayer(int(16*slim_factor), int(16*slim_factor), kernel_size=4, stride=2, groups=int(16*slim_factor), padding=0, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(16*slim_factor), int(16*slim_factor), kernel_size=1, stride=1, groups=1, padding=0, output_padding=0),
            nn.ReLU(inplace=True),

            # Convolve
            ConvLayer(int(16*slim_factor), 3, kernel_size=5, stride=1, groupnum=1),

            # Final Convolution
            ResidualLayer(int(16*slim_factor), kernel_size=9)
            )
        
    def forward(self, x, s, w, b, alpha):
        x1 = featMod(x[1], s[1])
        x1 = alpha * x1 + (1-alpha) * x[1]

        x2 = self.dec1(x1, w[1], b[1], filterMod=True)
        
        x3 = featMod(x2, s[0])
        x3 = alpha * x3 + (1-alpha) * x2
        
        x4 = self.dec2(x3, w[0], b[0], filterMod=True)

        out = self.dec3(x4)
        return out

class DecoderTuned(nn.Module):
    def __init__(self):
        super(DecoderTuned, self).__init__()
        self.dec1 = ResidualLayer(int(64*slim_factor), kernel_size=3)
        self.dec2 = ResidualLayer(int(64*slim_factor), kernel_size=3)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ConvLayer(int(64*slim_factor), int(32*slim_factor), kernel_size=3, stride=1, groupnum=int(32*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32*slim_factor), int(32*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ConvLayer(int(32*slim_factor), int(16*slim_factor), kernel_size=3, stride=1, groupnum=int(16*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), int(16*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),

            ConvLayer(int(16*slim_factor), int(16*slim_factor), kernel_size=7, stride=1, groupnum=int(16*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), 3, kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),

            # Final Convolution
            ConvLayer(3, 3, kernel_size=15, stride=1, groupnum=1),
            )
        
    def forward(self, x, s, w, b, alpha):
        x1 = featMod(x[1], s[1])
        x1 = alpha * x1 + (1-alpha) * x[1]

        x2 = self.dec1(x1, w[1], b[1], filterMod=True)
        
        x3 = featMod(x2, s[0])
        x3 = alpha * x3 + (1-alpha) * x2
        
        x4 = self.dec2(x3, w[0], b[0], filterMod=True)

        out = self.dec3(x4)
        return out

class Modulator(nn.Module):
    def __init__(self):
        super(Modulator, self).__init__()
        self.weight1 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )  
        self.bias1 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )
        self.weight2 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )  
        self.bias2 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )

    def forward(self, x):
        w1 = self.weight1(x[0])
        b1 = self.bias1(x[0])
        
        w2 = self.weight2(x[1])
        b2 = self.bias2(x[1])
        
        return [w1,w2], [b1,b2]


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class Net(pl.LightningModule):
    def __init__(self, vgg, content_encoder, style_encoder, modulator, decoder, lr, decay_rate, train_dataset, style_weight, content_weight, SSC_weight, TV_weight, \
        batch_size=8, sample_path='samples/', log_steps=64, n_workers=16):
        super(Net, self).__init__()
        self.save_hyperparameters(ignore=['vgg', 'content_encoder', 'style_encoder', 'modulator', 'decoder'])

        vgg_enc_layers = list(vgg.children())
        self.vgg_enc_1 = nn.Sequential(*vgg_enc_layers[:4])  # input -> relu1_1
        self.vgg_enc_2 = nn.Sequential(*vgg_enc_layers[4:11])  # relu1_1 -> relu2_1
        self.vgg_enc_3 = nn.Sequential(*vgg_enc_layers[11:18])  # relu2_1 -> relu3_1
        self.vgg_enc_4 = nn.Sequential(*vgg_enc_layers[18:31])  # relu3_1 -> relu4_1
        
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.modulator = modulator
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['vgg_enc_1', 'vgg_enc_2', 'vgg_enc_3', 'vgg_enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_vgg_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'vgg_enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


    # extract relu4_1 from input image
    def encode_vgg_content(self, input):
        for i in range(4):
            input = getattr(self, 'vgg_enc_{:d}'.format(i + 1))(input)
        return input
    
    
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
                self.mse_loss(input_std, target_std)

    def forward(self, batch, alpha=1.0):
        assert 0 <= alpha <= 1
        
        content, style = batch

        # extract style modulation signals
        style_feats = self.style_encoder(style)
        filter_weights, filter_biases = self.modulator(style_feats)

        # extract content features
        content_feats = self.content_encoder(content)

        # generate results  
        res = self.decoder(content_feats, style_feats, filter_weights, filter_biases, alpha)
        
        # vgg content and style loss
        res_feats_vgg = self.encode_with_vgg_intermediate(res)
        
        style_feats_vgg = self.encode_with_vgg_intermediate(style)
        content_feats_vgg = self.encode_vgg_content(content)

        loss_c = self.calc_content_loss(res_feats_vgg[-1], content_feats_vgg)
        loss_s = self.calc_style_loss(res_feats_vgg[0], style_feats_vgg[0])
        for i in range(1, 4):
            loss_s = loss_s + self.calc_style_loss(res_feats_vgg[i], style_feats_vgg[i])

        res_style_feats = self.style_encoder(res)
        res_filter_weights, res_filter_biases = self.modulator(res_style_feats)
        
        # style signal contrastive loss
        loss_contrastive = 0.
        for i in range(int(style.size(0))):
            pos_loss = 0.
            neg_loss = 0.

            for j in range(int(style.size(0))):
                if j==i:
                    FeatMod_loss = self.calc_style_loss(res_style_feats[0][i].unsqueeze(0), style_feats[0][j].unsqueeze(0)) + \
                                  self.calc_style_loss(res_style_feats[1][i].unsqueeze(0), style_feats[1][j].unsqueeze(0))
                    FilterMod_loss = self.calc_content_loss(res_filter_weights[0][i], filter_weights[0][j]) + \
                                    self.calc_content_loss(res_filter_weights[1][i], filter_weights[1][j]) + \
                                    self.calc_content_loss(res_filter_biases[0][i], filter_biases[0][j]) + \
                                    self.calc_content_loss(res_filter_biases[1][i], filter_biases[1][j])
                    pos_loss = FeatMod_loss + FilterMod_loss
                else:
                    FeatMod_loss = self.calc_style_loss(res_style_feats[0][i].unsqueeze(0), res_style_feats[0][j].unsqueeze(0)) + \
                                  self.calc_style_loss(res_style_feats[1][i].unsqueeze(0), style_feats[1][j].unsqueeze(0))
                    FilterMod_loss = self.calc_content_loss(res_filter_weights[0][i], filter_weights[0][j]) + \
                                    self.calc_content_loss(res_filter_weights[1][i], filter_weights[1][j]) + \
                                    self.calc_content_loss(res_filter_biases[0][i], filter_biases[0][j]) + \
                                    self.calc_content_loss(res_filter_biases[1][i], filter_biases[1][j])
                    neg_loss = neg_loss + FeatMod_loss + FilterMod_loss
                    
        
            loss_contrastive = loss_contrastive + pos_loss/neg_loss

        # total variation loss
        loss_tv = tv_loss(res)

        # apply weights
        loss_c = self.hparams.content_weight * loss_c
        loss_s = self.hparams.style_weight * loss_s
        loss_contrastive = self.hparams.SSC_weight * loss_contrastive
        loss_tv = self.hparams.TV_weight * loss_tv

        loss = loss_c + loss_s + loss_contrastive + loss_tv

        return res, loss_c, loss_s, loss_contrastive, loss_tv, loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam([
          {'params':self.content_encoder.parameters()}, 
          {'params':self.style_encoder.parameters()}, 
          {'params':self.modulator.parameters()},
          {'params':self.decoder.parameters()}
          ], lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.decay_rate)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def train_dataloader(self):
      from torch.utils.data import DataLoader
      from sampler import InfiniteSamplerWrapper
      return DataLoader(self.hparams.train_dataset, batch_size=self.hparams.batch_size, sampler=InfiniteSamplerWrapper(self.hparams.train_dataset), num_workers=self.hparams.n_workers)

    def training_step(self, batch, batch_idx):
        res, loss_c, loss_s, loss_contrastive, loss_tv, loss = self(batch)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_loss_c', loss_c, prog_bar=True)
        self.log('train_loss_s', loss_s, prog_bar=True)
        self.log('train_loss_contrastive', loss_contrastive, prog_bar=True)
        self.log('train_total_variation', loss_tv, prog_bar=True)

        return {"loss": loss, "res": res}


class TestNet(nn.Module):
    def __init__(self, content_encoder, style_encoder, modulator, decoder):
        super(TestNet, self).__init__()
        
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.modulator = modulator
        self.decoder = decoder

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        
        style_feats = self.style_encoder(style)
        filter_weights, filter_biases = self.modulator(style_feats)

        content_feats = self.content_encoder(content)
            
        res = self.decoder(content_feats, style_feats, filter_weights, filter_biases, alpha)
        
        return res

class LogPredictionsCallback(Callback):
  def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int):
        from torchvision.utils import save_image
        import wandb
        
        if(trainer.global_step % 50 == 0):
          n = 10
          content, style = batch

          styled_images = outputs['res'][:n]
          nrow = min(n, styled_images.shape[0])

          images_t = torch.concat([content[:nrow], style[:nrow], styled_images], dim=0)

          image_array = make_grid(images_t, nrow=nrow)
          
          images = wandb.Image(image_array, caption="Top: content, Middle: style, Bottom: styled images")

          output_dir = Path(pl_module.hparams.sample_path)
          output_dir.mkdir(exist_ok=True, parents=True)

          output_name = output_dir / 'output{:d}.jpg'.format(batch_idx + 1)
          save_image(images_t, str(output_name), nrow=nrow)
        
          wandb.log({"examples": images})