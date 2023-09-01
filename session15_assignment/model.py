"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import config 
import random
import math
from tqdm import tqdm
from pytorch_lightning import LightningModule, Trainer, Callback
from loss import YoloLoss
from utils import plot_couple_examples, get_evaluation_bboxes, mean_average_precision, cells_to_bboxes, check_class_accuracy
from torchsummary import summary

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
model_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, avg_pool=None):
        super().__init__()
        layers = []
        if avg_pool:
            layers = [nn.AdaptiveAvgPool2d(avg_pool)]
        self.pred = nn.Sequential( 
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            *layers,
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        x1 = self.pred(x)
        return (
            x1.reshape(x1.shape[0], 3, self.num_classes + 5, x1.shape[2], x1.shape[3])
              .permute(0, 1, 3, 4, 2)
        )
        # return (
        #     self.pred(x)
        #     .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
        #     .permute(0, 1, 3, 4, 2)
        # )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in model_config:
            print('In step ', module)
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers

class s13Model(LightningModule):
    def __init__(self, in_channels=3, num_classes=80, max_lr=1e-3, multi_scale=False):
        super().__init__()
        self.loss_fn = YoloLoss()
        self.max_lr = max_lr
        self.imgsz = 416 # image size
        self.gs = 32 # grid size
        self.multi_scale = multi_scale
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        self.training_step_outputs = []
#         self.scaler = torch.cuda.amp.GradScaler()
        self.scaled_anchors = (torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)
        # print(self.scaled_anchors)
        
    def criterion(self, out, y):
        y0, y1, y2 = (
                y[0],
                y[1],
                y[2]
            )
        # print('y and out zero are : ', y[0].shape, out[0].shape)
        # print('y and out one are : ', y[1].shape, out[1].shape)
        # print('y and out two are : ', y[2].shape, out[2].shape)
        loss = (
                    self.loss_fn(out[0], y0, self.scaled_anchors[0])
                    + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                    + self.loss_fn(out[2], y2, self.scaled_anchors[2])
                )
        return loss
    
    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                # print(x.shape)
                # print(layer)
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                # print(layer)
                # print('x and route_connections shape: ', x.shape, route_connections[-1].shape) 
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        s_count=0
        for module in model_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    ## set the avg_pool output size to 13 for first time and them double each time, 26, 52 etc
                    if s_count == 0:
                        s_count=13
                    else:
                        s_count= s_count*2
                        
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes, avg_pool=s_count),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.multi_scale:
            sz = random.randrange(self.imgsz-160, self.imgsz+32, self.gs)  #size (pick 5*32=160 before and 1*32=32 after the image size)
            sf = sz / max(x.shape[2:])  # scale factor
            # print('size and scale factor ', sz, sf)
            if sf != 1:
                ns = [math.ceil(x1 * sf / self.gs) * self.gs for x1 in x.shape[2:]]  # new shape (stretched to gs-multiple)
                # ns = 416
                x = F.interpolate(x, size=ns, mode='bilinear', align_corners=False)
                # print('resized shape : ', x.shape, len(y))

        with torch.cuda.amp.autocast():
            out = self(x)
            loss = self.criterion(out, y)
        self.training_step_outputs.append(loss)
#         losses.append(loss.item())
        train_loss = loss #self.scaler.scale(loss)
        # update progress bar
#         mean_loss = sum(losses) / len(losses)
        values = {'train_loss': train_loss}
        self.log_dict(values, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return train_loss #, train_acc

#         x = x.to(config.DEVICE)
#         y0, y1, y2 = (
#             y[0].to(config.DEVICE),
#             y[1].to(config.DEVICE),
#             y[2].to(config.DEVICE),
#         )

#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         scheduler.step()

#         # update progress bar
#         mean_loss = sum(losses) / len(losses)
#         loop.set_postfix(loss=mean_loss)

    def on_train_epoch_end(self):
            # do something with all training_step outputs, for example:
            epoch_mean = torch.stack(self.training_step_outputs).mean()
            self.log("training_epoch_mean", epoch_mean)
            # free up the memory
            self.training_step_outputs.clear()
            
    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.cuda.amp.autocast():
            out = self(x)
            loss = self.criterion(out, y)
#         losses.append(loss.item())
        val_loss = loss #self.scaler.scale(loss)
        # update progress bar
#         mean_loss = sum(losses) / len(losses)
        values = {'val_loss': val_loss}
        self.log_dict(values, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        preds = self(X_batch)
        return preds    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=config.LEARNING_RATE, 
                                     weight_decay=config.WEIGHT_DECAY
                                    )
        stepping_batches = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
#                                               max_lr=5.21E-04, 
                                                        max_lr = self.max_lr,
                                                        pct_start=0.2,
                                                        total_steps = stepping_batches,
#                                               epochs=self.max_epochs, 
                                                        div_factor=100.0, 
                                                        final_div_factor=100.0, 
                                                        anneal_strategy='linear',
#                                               steps_per_epoch=self.steps_per_epoch
                                                       )
        return ({'optimizer': optimizer, 
                 'lr_scheduler': {'scheduler': scheduler, 
                                  "monitor":"val_loss", 
                                  "interval":"step", 
                                  "frequency":1}
                })    
    
class PrintCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch > 25:
            plot_couple_examples(trainer.model, trainer.val_dataloaders, 0.6, 0.5, trainer.model.scaled_anchors)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print(f"Epoch: {trainer.current_epoch}, "
              f"Train Loss: {metrics['train_loss']:.4f}")

        check_class_accuracy(trainer.model, trainer.train_dataloader, threshold=config.CONF_THRESHOLD)

        if trainer.current_epoch > 0 and trainer.current_epoch % 5 == 0:
            print(f"Validation Loss: {metrics['val_loss']:.4f}")
            check_class_accuracy(trainer.model, trainer.val_dataloaders, threshold=config.CONF_THRESHOLD)

        print('Creating Checkpoint..')
        trainer.save_checkpoint("s13Model.ckpt")
        
        if trainer.current_epoch == trainer.max_epochs-1:

            pred_boxes, true_boxes = get_evaluation_bboxes(
                trainer.val_dataloaders,
                trainer.model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )

            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
        trainer.model.train()

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    NEW_IMAGE_SIZE = 320
    #model = YOLOv3(num_classes=num_classes)
    model = s13Model(num_classes=num_classes) #.to('cuda')
    # summary(model, (3, 256, 256))
    x = torch.randn((2, 3, NEW_IMAGE_SIZE, NEW_IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")