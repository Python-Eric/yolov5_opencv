"""
该代码是可以将图片的预处理（减均值除方差）写入网络层中
"""

import argparse

import torch
import torch.nn as nn
import os

import sys

sys.path.append("..")

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging


class preprocess_conv_layer(nn.Module):
    """
        docstring for preprocess_conv_layer
        params : input_module， 原本的模型
        mean_value ： 预处理的均值, list， 默认[r_chan, g_chan, b_chan]顺序
        std_value : 预处理的方差
        bgr_flag ： opencv读取的Mat, 若已经转换为RBG，则为false； 否则为True
    """
    #   实际上就是利用卷积操作，替代原有的减去均值除以标准差的功能，在部署时就可以直接不用写除Resize之外的预处理了
    #   原始操作 ： (img/255 - u)/std
    #                 =====>  img*1/(255*std) - u/std
    #                 =====>  根据卷积操作：Conv2d = W * x + b
    #                 =====>  W = 1/(255*std), b = - u/std
    #   使用示例：
    #       model_A = create_model()
    #       model_output = preprocess_co_nv_layer(model_A, mean_value, std_value, BGR2RGB)
    #       onnx_export(model_output)
    #
    #   量化：：
    #       注意，若将该预处理层写入onnx， 量化时该层不应该进行处理。
    def __init__(self, input_module, mean_value=0.0, std_value=255.0, bgr_flag=False):
        super(preprocess_conv_layer, self).__init__()
        if mean_value is None:
            mean_value = [0.0, 0.0, 0.0]
        if isinstance(mean_value, float):
            mean_value = [mean_value] * 3
        if isinstance(std_value, float):
            std_value = [std_value] * 3

        self.input_module = input_module

        with torch.no_grad():
            self.conv1 = nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1))
            # 定义conv操作的权重weights
            if bgr_flag is False:  # 数据读取为RGB。 yolov5中， brg--》rbg. hwc--> chw. dataloader打包成batch
                self.conv1.weight[:, :, :, :] = 0
                self.conv1.weight[0, 0, :, :] = 1.0 / std_value[0]
                self.conv1.weight[1, 1, :, :] = 1.0 / std_value[1]
                self.conv1.weight[2, 2, :, :] = 1.0 / std_value[2]
            else:  # 数据读取为BGR
                self.conv1.weight[:, :, :, :] = 0
                self.conv1.weight[0, 2, :, :] = 1.0 / std_value[0]
                self.conv1.weight[1, 1, :, :] = 1.0 / std_value[1]
                self.conv1.weight[2, 0, :, :] = 1.0 / std_value[2]
            # 定义conv操作的偏置， bias
            self.conv1.bias[0] = -mean_value[0]/std_value[0]  # 若不传入均值，默认为0， 相当于bias=0
            self.conv1.bias[1] = -mean_value[1]/std_value[1]
            self.conv1.bias[2] = -mean_value[2]/std_value[2]
        # eval模式
        self.conv1.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.input_module(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/exp10/weights/best.pt',
                        help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    device = torch.device('cpu')
    # print(opt)
    # set_logging()

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size)).to(device)  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    # 在attempt_load的时候，已经用了model.eval()了
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
        # models.common指的是models文件夹下common.py里面的Conv卷积
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()

    model = preprocess_conv_layer(model, 0.0, 255.0, False)

    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx
        from onnxsim import simplify

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '_pre2.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False,
                          opset_version=13,
                          input_names=['images'],
                          output_names=['outputs'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, f)
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with netron app.')


