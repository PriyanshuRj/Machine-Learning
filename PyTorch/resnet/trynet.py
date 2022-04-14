import torch
from resnet import ResNet101
def test():
    net = ResNet101(img_channel=3, num_classes=1000)
    y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    print(y.size())

if __name__ == "__main__":
    test()