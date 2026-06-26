from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


def shuffle_channels(x: Tensor, num_groups: int) -> Tensor:
    batch, channels, h, w = x.size()
    ch_per_group = channels // num_groups
    x = x.view(batch, num_groups, ch_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, h, w)
    return x


class SplitBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("wrong stride value")
        self.stride = stride

        mid_ch = out_ch // 2
        if (self.stride == 1) and (in_ch != mid_ch << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, in_ch {in_ch} and out_ch {out_ch} values. "
                "If stride == 1 then in_ch should be equal to out_ch // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self._dw_conv(in_ch, in_ch, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_ch if (self.stride > 1) else mid_ch,
                mid_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            self._dw_conv(mid_ch, mid_ch, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _dw_conv(
        in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias, groups=in_ch)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = shuffle_channels(out, 2)
        return out


class ShuffleNet(nn.Module):
    def __init__(
        self,
        block_repeats: list[int],
        layer_channels: list[int],
        num_classes: int = 1000,
        block: Callable[..., nn.Module] = SplitBlock,
    ) -> None:
        super().__init__()

        if len(block_repeats) != 3:
            raise ValueError("expected block_repeats as list of 3 positive ints")
        if len(layer_channels) != 5:
            raise ValueError("expected layer_channels as list of 5 positive ints")
        self._layer_channels = layer_channels

        in_ch = 3
        out_ch = self._layer_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        in_ch = out_ch

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2: nn.Sequential
        self.layer3: nn.Sequential
        self.layer4: nn.Sequential
        layer_names = [f"layer{i}" for i in [2, 3, 4]]
        for name, repeats, out_ch in zip(layer_names, block_repeats, self._layer_channels[1:]):
            seq = [block(in_ch, out_ch, 2)]
            for _ in range(repeats - 1):
                seq.append(block(out_ch, out_ch, 1))
            setattr(self, name, nn.Sequential(*seq))
            in_ch = out_ch

        out_ch = self._layer_channels[-1]
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(out_ch, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_conv(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def shufflenet(
    weights_path: str,
    num_classes: int = 1000,
) -> ShuffleNet:
    model = ShuffleNet(
        block_repeats=[4, 8, 4],
        layer_channels=[24, 48, 96, 192, 1024],
        num_classes=num_classes,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def make_model():
    model = shufflenet("shufflenet.pth")
    
    return model.eval()


if __name__ == "__main__":
    model = make_model()
    test_input = torch.randn(1, 3, 224, 224)
    test_output = model(test_input)

    print("Successfully created model!")
    print(f"input size: {test_input.shape}")
    print(f"output size (number of classes): {test_output.shape}")
