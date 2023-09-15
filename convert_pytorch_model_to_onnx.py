# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.onnx
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights

device = torch.device("cpu")
model = (
    keypointrcnn_resnet50_fpn(
        weights=None,
        num_classes=2,
        num_keypoints=4,
        weights_backbone=ResNet50_Weights.DEFAULT,
    )
    .eval()
    .to(device)
)
model.load_state_dict(torch.load("keypoint_rcnn_resnet50_fpn_2.pth"))

x = torch.randn(1, 3, 300, 300, requires_grad=True)

torch.onnx.export(
    model,
    x,
    "keypoint_model_7_26_2023.onnx",
    export_params=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {2: 'height', 3: 'width'},
    }
)
