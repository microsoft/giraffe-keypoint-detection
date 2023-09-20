# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
from torch import Tensor
from tqdm import tqdm

from dataset import GiraffeDataset


def convert_image_annotations_to_thumbnail(
    image: Tensor, annotations: Tensor, point_size: int = 40, resize_factor: float = 0.1
):
    """Converts an image and annotations to a thumbnail image with annotations.

    Args:
        image (torch.Tensor): Image tensor of shape (3, height, width)
        annotations (torch.Tensor): Annotation tensor of shape (num_annotations, 2)

    Returns:
        PIL.Image: Thumbnail image with annotations drawn on it.
    """
    annotations = annotations.numpy()
    image = np.array(image * 255).astype(np.uint8).transpose(1, 2, 0)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    original_width, original_height = image.size
    new_width, new_height = int(original_width * resize_factor), int(
        original_height * resize_factor
    )

    for x, y in annotations:
        if np.isnan(x) or np.isnan(y):
            continue
        bounds = (x - point_size, y - point_size, x + point_size, y + point_size)
        draw.ellipse(bounds, fill="blue")

    image = image.resize((new_width, new_height))
    return image


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = GiraffeDataset(
        args.dataset_dir, split="train", keypoint_aggregation="median"
    )

    for i in tqdm(range(len(dataset))):
        image, annotations = dataset[i]
        base_fn = os.path.basename(dataset.fns[i])
        image = convert_image_annotations_to_thumbnail(image, annotations)
        image.save(os.path.join(args.output_dir, f"{base_fn}.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generates thumbnails of each image in the WNI Giraffes dataset with"
            + " median keypoint locations overlayed."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory of the WNI Giraffes dataset.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save thumbnails to."
    )
    args = parser.parse_args()
    main(args)
