# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class GiraffeDataset(Dataset):
    keypoint_code_to_english = {
        "too": "Top of ossicones (horns)",
        "toh": "Top of head",
        "ni": "Indent where neck meets chest",
        "fbh": "Bottom front of front hoof",
    }
    keypoint_codes = ["too", "toh", "ni", "fbh"]

    split_fns = {
        "train": "wni_giraffes_train.json",
    }
    aggregation_types = set(["median", "mean"])

    def __init__(
        self,
        root,
        split="train",
        keypoint_aggregation="median",
        transforms=None,
        skipimages=False,
        valid_idxs=None,
    ):
        super().__init__()
        assert (
            keypoint_aggregation in self.aggregation_types
        ), "keypoint_aggregation must be one of {}".format(self.aggregation_types)
        assert split in self.split_fns, "split must be one of {}".format(
            self.split_fns.keys()
        )

        self.root = root
        self.split = split
        self.keypoint_aggregation = keypoint_aggregation
        self.transforms = transforms
        self.skipimages = skipimages

        self.fns = []
        self.keypoints = []
        with open(os.path.join(root, self.split_fns[split])) as f:
            data = json.load(f)
            for annotation in data["annotations"]:
                self.fns.append(annotation["filename"])
                keypoints = dict()
                for keypoint_code in self.keypoint_codes:
                    if keypoint_code in annotation["keypoints"]:
                        keypoints[keypoint_code] = np.array(
                            list(
                                zip(
                                    annotation["keypoints"][keypoint_code]["x"],
                                    annotation["keypoints"][keypoint_code]["y"],
                                )
                            )
                        )
                self.keypoints.append(keypoints)

        if valid_idxs is not None:
            new_fns = []
            new_keypoints = []
            for idx in valid_idxs:
                new_fns.append(self.fns[idx])
                new_keypoints.append(self.keypoints[idx])
            self.fns = new_fns
            self.keypoints = new_keypoints

    def __getitem__(self, idx):
        image = torch.zeros((3, 224, 224)).to(torch.float64)
        if not self.skipimages:
            image = Image.open(os.path.join(self.root, self.fns[idx]))
            image = (transforms.functional.pil_to_tensor(image) / 255.0).double()
        keypoints = []

        if self.keypoint_aggregation == "mean":
            for keypoint_code in self.keypoint_codes:
                if (
                    keypoint_code in self.keypoints[idx]
                    and len(self.keypoints[idx][keypoint_code]) > 0
                ):
                    keypoints.append(
                        np.mean(self.keypoints[idx][keypoint_code], axis=0)
                    )
                else:
                    keypoints.append((np.nan, np.nan))
        elif self.keypoint_aggregation == "median":
            for keypoint_code in self.keypoint_codes:
                if (
                    keypoint_code in self.keypoints[idx]
                    and len(self.keypoints[idx][keypoint_code]) > 0
                ):
                    keypoints.append(
                        np.median(self.keypoints[idx][keypoint_code], axis=0)
                    )
                else:
                    keypoints.append((np.nan, np.nan))

        keypoints = torch.tensor(np.array(keypoints)).double()

        if self.transforms is not None:
            # the squeezing and unsqueezing here is because Kornia expects batch
            # #dimensions on everything
            image, keypoints = self.transforms(
                image.unsqueeze(0), keypoints.unsqueeze(0)
            )
            image = image.squeeze(0)
            keypoints = keypoints.squeeze(0)

        return image, keypoints

    def __len__(self):
        return len(self.fns)
