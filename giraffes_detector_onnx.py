# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import onnxruntime

onnxruntime.set_default_logger_severity(3)


def adjust_keypoints(keypoints, padding, reverse=False):
    left, right, top, bottom = padding

    if reverse:
        keypoints[:, 0] -= left
        keypoints[:, 1] -= top
    else:
        keypoints[:, 0] += left
        keypoints[:, 1] += top

    return keypoints


def reverse_custom_lambda(image, padding):
    _, _, top, _ = padding
    height = image.shape[0] - top
    return image[-height:, :, :]


def scale_up_keypoints(image_path, keypoints):
    image_orig = Image.open(image_path)
    image_orig = np.array(image_orig)
    img_size = image_orig.shape

    keypoints = adjust_keypoints(np.array(keypoints), (0, 0, 100, 0), reverse=True)

    keypoints[:, 0] *= img_size[0] / 200
    keypoints[:, 1] *= img_size[1] / 300

    return img_size, keypoints


def transform_image(path, rotation=0):
    image = Image.open(path)
    image = image.resize((300, 200))  # width, height format
    image = image.rotate(rotation)
    image = np.array(image, copy=True)

    # pad image to 300x300
    image = np.pad(
        image, ((100, 0), (0, 0), (0, 0)), mode="constant", constant_values=0
    )

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32)

    return image


def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def save_image_with_keypoints(image_path, keypoints, rotation, output_filename):
    image_orig = Image.open(image_path)
    image_orig = image_orig.rotate(rotation)
    image_orig = np.array(image_orig)
    keypoints = np.array(keypoints)

    plt.figure(frameon=False)
    plt.imshow(image_orig)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], color="red", label="Predicted")
    plt.legend(fontsize=12, loc="upper left")
    plt.axis("off")
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()


def make_inferences(model_fn, rotation, input_images, output_images, output_csv):
    ort_session = onnxruntime.InferenceSession(model_fn)

    image_files = [
        f
        for f in os.listdir(input_images)
        if f.lower().endswith(".jpg")
        or f.lower().endswith(".png")
        or f.lower().endswith(".jpeg")
    ]

    if len(image_files) == 0:
        print(f"No valid images found in {input_images}, exiting...")
        print("The files in the directory are:")
        for fn in os.listdir(input_images):
            print(fn)
        return

    data = []
    no_data = []
    for i, image_file in tqdm(enumerate(image_files), total=len(image_files)):
        image_path = os.path.join(input_images, image_file)
        image = transform_image(image_path, rotation)

        ort_inputs = {ort_session.get_inputs()[0].name: image[np.newaxis, ...]}
        ort_outs = ort_session.run(None, ort_inputs)

        num_results = len(ort_outs[3])

        if num_results > 0:
            keypoints = ort_outs[3][0]

            img_size, keypoints = scale_up_keypoints(image_path, keypoints)

            too_x = keypoints.tolist()[0][0]
            too_y = keypoints.tolist()[0][1]
            toh_x = keypoints.tolist()[1][0]
            toh_y = keypoints.tolist()[1][1]
            ni_x = keypoints.tolist()[2][0]
            ni_y = keypoints.tolist()[2][1]
            fbh_x = keypoints.tolist()[3][0]
            fbh_y = keypoints.tolist()[3][1]

            data.append(
                {
                    "filename": image_path,
                    "id": os.path.basename(image_file),
                    "image_size": img_size,
                    "keypoints": keypoints.tolist(),
                    "too_x": too_x,
                    "too_y": too_y,
                    "toh_x": toh_x,
                    "toh_y": toh_y,
                    "ni_x": ni_x,
                    "ni_y": ni_y,
                    "fbh_x": fbh_x,
                    "fbh_y": fbh_y,
                    "too-toh": calculate_distance(too_x, too_y, toh_x, toh_y),
                    "too-ni": calculate_distance(too_x, too_y, ni_x, ni_y),
                    "too-fbh": calculate_distance(too_x, too_y, fbh_x, fbh_y),
                    "toh-ni": calculate_distance(toh_x, toh_y, ni_x, ni_y),
                    "toh-fbh": calculate_distance(toh_x, toh_y, fbh_x, fbh_y),
                    "ni-fbh": calculate_distance(ni_x, ni_y, fbh_x, fbh_y),
                    "too_y-fbh_y": too_y - fbh_y,
                }
            )
        else:
            no_data.append({
                "filename": image_path,
                "id": os.path.basename(image_file),
            })

    no_data_fn = os.path.join(os.path.dirname(output_csv), "empty_images.csv")
    df_no_data = pd.DataFrame(no_data)
    df_no_data.to_csv(no_data_fn, index=False)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    for i in range(len(df)):
        index = i
        image_path = df["filename"][index]
        id = df["id"][index]
        keypoints = df["keypoints"][index]
        output_filename = os.path.join(output_images, f"{id}.jpg")

        save_image_with_keypoints(image_path, keypoints, rotation, output_filename)


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Giraffe keypoint detection script")

    parser.add_argument(
        "--input_directory",
        type=str,
        required=True,
        help="Path to the input directory where giraffes are located.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Path to the output directory where the giraffe images will be saved.",
    )
    parser.add_argument(
        "--output_csv_fn",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to the output CSV file that will be generated (defaults to"
            + " 'output.csv' in the `--output_directory`)."
        ),
    )
    parser.add_argument(
        "--model_fn",
        type=str,
        required=False,
        default="keypoint_model_7_26_2023.onnx",
        help="Path to a ONNX model file to load.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output files if they already exist",
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--counter_clockwise",
        action="store_true",
        help="Whether to rotate the images 90 degrees counter-clockwise.",
    )
    group.add_argument(
        "--clockwise",
        action="store_true",
        help="Whether to rotate the images 90 degrees clockwise.",
    )


    args = parser.parse_args()

    # Set the output CSV filename
    output_csv_fn = args.output_csv_fn or os.path.join(
        args.output_directory, "output.csv"
    )

    if not os.path.exists(args.model_fn):
        print(f"Model file {args.model_fn} does not exist, exiting.")
        return

    # Check to see if the output directory exists
    if os.path.exists(args.output_directory) and not args.overwrite:
        print(
            f"Output directory {args.output_directory} already exists. "
            f"Use --overwrite to overwrite the existing files."
        )
        return
    if os.path.exists(output_csv_fn) and not args.overwrite:
        print(
            f"Output CSV file {output_csv_fn} already exists. "
            f"Use --overwrite to overwrite the existing files."
        )
        return

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_directory):
        print("Creating output directory as it doesn't already exist...")
        os.makedirs(args.output_directory, exist_ok=True)
    else:
        print(
            "Warning, the output directory exists, but you have specified"
            + " `--overwrite`, you might overwrite existing files."
        )

    if not os.path.exists(os.path.dirname(output_csv_fn)):
        os.makedirs(os.path.dirname(output_csv_fn), exist_ok=True)

    rotation = 0
    if args.counter_clockwise:
        rotation = 90
    elif args.clockwise:
        rotation = -90

    # Call the function with for key-points predictions
    make_inferences(
        args.model_fn, rotation, args.input_directory, args.output_directory, output_csv_fn
    )


if __name__ == "__main__":
    main()
