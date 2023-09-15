# Giraffe Keypoint Detection

**Jump to: [Setup](#setup) | [Windows installation instructions](#creating-a-windows-executable) | [Usage example](#usage-example)**

This repository contains code for running a giraffe keypoint detection model. The model estimates 4 keypoint locations in images containing a single giraffe, namely: top of ossicone, top of head, neck indent, and front bottom hoof. This model was trained using the [WNI Giraffes](https://lila.science/datasets/wni-giraffes) datasets on [LILA.science](https://lila.science/). The model inference script is set up to process a directory of images at a time and generate a summary CSV file containing the keypoint locations for each input image, as well as a mirror directory that contains the keypoints overlaid on downscaled versions of each image in the input directory.

![project overview](images/main.png =800x)


## Setup

Install the development environment using the following commands:
```
conda env create -f environment.yml
conda activate giraffes
```

Download the PyTorch file from [here](https://researchlabwuopendata.blob.core.windows.net/wni-giraffe-keypoint/keypoint_rcnn_resnet50_fpn_2.pth) and place it in the root directory of this repository, then run `python convert_pytorch_model_to_onnx.py` and/or download the pre-converted ONNX model file from [here](https://researchlabwuopendata.blob.core.windows.net/wni-giraffe-keypoint/keypoint_model_7_26_2023.onnx) and place it in the root directory of this repository.

### Creating a Windows Executable

On a Windows machine, run the following commands:

```
conda env create -f environment.yml
conda activate giraffesdist
pyinstaller -F giraffes_detector_onnx.py
```

This will create a self-contained executable, `dist/giraffes_detector_onnx.exe`, that can be run on any Windows machine (inference runs on CPU). This executable will behave exactly as `python giraffes_detector_onnx.py`, but removes the need for end users to install Python and the necessary dependencies on their machine.

## Usage example

The following command will run the model on all images in the `data/` directory and save the results to the `output/` directory. Additionally, the files `output/output.csv` and `output/empty_images.csv` will be created. The former contains the keypoint locations for each input image, and the latter contains the names of any images that were empty (i.e. no giraffe detected).

```
python giraffes_detector_onnx.py --input_directory data/ --output_directory output/ --model_fn keypoint_model_7_26_2023.onnx
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


## License

This project is licensed under the [MIT License](LICENSE).

The two example image files in `data/` are part of the [WNI Giraffes](https://lila.science/datasets/wni-giraffes) dataset and are licensed under the [Community Data License Agreement (permissive variant)](https://cdla.dev/permissive-1-0/) license.