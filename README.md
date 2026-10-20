# Chest X-ray Multimodal Segmentation

Text-guided segmentation of infection regions in chest X-rays using the LanGuideMedSeg research architecture. This repository contains research code and a local Flask interface for experimentation with image-and-prompt segmentation.

> **Research prototype only.** This project is not a medical device and is not intended for diagnosis, treatment, triage, or other clinical decision-making. No clinical performance or clinical suitability has been established.

## Project Overview

The project combines chest X-ray image features with a corresponding text prompt to produce a pixel-level infection probability map. It preserves research workflows alongside a Flask application and the original Streamlit application.

Required datasets, annotation workbooks, trained checkpoints, and pretrained model weights are not bundled. Access to these resources and permission to use or redistribute them must be established independently. A Git checkout alone is not sufficient to reproduce inference or training.

## Research Motivation

This research explores whether language prompts describing a finding or location can guide segmentation of infected regions in chest radiographs. The work builds on LanGuideMedSeg and the QaTa-COV19 research setting. It does not establish patient-independent generalization, performance on unconstrained prompts, or clinical utility.

## Model Architecture

```text
Chest X-ray -> ConvNeXt-Tiny -> multi-scale visual features --+
                                                           +-> GuideDecoder -> infection probability map
Text prompt -> CXR-BERT -> token features ------------------+
```

The decoder combines visual features with text-guided cross-attention and upsamples to a one-channel probability map. In the documented Phase-2 workflow, the text backbone is frozen. Images use the existing MONAI preprocessing at 224 x 224; prompts are tokenized to a maximum of 24 tokens. Evaluation metrics are calculated on the model grid. Display rendering resizes probabilities for visualization and is not a new evaluation pass.

## Technologies Used

- Python 3.10
- PyTorch and MONAI
- Hugging Face Transformers, ConvNeXt-Tiny, and CXR-BERT
- Flask and Waitress for the web application
- NumPy, Pillow, pandas, Matplotlib, and openpyxl for research workflows
- pytest for automated tests

The requirement files contain version ranges and platform-specific options; they are not a complete cross-platform lockfile. See the installation notes below and choose a compatible PyTorch build for your hardware.

## Project Structure

```text
src/             Shared data, model, training, and inference implementation
webapp/          Flask application, routes, services, templates, and static assets
run.py           Flask/Waitress entry point
app.py           Preserved Streamlit application
scripts/         Training, evaluation, inference, and verification utilities
experiments/     Research experiment code, configurations, and protocols
tests/           Unit and integration tests
docs/            Dataset, checkpoint, application, and reproducibility documentation
```

Research notebooks are not included in this repository snapshot. No notebook links are provided. Do not add notebook copies or generated output without separate privacy, permissions, attribution, and research-owner review.

## Installation

Use Python 3.10 and select a PyTorch build appropriate to your operating system and device. For the Flask runtime on Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On POSIX systems, create and activate a Python 3.10 virtual environment using `python3.10 -m venv .venv` and `source .venv/bin/activate`, then install the requirements. Choose/install a compatible PyTorch build as appropriate for your platform before running the application.

`requirements-local-cuda.txt` is the existing recipe for the observed Windows CUDA 12.4 environment; it selects a CUDA-specific PyTorch index and includes the broader notebook, web-test, and legacy requirement groups. It is not a portable environment lock. A fresh-clone installation has not been verified for every operating system or accelerator.

## Flask Web Application

The Flask interface accepts a single-channel grayscale PNG or JPEG and a nonempty text prompt. It returns the input image, binary mask, and overlay. Uploads use temporary files that are removed after each request. The application does not report Dice, IoU, accuracy, or clinical confidence for an uploaded image without ground truth.

After restoring authorized runtime resources and resolving the model's intended-use terms, start the local application:

```powershell
python run.py
```

The default address is `http://127.0.0.1:5000`. The application is configured for a local research demonstration; remote deployment, authentication, TLS, and a clinical environment have not been validated. The current CXR-BERT model card says deployed use is outside its intended scope. Obtain guidance from the model owner and institution before presenting this application as suitable for deployment.

The original Streamlit application remains available through `app.py` and the optional `requirements-legacy.txt` group. This README does not change either application's behavior.

## Inference Workflow

1. Obtain authorized access to the compatible dataset, annotation manifests, checkpoint, and complete pretrained resources required by the selected workflow.
2. Restore the exact expected local resources. No public checkpoint or complete runtime bundle is provided here.
3. Start the Flask application locally.
4. Submit a supported grayscale image with a nonempty prompt.
5. Inspect the generated mask and overlay as research visualizations only.

The existing implementation preserves its model, preprocessing, threshold, and rendering behavior. Do not substitute checkpoints or regenerate provenance files to bypass validation. Checkpoint files are trusted PyTorch pickle artifacts; load only resources obtained from a source you trust.

## Evaluation Metrics

The project reports Dice, intersection over union (IoU), precision, recall, and pixel accuracy. Image-wise macro means average per-image metrics; dataset-global values pool pixel-level counts. They are distinct summaries and should not be compared as if they were the same statistic. The reported Phase-2 metrics use 224 x 224 masks and threshold 0.5.

## Results

The Phase-2 epoch-2 checkpoint was selected using the same 1,423-image validation partition summarized below. **These are validation-set selection results, not an independent test evaluation.** Values are reproduced as recorded; no new evaluation was run for this README.

| Metric | Image-wise mean | Dataset-global |
|---|---:|---:|
| Dice | 0.819383 | 0.884882 |
| IoU | 0.721275 | 0.793532 |
| Precision | 0.803224 | 0.859318 |
| Recall | 0.873062 | 0.912014 |
| Pixel accuracy | 0.970939 | 0.970939 |

For context only, the original epoch-18 checkpoint recorded macro Dice **0.185754** and macro IoU **0.106600** on a separate 2,113-image test split. This is a different checkpoint and population and is not a Phase-2 test result; the values are not a direct like-for-like comparison. No independent Phase-2 test score is claimed.

These metrics do not demonstrate clinical performance, patient-level independence, or generalization beyond the evaluated data and prompts. Their public presentation remains subject to research-owner approval.

## Reproducibility Notes

- The Phase-2 pilot continued from the original epoch-18 checkpoint for two epochs, using 5,722 training images and 1,423 validation images.
- The recorded protocol uses FP32, microbatch size 1, gradient accumulation 8, no augmentation, frozen decoder BatchNorm running statistics with trainable affine parameters, and a frozen text backbone in evaluation mode.
- The selected epoch maximized full-validation macro Dice. The validation partition is therefore a model-selection partition, not an independent test set.
- Data, annotations, checkpoints, and complete pretrained resources are external prerequisites and are not included. A Git-only clone cannot reproduce the recorded results.
- Dependency ranges do not lock the full transitive environment. Clean-clone installation and reproduction have not been demonstrated for every platform.
- The recorded full test suite had 36 passing tests and 1 failure; the remaining failure concerns a stale encoder-checkpointing expectation. Do not infer that the full suite passes.

## Dataset and Model Availability

No dataset images, masks, annotation workbooks, checkpoints, or pretrained weights are included. The current QaTa Kaggle page displays CC BY-NC-SA 3.0 IGO and requests citation, but the rights and provenance of the exact locally used version and its derivatives have not been independently verified. Rights for separately hosted text annotations are also unverified.

The code references `facebook/convnext-tiny-224` and `microsoft/BiomedVLP-CXR-BERT-specialized`. Runtime revisions are not pinned to immutable identifiers. The current CXR-BERT model card describes research/reproducibility use and states deployed use is out of scope. Verify the applicable licenses, revisions, intended-use terms, data permissions, and institutional approvals before using or redistributing these resources.

## Limitations

- This is research software, not a validated medical device or clinical decision-support system.
- No clinical performance or independent Phase-2 test result is established.
- Filename-based grouping does not by itself prove patient-level independence.
- Generalization to arbitrary prompts and populations has not been established.
- Dataset, annotation, image, checkpoint, and pretrained-model redistribution permissions are not confirmed by the repository's code license.
- Full reproduction requires external resources and a compatible environment.

## Acknowledgements

This work builds on Zhong et al., “Ariadne's Thread: Using Text Prompts to Improve Segmentation of Infected Areas from Chest X-ray Images,” MICCAI 2023 ([paper](https://arxiv.org/abs/2307.03942), [LanGuideMedSeg source](https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023)). Related resources include the [QaTa-COV19 dataset page](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset), Liu et al.'s [ConvNeXt paper](https://arxiv.org/abs/2201.03545), Boecking et al.'s [CXR-BERT paper](https://arxiv.org/abs/2204.09817), and [MONAI](https://arxiv.org/abs/2211.02701). Confirm which dataset citation applies to the exact local version used.

The repository includes a GPL-3.0 license. Confirm project-wide ownership, applicable upstream notices, and the scope of that license before redistribution. The code license does not grant rights to datasets, annotation workbooks, medical images, pretrained weights, or research results. Confirm which QaTa citation applies to the exact dataset version used.
