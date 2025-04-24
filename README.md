<p align="center">
  <picture>
    <!-- When the user prefers dark mode, show the white logo -->
    <source media="(prefers-color-scheme: dark)" srcset="./images/Blueprint-logo-white.png">
    <!-- When the user prefers light mode, show the black logo -->
    <source media="(prefers-color-scheme: light)" srcset="./images/Blueprint-logo-black.png">
    <!-- Fallback: default to the black logo -->
    <img src="./images/Blueprint-logo-black.png" width="35%" alt="Project logo"/>
  </picture>
</p>
<p align="center">
  <picture>
    <!-- Fallback: default to the black logo -->
    <img src="./images/uncovai-Mozzila.jpg" width="50%" alt="Project logo"/>
  </picture>
</p>
<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr) <br>
[![https://uncovai.com/](https://img.shields.io/badge/UncovAI-0071E5)](https://uncovai.com/) <br>
[![Docs](https://github.com/mozilla-ai/fake-audio-detection/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/fake-audio-detection/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/fake-audio-detection/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/fake-audio-detection/actions/workflows/tests.yaml/)
[![Ruff](https://github.com/mozilla-ai/fake-audio-detection/actions/workflows/lint.yaml/badge.svg?label=Ruff)](https://github.com/mozilla-ai/fake-audio-detection/actions/workflows/lint.yaml/)

[Blueprints Hub](https://developer-hub.mozilla.ai/)
| [Documentation](https://mozilla-ai.github.io/fake-audio-detection/)
| [Getting Started](https://mozilla-ai.github.io/fake-audio-detection/getting-started)
| [Contributing](CONTRIBUTING.md)

</div>

# Lightweight Machine Learning Method for Audio Forgery Detection

This blueprint guides you through training and deploying a machine learning model that effectively detects synthetic and modified audio content.

The primary objective of this model is to provide a **lightweight alternative** to deep learning approaches, allowing for easier training and deployment while delivering superior detection results. This approach makes audio forgery detection more accessible for applications with limited computational resources.

> **Warning:** Despite its advantages, this model has inherent limitations and may not detect all types of audio manipulations.


## Quick-start

Try out our demo on HF Spaces: [![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-blue)](https://huggingface.co/spaces/mozilla-ai/fake-audio-detection)


### Try it out the demo locally

Install dependencies with pip:

```bash
pip install .
```

Run the demo using the `run.sh` script:

```bash
# run this in the fake_audio_detection root directory
./demo/run.sh
```


## How It Works

This demo uses an SVM model trained with FOR_rerec and FOR_2sec datasets. You can retrieve these datasets from **[UncovAI's HuggingFace page](https://huggingface.co/datasets/UncovAI/FOR-norm/tree/main)** to train your own model. For a detailed guide, please check out the **[Step-by-step guide](https://mozilla-ai.github.io/fake-audio-detection/step-by-step-guide)**.

### Features Extraction

In the `fake-audio-detection/` folder, you'll find `extract_features.py`, which contains functions to extract features like MFCC, IMFCC, and spectral information from raw audio and datasets. There are **many features** you can add to improve model performance!

### Training

`model.py` contains the basics for training and making predictions using your model. You can modify it to find the best performance for your use case.

### Results

For a detailed overview of the datasets and the results of this method, check out the **[Results](https://mozilla-ai.github.io/fake-audio-detection/results)** section.


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! To get started, you can check out the [CONTRIBUTING.md](CONTRIBUTING.md) file.
