# **Step-by-Step Guide: How this Blueprint Works**

Here we will describe all the steps we are taking to train an SVM model for fake audio detection.

---

## **Overview**

To train an audio classification model in machine learning, you need the following components:

- Dataset of audio files (format doesn't matter - it can be mp3, wav, etc.)
- Audio feature extraction tools to extract important information from the dataset. These same tools will be needed when you want to predict whether an audio file outside the dataset is fake or real.
- The base model - in our case, an SVM model using an RBF kernel
- Some patience! Training with SVM can take significant time

---

## **Step 1: Download the Dataset**

First, download the FOR_rerec and FOR_2sec datasets from [UncovAI's Huggingface page](https://huggingface.co/UncovAI). These are common datasets used for training models against audio spoofing and deepfakes.

Place both datasets under one folder like `datasets`. The system will check for all sub-datasets in this location and use them for training only if they follow the same format as FOR_rerec and FOR_2sec (audio files organized under FAKE or REAL folders).

The location of the file doesn't matter; you will pass its absolute path in the arguments of the AI trainer function `train_svm` in `model.py`.

## **Step 2: Create a Training Script**

Now that we have the dataset, you can create a simple Python file that uses the `train_svm` function. This will process the referenced dataset, train the model, and save it.

```python
# audio-detection-trainer.py

from fake_audio_detection.model import train_svm

metrics = train_svm("path/to/dataset", ".")

print("Classification Metrics:")
print("-" * 30)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name.capitalize()}: {metric_value:.4f}")
```

To run it do

```bash
cd src
python3 -m fake_audio_detection.audio-detection-trainer
```

## **Step 3: Update the Model Path**

After training, `train_svm` will print the trained model's location:

```bash
Model saved to: path/to/my_model.joblib
```

Now in `demo/app.py`, you can modify the following code:

```python
# app.py

MODEL_PATH = os.path.join(APP_DIR, "model/noma-1")
```

Replace it with:

```python
# app.py

MODEL_PATH = "path/to/my_model.joblib"
```

Congratulations! You can now test your model using Streamlit:

```bash
# from repository root
./demo/run.sh
```

## 🎨 **Customizing the Blueprint**

To better understand how you can tailor this Blueprint to suit your specific needs, please visit the **[Customization Guide](customization.md)**.
