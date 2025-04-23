# 🎨 **Customization Guide**

This Blueprint is designed to be flexible and easily adaptable to your specific needs. This guide will walk you through some key areas you can customize to make the Blueprint your own.

---

## 🧠 **Changing the Model**

You can test other classification models by modifying the code in `train_svm.py`. Keep the pipeline with the normalizer, as it improves accuracy and generalization.

```python
pipeline = Pipeline([
        ('normalizer', CustomNormalizer(method=normalisation_type)),
        ('svm', SVC(kernel=kernel, C=C, gamma=gamma,
                   probability=probability, class_weight=class_weight,
                   random_state=random_state))
    ])
```

There are many other available models that could work well for audio classification, such as:

- **Random Forest**: Good for handling large datasets with high dimensionality
- **XGBoost**: Provides excellent performance for classification tasks with gradient boosting

## 🎵 **Feature Extraction Customization**

The current implementation uses a specific set of audio features. You can enhance or modify the feature extraction process by:

- Experimenting with different MFCC configurations

you can add them here

```python
# extract_features.py

def extract_features_from_array(y: np.ndarray, sr: float) -> np.ndarray:
    """
    Extracts audio features from a loaded audio sample.

    Args:
        y: Audio time series array
        sr: Sample rate of the audio

    Returns:
        Array of extracted audio features
    """

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    imfccs = compute_imfcc(y=y, sr=sr, n_imfcc=13)

    [...]
```

## ⚙️ **Hyperparameter Tuning**

For better model performance, consider implementing:

- Grid search or random search for optimal parameters
- Cross-validation strategies to prevent overfitting
- Bayesian optimization for more efficient parameter search

## 🤝 **Contributing to the Blueprint**

Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!
