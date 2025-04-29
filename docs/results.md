# **Results: Comparison with other methods & metrics**

Below you'll find our results with an SVM model trained on FOR_rerec and FOR_2sec datasets and tested on the InTheWild dataset ***(available [here](https://huggingface.co/datasets/UncovAI/InTheWild))***. We can compare these results to those from the paper [MLAAD: The Multi-Language Audio Anti-Spoofing Dataset](https://arxiv.org/pdf/2401.09512).

### Model Performance Comparison: All Models Trained on Identical Datasets
| Model | Accuracy |
|-------|----------|
| **Our SVM** | **68.9%** |
| SLL W2V2 | 57.8% |
| Whisper DF | 54.1%|
| RAWGAT-ST | 49.8% |

**Performance Gain:** Our SVM model shows an improvement of at least 11% compared to the best model (SLL W2V2) reported in the [MLAAD paper](https://arxiv.org/pdf/2401.09512).

*The InTheWild dataset is available [here](https://huggingface.co/datasets/UncovAI/InTheWild2).*

----
### detailed results
#### 🔍 **Overall Metrics**

|Metric|Value|
|---|---|
|Accuracy|0.6886|
|Precision|0.6834|
|Recall|0.6886|
|F1 Score|0.6801|
|ROC AUC|0.7484|
|Error Rate|0.3114|

#### 🧾 **Classification Report**

|Class|Label|Precision|Recall|F1 Score|Support|
|---|---|---|---|---|---|
|0|Fake|0.7068|0.8142|0.7568|46,966|
|1|Real|0.6490|0.5042|0.5675|31,991|
