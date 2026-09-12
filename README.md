# Teaching a Transformer to Feel  
### Emotion Detection in English Text with DistilRoBERTa

This project investigates how well a pre-trained Transformer model can recognize emotions in English text and whether additional fine-tuning on a larger dataset improves its performance.

The starting point is the `j-hartmann/emotion-english-distilroberta-base` model, which predicts seven emotion classes:

- Anger
- Disgust
- Fear
- Joy
- Neutral
- Sadness
- Surprise

The model is first evaluated as a baseline on the DAIR-AI Emotion dataset. It is then fine-tuned on a stratified sample from the Boltuix Emotions dataset and evaluated again to measure changes in accuracy, macro F1 score, and class-level performance.

## Project Workflow

```text
Pre-trained DistilRoBERTa
        ↓
Baseline evaluation on DAIR-AI
        ↓
Error and confusion analysis
        ↓
Boltuix dataset preparation
        ↓
Label mapping and stratified split
        ↓
Transformer fine-tuning
        ↓
Validation and final evaluation
        ↓
Baseline vs. fine-tuned comparison
```

The project covers:

- Loading pre-trained models and tokenizers with Hugging Face Transformers
- Preparing and mapping emotion datasets with Pandas
- Creating stratified training and validation splits
- Tokenizing text and applying dynamic padding
- Fine-tuning DistilRoBERTa with the Hugging Face `Trainer`
- Evaluating accuracy, precision, recall, and macro F1
- Analysing confusion matrices and misclassified examples

## Results

The baseline model achieved an accuracy of **83.9%** on the DAIR-AI test set.

After fine-tuning on 30,000 sampled Boltuix observations, the observed accuracy on the same benchmark increased to **86.75%**. Performance improved particularly for anger, fear, sadness, and surprise.

These results suggest that additional supervised training can improve cross-dataset performance. However, differences between the label spaces of the two datasets limit direct comparability and require careful interpretation.

## Important Limitation

The datasets do not use identical emotion categories.

DAIR-AI contains a `love` class but no `neutral` or `disgust` observations. The selected model predicts `neutral` and `disgust`, but not `love`. Consequently, the model cannot correctly predict DAIR-AI's `love` examples, and some macro-averaged metrics are affected by unsupported or absent classes.

A stronger future evaluation would align both datasets to a shared label space before training and comparison.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Pandas
- Scikit-learn
- DistilRoBERTa
- Jupyter Notebook

## Data and Model Sources

- [Boltuix Emotions Dataset](https://huggingface.co/datasets/boltuix/emotions-dataset)
- [DAIR-AI Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)
- [Emotion English DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

The datasets and model weights are not included directly in this repository. They are loaded from Hugging Face when the notebook is executed.

## Future Work

Possible extensions include:

- Aligning both datasets to a consistent emotion taxonomy
- Checking for duplicate or overlapping samples between datasets
- Addressing class imbalance through weighted loss or resampling
- Comparing DistilRoBERTa with DeBERTa or larger RoBERTa models
- Performing hyperparameter optimization
- Evaluating the model on additional real-world text
- Adding model calibration and confidence analysis
- Deploying the classifier through an API or interactive web application

## Responsible Use

Emotion classification is inherently uncertain and context-dependent. This project is intended for educational and research purposes and should not be used for medical, psychological, employment, or other high-stakes decisions.

## Author

**Osman Türkmen**
