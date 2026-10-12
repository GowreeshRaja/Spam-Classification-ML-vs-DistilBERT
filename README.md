# Spam Classification Using Traditional Machine Learning and DistilBERT

## Project Overview

This project presents a comparative study of traditional machine learning algorithms and a transformer-based deep learning model for spam message classification.

The objective is to classify messages as **Spam** or **Ham (Not Spam)** and compare the models based on accuracy, precision, recall, F1-score, and computational efficiency.

## Dataset

**Source:** [Spam Email Classification Dataset — Kaggle](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification)

After data cleaning and duplicate removal:

- Total messages: 5,157
- Ham messages: 4,516
- Spam messages: 641
- Train/Validation/Test split: 70% / 15% / 15%

## Models Implemented

1. **Logistic Regression** — TF-IDF feature extraction with unigram and bigram features.
2. **Multinomial Naive Bayes** — TF-IDF-based probabilistic text classification.
3. **DistilBERT** — Transformer-based model fine-tuned using a custom PyTorch training loop without the Hugging Face Trainer API.

## Experimental Results

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 96.51% | 100.00% | 71.88% | 83.64% |
| Naive Bayes | 97.80% | 100.00% | 82.29% | 90.29% |
| **DistilBERT** | **98.84%** | 97.80% | **92.71%** | **95.19%** |

All results were obtained using the held-out test set.

## Technologies Used

- Python
- Google Colab
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Pandas and NumPy
- Matplotlib and Seaborn

## How to Run

1. Open the `.ipynb` notebook in Google Colab.
2. Enable GPU acceleration for DistilBERT training.
3. Install the required libraries using the notebook cells.
4. Download the dataset from Kaggle and load the CSV.
5. Execute the notebook cells sequentially to reproduce the experiments.

## Key Findings

DistilBERT achieved the highest classification accuracy and spam F1-score, demonstrating the effectiveness of transformer-based contextual representations.

Naive Bayes provided a computationally efficient alternative with strong classification performance and significantly lower training time.

## Project Files

- `Spam_Classification_Project.ipynb` — Complete implementation, training, evaluation, and visualizations.
- `Spam_Classification_Technical_Report.pdf` — Detailed technical report containing methodology, results, analysis, and conclusions.
- `README.md` — Project documentation.

## Conclusion

The comparative study demonstrates that DistilBERT provides superior spam detection performance, particularly in identifying spam messages, while traditional machine learning approaches remain valuable for resource-constrained applications.
