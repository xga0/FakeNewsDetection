# Fake News Detection

This repository contains implementations of an LSTM-based model for Fake News Detection using pre-trained GloVe word vectors. The project includes both Keras and PyTorch implementations.

## Dataset

The dataset used in this project can be found on [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). It contains two CSV files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

## Implementations

### 1. Keras Implementation (`fake_news_detection.py`)
- Uses Keras with TensorFlow backend
- Single-layer LSTM with 128 units
- GloVe 100-dimensional word embeddings
- Early stopping based on validation loss

### 2. PyTorch Implementation (`fake_news_detection_pt.py`)
- PyTorch-based implementation
- Two-layer LSTM with 128 units and dropout
- GloVe 50-dimensional word embeddings
- Training progress visualization with tqdm
- Performance metrics and plots
- Early stopping with validation monitoring

## Model Architecture

Both implementations use a similar architecture:
1. GloVe word embeddings (frozen during training)
2. LSTM layers for sequence processing
3. Global max pooling
4. Dense layers with dropout
5. Sigmoid output for binary classification

## Results

The training history from the Keras implementation:

![history](https://raw.githubusercontent.com/xga0/FakeNewsDetection/master/img/img.png)

The testing results:

<img src="https://raw.githubusercontent.com/xga0/FakeNewsDetection/master/img/img1.png" width="600">

The PyTorch implementation generates additional visualizations and metrics:
- Training/validation accuracy and loss plots
- ROC curve with AUC score
- Detailed classification metrics including precision, recall, and F1-score

## Requirements

### Keras Implementation
- TensorFlow
- Keras
- pandas
- numpy
- scikit-learn
- matplotlib
- nltk

### PyTorch Implementation
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- nltk
- tqdm

## Usage

1. Download the dataset from Kaggle
2. Download GloVe embeddings:
   - For Keras: `glove.6B.100d.txt`
   - For PyTorch: `glove.6B.50d.txt`
3. Place the dataset and GloVe files in the project directory
4. Run either implementation:
   ```bash
   python fake_news_detection.py  # For Keras implementation
   python fake_news_detection_pt.py  # For PyTorch implementation
   ```

## Output

The PyTorch implementation saves the following in a `results` directory:
- `training_history.png`: Training and validation metrics
- `roc_curve.png`: ROC curve with AUC score
- `classification_metrics.txt`: Detailed performance metrics

## Performance

Both implementations achieve high accuracy in distinguishing between fake and real news articles. The PyTorch implementation includes additional metrics and visualizations for performance analysis.
