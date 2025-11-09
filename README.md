# Sentiment Analysis Pipeline with BERT

A collaborative project implementing a sentiment analysis pipeline using BERT (Bidirectional Encoder Representations from Transformers) for app review classification.

## 👥 Team Members

- **Adeniwa Olomola** (GitHub: [@niwaifiok-arch](https://github.com/niwaifiok-arch)) - Student 1
- **Adelewa Olomola** - Student 2

## 📋 Project Overview

This project builds a complete sentiment analysis pipeline that:
1. Loads and processes app review data
2. Cleans and tokenizes text for BERT
3. Fine-tunes a pretrained BERT model
4. Provides an inference interface for predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sentiment-analysis-pipeline
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
```bash
# Place dataset.csv in the data/ directory
```

## 📁 Project Structure

```
sentiment-analysis-pipeline/
├── src/
│   ├── data_extraction.py      # Data loading and validation
│   ├── data_processing.py      # Text cleaning and tokenization
│   ├── model.py                # BERT model training
│   └── inference.py            # Prediction interface
├── tests/
│   └── unit/
│       ├── test_data_extraction.py
│       ├── test_data_processing.py
│       ├── test_model.py
│       └── test_inference.py
├── data/
│   └── dataset.csv             # App reviews dataset
├── models/                     # Saved trained models
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 💻 Usage

### 1. Data Extraction

Load and explore the dataset:

```python
from src.data_extraction import load_dataset, get_data_statistics

# Load dataset
df = load_dataset('data/dataset.csv')

# Get statistics
stats = get_data_statistics(df)
print(f"Total reviews: {stats['total_reviews']}")
print(f"Score distribution: {stats['score_distribution']}")
```

### 2. Data Processing

Clean and tokenize text:

```python
from src.data_processing import prepare_data_for_bert
from src.data_extraction import extract_sentiment_data, convert_scores_to_sentiment

# Extract texts and scores
texts, scores = extract_sentiment_data(df)

# Convert to binary sentiment (negative/positive)
labels = convert_scores_to_sentiment(scores, binary=True)

# Prepare data for BERT
train_data, val_data = prepare_data_for_bert(texts, labels)
```

### 3. Model Training

Train the BERT model:

```python
from src.model import BERTSentimentModel, create_dataloaders

# Create data loaders
train_loader, val_loader = create_dataloaders(train_data, val_data, batch_size=16)

# Initialize and train model
model = BERTSentimentModel(num_labels=2)
history = model.train(train_loader, val_loader, epochs=3)

# Save model
model.save_model('models/sentiment_model')
```

### 4. Inference

Make predictions on new reviews:

```python
from src.inference import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor('models/sentiment_model')

# Predict single text
result = predictor.predict("This app is amazing!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")

# Predict multiple texts
texts = ["Great app!", "Terrible experience", "It's okay"]
results = predictor.predict(texts)
```

### Command Line Interface

The inference module can also be used from command line:

```bash
# Interactive mode
python src/inference.py --model models/sentiment_model --interactive

# Single prediction
python src/inference.py --model models/sentiment_model --text "This is great!"

# Batch prediction from CSV
python src/inference.py --model models/sentiment_model \
    --input data/test.csv --output data/predictions.csv
```

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Run tests with coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
```

View coverage report:

```bash
open htmlcov/index.html  # On macOS
# or
start htmlcov/index.html  # On Windows
```

Run specific test file:

```bash
pytest tests/unit/test_data_extraction.py -v
```

## 📊 Dataset

The dataset contains app reviews with the following columns:

- `reviewId`: Unique identifier for each review
- `userName`: Name of the reviewer
- `content`: Review text (main input)
- `score`: Rating from 1-5 stars (target variable)
- `at`: Timestamp of the review
- Other metadata fields

### Sentiment Mapping

**Binary Classification:**
- Negative (0): Scores 1-2
- Positive (1): Scores 4-5
- Neutral (3): Dropped

**Multi-class Classification:**
- Negative (0): Scores 1-2
- Neutral (1): Score 3
- Positive (2): Scores 4-5

## 🔄 Git Workflow

### Branching Strategy

- `main`: Production-ready code
- `feature/<feature-name>`: New features
- `bugfix/<bug-description>`: Bug fixes

### Example Workflow

1. Create a new branch:
```bash
git checkout -b feature/data-extraction
```

2. Make changes and commit:
```bash
git add .
git commit -m "feat: implement data loading with error handling"
```

3. Push branch:
```bash
git push origin feature/data-extraction
```

4. Create Pull Request on GitHub
5. Partner reviews and approves (Adelewa reviews Adeniwa's PR, and vice versa)
6. Merge to main

## 📋 Trello Board

Board Name: **"Sentiment Analysis Project - Adeniwa & Adelewa"**

Link: [Add your Trello board URL here]

Board structure:
- **To Do**: Tasks not yet started
- **In Progress**: Currently being worked on
- **In Review**: Awaiting code review
- **Done**: Completed and merged

## 🤝 Collaboration

### Division of Labor

**Adeniwa Olomola (Student 1):**
- Data Extraction module (lead)
- Text Preprocessing
- Unit tests for data modules
- Documentation

**Adelewa Olomola (Student 2):**
- Model Training module (lead)
- BERT Tokenization
- Inference module
- Integration tests

**Both:**
- Data Processing module
- Code reviews
- Final report

### Communication

- Daily standups on [Teams/Slack]
- Code reviews on GitHub
- Questions and discussions on [Platform]

## 📈 Model Performance

[Add your model's performance metrics here after training]

| Metric | Value |
|--------|-------|
| Accuracy | X.XX% |
| Precision | X.XX% |
| Recall | X.XX% |
| F1 Score | X.XX% |

## 🎯 Evaluation Criteria

### Project Grade Breakdown (20 points)

1. **Git & Branch Management (5 pts)**
   - Clear branching model
   - Meaningful commit messages
   - Well-managed pull requests

2. **Unit Testing & Coverage (5 pts)**
   - Test coverage >90%
   - Comprehensive test cases

3. **Trello Board & Workflow (5 pts)**
   - Complete board structure
   - Detailed cards
   - Active usage

4. **Code Review & Pull Requests (5 pts)**
   - Constructive reviews
   - Both team members participate

## 🚧 Challenges & Solutions

[Document challenges faced during development and how you solved them]

## 🔮 Future Improvements

- [ ] Implement model ensembling
- [ ] Add support for multilingual reviews
- [ ] Create web interface
- [ ] Add more preprocessing options
- [ ] Experiment with other transformer models (RoBERTa, DistilBERT)

## 📚 Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Sentiment Analysis Tutorial](https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert)


## 📧 Contact

- **Adeniwa Olomola**: [@niwaifiok-arch](https://github.com/niwaifiok-arch)
- **Adelewa Olomola**: [GitHub username]

---

**Last Updated**: November 6, 2025
**Project Repository**: https://github.com/niwaifiok-arch/sentiment-analysis-bert
