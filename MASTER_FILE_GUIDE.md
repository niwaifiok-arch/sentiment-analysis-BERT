# 📦 Project Starter Package - Complete File Guide

**Team Members:**
- **Adeniwa Olomola** ([@niwaifiok-arch](https://github.com/niwaifiok-arch)) - Student 1
- **Adelewa Olomola** - Student 2

**Project Repository:** https://github.com/niwaifiok-arch/sentiment-analysis-bert

---

## Overview

This package contains everything you need to start your Collaborative Sentiment Analysis Pipeline project. All template files include detailed TODO comments to guide your implementation.

---

## 📋 Core Module Files

### 1. **data_extraction.py** (Adeniwa Lead)
**Purpose**: Load and extract sentiment data from CSV file

**Key Functions**:
- `load_dataset()`: Load CSV with error handling
- `extract_sentiment_data()`: Extract texts and labels
- `convert_scores_to_sentiment()`: Map scores to sentiment labels
- `get_data_statistics()`: Calculate dataset statistics

**TODO**: Uncomment and implement all functions marked with TODO comments

**Tests**: test_data_extraction.py

---

### 2. **data_processing.py** (Both Students)
**Purpose**: Clean text and tokenize for BERT

**Key Classes**:
- `TextPreprocessor` (Adeniwa): Text cleaning operations
- `BERTTokenizer` (Adelewa): BERT tokenization

**Key Functions**:
- `prepare_data_for_bert()`: Complete preprocessing pipeline
- `analyze_text_lengths()`: Help choose optimal max_length

**TODO**: Implement both classes and integrate them

**Tests**: test_data_processing.py (create similar to test_data_extraction.py)

---

### 3. **model.py** (Adelewa Lead)
**Purpose**: BERT model training and evaluation

**Key Classes**:
- `SentimentDataset`: PyTorch Dataset wrapper
- `BERTSentimentModel`: BERT training interface

**Key Functions**:
- `train()`: Training loop with validation
- `evaluate()`: Calculate metrics
- `predict()`: Make predictions
- `save_model()` / `load_model()`: Persistence

**TODO**: Implement training loop, evaluation, and model management

**Tests**: test_model.py (create similar to test_data_extraction.py)

---

### 4. **inference.py** (Adelewa Lead)
**Purpose**: Interface for making predictions

**Key Classes**:
- `SentimentPredictor`: High-level prediction interface

**Key Functions**:
- `predict()`: Single/batch prediction
- `predict_from_file()`: Process CSV files
- `interactive_prediction()`: Interactive mode

**TODO**: Implement predictor and CLI

**Tests**: test_inference.py (create similar to test_data_extraction.py)

---

## 🧪 Test Files

### **test_data_extraction.py** (Adeniwa)
**Purpose**: Unit tests for data extraction module

**Test Classes**:
- `TestLoadDataset`: Tests for loading data
- `TestExtractSentimentData`: Tests for extraction
- `TestConvertScoresToSentiment`: Tests for conversion
- `TestGetDataStatistics`: Tests for statistics
- `TestIntegration`: End-to-end tests

**TODO**: Uncomment test methods and implement assertions

**Run**: `pytest tests/unit/test_data_extraction.py -v`

---

### **test_data_processing.py** (Both - Create New)
**Template**: Follow structure of test_data_extraction.py

**Suggested Tests**:
- Text cleaning functions
- Tokenization output shapes
- Batch processing
- Integration tests

---

### **test_model.py** (Adelewa - Create New)
**Suggested Tests**:
- Model initialization
- Training loop (on small sample)
- Evaluation metrics
- Prediction shapes
- Save/load functionality

---

### **test_inference.py** (Adelewa - Create New)
**Suggested Tests**:
- Single text prediction
- Batch prediction
- Confidence scores
- Error handling
- CLI functionality

---

## 📚 Documentation Files

### **README.md**
**Purpose**: Main project documentation

**Sections**:
- Project overview
- Installation instructions
- Usage examples
- Project structure
- Testing guide
- Team information

**TODO**: 
1. Add team member names
2. Fill in performance metrics after training
3. Add actual Trello/GitHub URLs
4. Update contact information

---

### **project_setup_guide.md**
**Purpose**: Detailed setup and workflow instructions

**Contents**:
- Step-by-step setup
- Git workflow examples
- Trello board structure
- Testing strategy
- Collaboration tips

**Use**: Reference throughout project development

---

### **project_report_template.md**
**Purpose**: Template for final project report

**Sections**:
- Executive summary
- Technical approach
- Division of labor
- Results & performance
- Challenges & solutions
- Future improvements

**TODO**: Fill in all [bracketed] sections with your actual data

**Due**: With final submission

---

### **QUICK_START_CHECKLIST.md**
**Purpose**: Week-by-week task breakdown

**Use**: 
- Track progress daily
- Ensure nothing is forgotten
- Self-evaluate before submission

**Tip**: Print this out or keep it open while working

---

## ⚙️ Configuration Files

### **requirements.txt**
**Purpose**: Python dependencies

**Contents**:
- pandas, numpy (data processing)
- torch, transformers (ML)
- pytest, pytest-cov (testing)
- Other utilities

**Install**: `pip install -r requirements.txt`

**Note**: May need to update versions based on your environment

---

### **.gitignore**
**Purpose**: Exclude files from Git

**Excludes**:
- Python cache files (__pycache__)
- Virtual environments (venv/)
- Model files (*.pt, *.pth)
- Large data files
- IDE files

**Important**: Add this to your repository first thing!

---

## 🗂️ Directory Structure to Create

```
sentiment-analysis-pipeline/
│
├── src/                          # Source code
│   ├── __init__.py              # Create empty file
│   ├── data_extraction.py       # ✓ Provided
│   ├── data_processing.py       # ✓ Provided
│   ├── model.py                 # ✓ Provided
│   └── inference.py             # ✓ Provided
│
├── tests/                        # Tests
│   ├── __init__.py              # Create empty file
│   └── unit/
│       ├── __init__.py          # Create empty file
│       ├── test_data_extraction.py    # ✓ Provided
│       ├── test_data_processing.py    # TODO: Create
│       ├── test_model.py              # TODO: Create
│       └── test_inference.py          # TODO: Create
│
├── data/                         # Data files
│   └── dataset.csv              # Your dataset
│
├── models/                       # Saved models
│   └── sentiment_model/         # Created after training
│
├── docs/                         # Additional documentation
│   └── screenshots/             # Trello & GitHub screenshots
│
├── requirements.txt              # ✓ Provided
├── .gitignore                   # ✓ Provided
├── README.md                    # ✓ Provided
└── project_report.md            # Copy from template

```

---

## 🚀 Getting Started Steps

### Step 1: Repository Setup (Day 1)
1. Adeniwa creates GitHub repo at: https://github.com/niwaifiok-arch/sentiment-analysis-bert
2. Add README.md, .gitignore, requirements.txt
3. Add Adelewa as collaborator
4. Both clone repository

### Step 2: Initial Structure (Day 1)
```bash
cd sentiment-analysis-pipeline

# Create directories
mkdir -p src tests/unit data models docs/screenshots

# Create __init__.py files
touch src/__init__.py tests/__init__.py tests/unit/__init__.py

# Copy provided files to appropriate locations
cp data_extraction.py src/
cp data_processing.py src/
cp model.py src/
cp inference.py src/
cp test_data_extraction.py tests/unit/

# Add your dataset
cp /path/to/dataset.csv data/
```

### Step 3: Virtual Environment (Day 1)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4: Test Setup (Day 1)
```bash
# Verify tests can run (they'll fail/skip initially - that's OK!)
pytest tests/unit/test_data_extraction.py -v
```

### Step 5: Trello Board (Day 1)
1. Create board: "Sentiment Analysis Project - [Names]"
2. Add lists: To Do, In Progress, In Review, Done
3. Create initial cards from checklist

### Step 6: Start Coding! (Day 2+)
1. Create feature branch
2. Implement one function at a time
3. Write test for each function
4. Commit frequently
5. Create PR when complete
6. Partner reviews
7. Merge and repeat!

---

## 📝 Key Implementation Notes

### Data Extraction Module
- Start with `load_dataset()` - it's the foundation
- Test with a small sample CSV first
- Handle all error cases (missing file, wrong format, etc.)
- Calculate statistics to understand your data

### Data Processing Module
- Test text cleaning on a few examples first
- Check what BERT tokenizer returns (use print statements)
- Experiment with max_length (start with 128)
- Verify token IDs are in valid range

### Model Training Module
- Start with 1 epoch on small dataset to verify it works
- Print shapes of tensors to debug issues
- Training will take time (2-3 hours for full dataset)
- Save model after each epoch in case of crashes

### Inference Module
- Test with simple examples first
- Build CLI last (after core functionality works)
- Add confidence scores for better insights

---

## 🔍 Testing Tips

### Writing Good Tests
```python
def test_function_name():
    """Test should explain what it's testing."""
    # Arrange - set up test data
    input_data = ...
    
    # Act - call the function
    result = function_to_test(input_data)
    
    # Assert - verify the result
    assert result == expected_result
```

### Common Test Patterns
- Test normal cases
- Test edge cases (empty input, very long text)
- Test error cases (invalid input)
- Test boundary conditions (exactly at limits)

### Coverage Goals
- Aim for >95% (not just >90%)
- 100% is ideal but not always practical
- Focus on critical path coverage first

---

## 🐛 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**: Add to top of test files:
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
```

### Issue: CUDA out of memory during training
**Solution**: 
- Reduce batch_size (try 8 or 4)
- Use CPU instead: `device='cpu'`
- Close other applications

### Issue: Tests taking too long
**Solution**:
- Use pytest markers for slow tests
- Run quick tests frequently, slow tests before commits

### Issue: Merge conflicts
**Solution**:
- Pull main frequently: `git pull origin main`
- Communicate about which files you're working on
- Use `git mergetool` if needed

---

## 📊 Expected Timeline

| Week | Task | Time Investment |
|------|------|----------------|
| 1 | Setup + Data Extraction | 10-15 hours |
| 2 | Data Processing + Model Training | 15-20 hours |
| 3 | Inference + Testing + Documentation | 10-15 hours |
| **Total** | | **35-50 hours** |

---

## ✅ Pre-Submission Checklist

Print this and check off before submitting:

**Code**
- [ ] All TODOs removed or implemented
- [ ] All functions have docstrings
- [ ] No print statements in production code
- [ ] Code follows PEP 8 style

**Tests**
- [ ] All tests passing
- [ ] Coverage >90% verified
- [ ] Edge cases tested

**Git**
- [ ] All branches merged
- [ ] Repository is public
- [ ] Clear commit history
- [ ] README complete

**Trello**
- [ ] All cards in Done
- [ ] Screenshots taken
- [ ] Shows clear workflow

**Documentation**
- [ ] README updated
- [ ] Report complete
- [ ] Screenshots included
- [ ] Names added everywhere

---

## 📧 Support

If you have questions:
1. Check the documentation first
2. Review the TODO comments in code
3. Consult the resources in README
4. Ask your partner
5. Check with instructor/TA

---

## 🎉 You're Ready!

You now have everything needed to build an excellent sentiment analysis pipeline. Remember:

- **Communicate** with your partner
- **Commit** frequently
- **Test** as you go
- **Document** everything
- **Ask** for help when stuck

Good luck with your project! 🚀

---

**Package Contents**: 9 files
**Last Updated**: [Date]
**Version**: 1.0
