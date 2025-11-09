# Collaborative Sentiment Analysis Pipeline - Setup Guide

## Project Overview
Building a BERT-based sentiment analysis pipeline collaboratively with proper Git workflow, testing, and project management.

**Team Members:**
- **Adeniwa Olomola** ([@niwaifiok-arch](https://github.com/niwaifiok-arch)) - Student 1
- **Adelewa Olomola** - Student 2

**Project Repository:** https://github.com/niwaifiok-arch/sentiment-analysis-bert

## Quick Start Checklist

### 1. Initial Setup (Both Students)
- [ ] **Adeniwa**: Create GitHub repository
- [ ] **Adeniwa**: Add Adelewa as collaborator
- [ ] **Both**: Clone repository locally
- [ ] **Both**: Create Trello board: "Sentiment Analysis Project - Adeniwa & Adelewa"
- [ ] **Both**: Set up communication channel (Teams/Slack)
- [ ] **Both**: Install Python dependencies

### 2. Repository Structure
```
sentiment-analysis/
├── src/
│   ├── data_extraction.py      # Student 1 Lead
│   ├── data_processing.py      # Both Students
│   ├── model.py                # Student 2 Lead
│   └── inference.py            # Student 2 Lead
├── tests/
│   └── unit/
│       ├── test_data_extraction.py
│       ├── test_data_processing.py
│       ├── test_model.py
│       └── test_inference.py
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    └── dataset.csv
```

### 3. Git Workflow

#### Branch Naming Convention
- `feature/data-extraction` - Adeniwa
- `feature/data-processing` - Both
- `feature/model-training` - Adelewa
- `feature/inference` - Adelewa
- `bugfix/issue-description` - As needed

#### Commit Message Format
```
<type>: <short description>

<optional detailed description>

Examples:
- "feat: implement CSV data loader for sentiment data"
- "test: add unit tests for tokenization"
- "fix: handle missing values in data extraction"
- "docs: update README with setup instructions"
```

#### Pull Request Process
1. Create feature branch from `main`
2. Implement feature with tests
3. Commit with clear messages
4. Push branch to GitHub
5. Create Pull Request
6. Partner reviews and comments
7. Address feedback
8. Partner approves
9. Merge to main
10. Update Trello card to "Done"

### 4. Trello Board Setup

#### Lists
1. **To Do** - Unstarted tasks
2. **In Progress** - Current work (assign member)
3. **In Review** - Awaiting code review
4. **Done** - Completed & merged

#### Card Structure
- **Title**: Clear task name
- **Description**: Brief summary
- **Checklist**: Subtasks/acceptance criteria
- **Labels**: `data`, `model`, `testing`, `documentation`
- **Members**: Assign responsible person
- **Attachments**: Link to PR when ready

#### Example Cards for "To Do"
```
Card 1: Setup Project Repository
- [ ] Create GitHub repo
- [ ] Add collaborator
- [ ] Initialize with README
- [ ] Add .gitignore for Python
Labels: documentation

Card 2: Implement Data Extraction
- [ ] Create data_extraction.py
- [ ] Load CSV dataset
- [ ] Handle missing values
- [ ] Add error handling
- [ ] Write unit tests
Labels: data
Assigned: Adeniwa

Card 3: Data Preprocessing & Tokenization
- [ ] Clean text (lowercase, remove special chars)
- [ ] Implement BERT tokenization
- [ ] Split train/validation sets
- [ ] Write unit tests
Labels: data
Assigned: Both

Card 4: Model Training Pipeline
- [ ] Load pretrained BERT
- [ ] Implement training loop
- [ ] Add model evaluation
- [ ] Save trained model
- [ ] Write unit tests
Labels: model
Assigned: Adelewa

Card 5: Inference Script
- [ ] Load trained model
- [ ] Create prediction function
- [ ] Handle single/batch predictions
- [ ] Write unit tests
Labels: model
Assigned: Adelewa
```

### 5. Testing Strategy

#### Coverage Target: >90%

**Test Types:**
- Unit tests for each function
- Edge case handling
- Error condition testing
- Integration tests for pipeline

**Running Tests:**
```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### 6. Python Dependencies

Create `requirements.txt`:
```
pandas>=1.3.0
numpy>=1.21.0
torch>=1.9.0
transformers>=4.10.0
scikit-learn>=0.24.0
pytest>=6.2.0
pytest-cov>=2.12.0
```

Install:
```bash
pip install -r requirements.txt
```

### 7. Dataset Information

**File**: `dataset.csv`

**Columns:**
- `reviewId`: Unique identifier
- `userName`: Reviewer name
- `content`: Review text (input for sentiment analysis)
- `score`: Rating 1-5 (target variable)
- `thumbsUpCount`: Helpful votes
- `at`: Review timestamp
- Other metadata fields

**Sentiment Mapping:**
- Score 1-2: Negative (0)
- Score 3: Neutral (1) 
- Score 4-5: Positive (2)

Or binary classification:
- Score 1-2: Negative (0)
- Score 4-5: Positive (1)
- Drop score 3

### 8. Division of Labor Example

#### Phase 1: Data Extraction (Week 1)
- **Adeniwa Lead**: Implement data loading
- **Adelewa**: Review data structure, plan preprocessing

#### Phase 2: Data Processing (Week 1-2)
- **Adeniwa**: Text cleaning functions
- **Adelewa**: BERT tokenization
- **Both**: Write tests together

#### Phase 3: Model Training (Week 2)
- **Adelewa Lead**: Training pipeline
- **Adeniwa**: Evaluation metrics, test scripts

#### Phase 4: Inference (Week 2-3)
- **Adelewa Lead**: Inference script
- **Adeniwa**: Documentation, test cases

#### Phase 5: Final Review (Week 3)
- **Both**: Cross-review tests, finalize documentation, prepare report

### 9. Communication Best Practices

- Daily standups (5 min sync-up)
- Update Trello cards promptly
- Comment on PRs within 24 hours
- Use clear, descriptive commit messages
- Document decisions in comments
- Ask questions early, don't block progress

### 10. Deliverables Checklist

- [ ] Public GitHub repository URL
- [ ] Well-structured code with proper branching
- [ ] >90% test coverage
- [ ] Complete README.md with:
  - [ ] Setup instructions
  - [ ] Usage examples
  - [ ] Component descriptions
- [ ] Trello board screenshots
- [ ] GitHub workflow screenshots (commits, PRs, reviews)
- [ ] Project report including:
  - [ ] Student names
  - [ ] Approach overview
  - [ ] Division of labor
  - [ ] Challenges faced
  - [ ] Future improvements

## Evaluation Criteria (20 points total)

### C01: Git & Branch Management (5 points)
- Clear branching model (feature/bugfix)
- Meaningful commit messages
- Well-managed pull requests

### C02: Unit Testing & Coverage (5 points)
- Test coverage >90%
- Tests for main functions
- Edge cases covered

### C03: Trello Board & Workflow (5 points)
- Complete board structure
- Detailed cards with checklists
- Aligned with Git workflow
- Active use throughout project

### C04: Code Review & Pull Requests (5 points)
- Complete, constructive reviews
- Both students validate each other's work
- Comments and feedback on PRs

## Tips for Success

1. **Start Early**: Don't wait until the last minute
2. **Communicate Often**: Over-communication is better than under-communication
3. **Test as You Go**: Don't save testing for the end
4. **Document Everything**: Future you will thank present you
5. **Review Thoroughly**: Good code reviews improve code quality
6. **Ask for Help**: Reach out if stuck
7. **Keep Trello Updated**: It's part of your grade
8. **Use Meaningful Names**: For branches, commits, variables

## Resources

- [Sentiment Analysis using BERT (Kaggle)](https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert)
- [GitHub Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [GitHub Hello World Guide](https://docs.github.com/en/get-started/start-your-journey/hello-world)
- [Pull Request Reviews](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews)
- [Trello Guide](https://trello.com/guide/trello-101)

Good luck! 🎓
