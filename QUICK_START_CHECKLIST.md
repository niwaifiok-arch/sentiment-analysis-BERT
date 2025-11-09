# Quick Start Checklist - Week by Week

**Team Members:**
- **Adeniwa Olomola** ([@niwaifiok-arch](https://github.com/niwaifiok-arch)) - Student 1
- **Adelewa Olomola** - Student 2

**Project Repository:** https://github.com/niwaifiok-arch/sentiment-analysis-bert

---

## 🗓️ Week 1: Setup & Data Extraction

### Day 1-2: Project Initialization
- [ ] **Adeniwa**: Create GitHub repository at https://github.com/niwaifiok-arch/sentiment-analysis-bert
- [ ] **Adeniwa**: Add Adelewa as collaborator
- [ ] **Both**: Clone repository locally
- [ ] **Both**: Set up virtual environment
  ```bash
  python -m venv venv
  source venv/bin/activate  # Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```
- [ ] **Both**: Create Trello board "Sentiment Analysis Project - Adeniwa & Adelewa"
- [ ] **Both**: Set up Teams/Slack channel
- [ ] **Both**: Create initial project structure (directories)

### Day 3-4: Data Extraction (Adeniwa Lead)
- [ ] Create branch: `feature/data-extraction`
- [ ] Implement `load_dataset()` function
- [ ] Implement `extract_sentiment_data()` function
- [ ] Implement `convert_scores_to_sentiment()` function
- [ ] Add error handling for all functions
- [ ] Write unit tests in `test_data_extraction.py`
- [ ] Create Pull Request
- [ ] **Adelewa**: Review PR and provide feedback
- [ ] Address feedback and merge

### Day 5-7: Data Processing Setup
- [ ] **Both**: Create branch from main
- [ ] **Adeniwa**: Implement `TextPreprocessor` class
- [ ] **Adelewa**: Implement `BERTTokenizer` class
- [ ] Integrate both components
- [ ] Test integration
- [ ] Write unit tests

---

## 🗓️ Week 2: Model Training & Processing

### Day 8-10: Complete Data Processing
- [ ] Finish `prepare_data_for_bert()` function
- [ ] Test on real dataset
- [ ] Analyze text length distribution
- [ ] Choose optimal max_length
- [ ] Complete unit tests for data_processing
- [ ] Create PR and review
- [ ] Merge to main

### Day 11-14: Model Training (Adelewa Lead)
- [ ] Create branch: `feature/model-training`
- [ ] Implement `SentimentDataset` class
- [ ] Implement `BERTSentimentModel` class
- [ ] Create training loop
- [ ] Add validation evaluation
- [ ] Implement model saving/loading
- [ ] Write unit tests for model
- [ ] Train model on full dataset (may take 2-3 hours)
- [ ] Document training results
- [ ] Create PR
- [ ] **Adeniwa**: Review PR
- [ ] Merge to main

---

## 🗓️ Week 3: Inference & Finalization

### Day 15-17: Inference Module
- [ ] Create branch: `feature/inference`
- [ ] Implement `SentimentPredictor` class
- [ ] Add single text prediction
- [ ] Add batch prediction
- [ ] Implement CLI interface
- [ ] Write unit tests for inference
- [ ] Test with various inputs
- [ ] Create PR and review
- [ ] Merge to main

### Day 18-19: Integration & Testing
- [ ] Run full integration tests
- [ ] Verify test coverage >90%
  ```bash
  pytest --cov=src --cov-report=html
  ```
- [ ] Fix any failing tests
- [ ] Optimize code based on coverage report
- [ ] Update documentation

### Day 20-21: Documentation & Report
- [ ] Complete README.md
  - [ ] Add setup instructions
  - [ ] Add usage examples
  - [ ] Add performance metrics
  - [ ] Add screenshots
- [ ] Write project report
  - [ ] Executive summary
  - [ ] Technical details
  - [ ] Division of labor
  - [ ] Challenges & solutions
  - [ ] Results & analysis
- [ ] Take Trello screenshots
- [ ] Take GitHub screenshots (commits, PRs, network graph)
- [ ] Final review of all deliverables

---

## ✅ Final Checklist Before Submission

### Code Quality
- [ ] All functions have docstrings
- [ ] Code follows PEP 8 style guide
- [ ] No commented-out code or TODOs
- [ ] All print statements removed or converted to logging
- [ ] Error handling in all appropriate places

### Testing
- [ ] All tests passing
- [ ] Test coverage >90%
- [ ] Edge cases tested
- [ ] Integration tests completed

### Git & GitHub
- [ ] All branches merged to main
- [ ] No uncommitted changes
- [ ] Repository is public
- [ ] README is complete
- [ ] .gitignore is properly configured
- [ ] Clear commit history
- [ ] All PRs have reviews

### Trello
- [ ] Board is complete
- [ ] All cards in "Done" list
- [ ] Screenshots taken
- [ ] Board shows clear workflow

### Documentation
- [ ] README.md complete with all sections
- [ ] All modules have docstrings
- [ ] Usage examples provided
- [ ] Installation instructions clear
- [ ] Requirements.txt up to date

### Report
- [ ] Student names included
- [ ] All sections complete
- [ ] Trello screenshots included
- [ ] GitHub screenshots included
- [ ] Performance metrics documented
- [ ] Challenges discussed
- [ ] Future improvements listed

### Deliverables Package
- [ ] GitHub repository URL
- [ ] Trello board URL
- [ ] Project report (PDF or MD)
- [ ] Screenshots folder
- [ ] Test coverage report

---

## 🎯 Evaluation Self-Check

### C01: Git & Branch Management (5 points)
- [ ] Multiple branches used (not just main)
- [ ] Branch names follow convention (feature/*, bugfix/*)
- [ ] Meaningful commit messages
- [ ] Pull requests used for all merges
- [ ] No direct commits to main

**Expected Score: ___ / 5**

### C02: Unit Testing & Coverage (5 points)
- [ ] Unit tests for all modules
- [ ] Test coverage >90%
- [ ] Tests are comprehensive
- [ ] Edge cases covered
- [ ] All tests passing

**Expected Score: ___ / 5**

### C03: Trello Board & Workflow (5 points)
- [ ] Board has 4 lists (To Do, In Progress, In Review, Done)
- [ ] Cards are detailed (description, checklist, labels)
- [ ] PR links attached to cards
- [ ] Clear progression shown
- [ ] Active throughout project

**Expected Score: ___ / 5**

### C04: Code Review & PRs (5 points)
- [ ] All PRs have reviews
- [ ] Both students reviewed each other's work
- [ ] Comments are constructive
- [ ] Feedback was addressed
- [ ] No PRs merged without approval

**Expected Score: ___ / 5**

**Total Expected Score: ___ / 20**

---

## 📞 Emergency Contacts

**If stuck on:**
- Git issues: [Resource link or TA contact]
- BERT/Transformers: [Hugging Face docs]
- Testing: [Pytest docs]
- General Python: [Office hours]

---

## 💡 Pro Tips

1. **Commit Often**: Small, frequent commits are better than large, infrequent ones
2. **Test as You Go**: Don't save all testing for the end
3. **Communicate Daily**: Quick check-ins prevent blocking issues
4. **Review Promptly**: Don't let PRs sit for more than 24 hours
5. **Document Immediately**: Write docs as you code, not after
6. **Start Training Early**: Model training takes time, start by Week 2
7. **Save Intermediate Results**: Keep trained models, don't rely on one final training run
8. **Update Trello**: Move cards as you work, not all at once at the end

---

## 🆘 Common Pitfalls to Avoid

❌ **Don't:**
- Commit directly to main
- Merge without code review
- Skip writing tests
- Wait until last minute to integrate
- Forget to update Trello
- Use vague commit messages like "fix" or "update"
- Test only on a small subset of data
- Hardcode file paths

✅ **Do:**
- Use feature branches
- Review each other's PRs
- Write tests first (TDD)
- Integrate and test frequently
- Keep Trello current
- Write descriptive commits
- Test on full dataset
- Use relative paths or config files

---

Good luck! 🚀
