"""
Unit Tests for Data Extraction Module
Adeniwa Olomola (Student 1) - Primary Responsibility

Tests for loading, extracting, and validating sentiment data.
"""

import pytest
import pandas as pd
import os
import sys

# TODO: Add the src directory to the path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# TODO: Import functions to test
# from src.data_extraction import (
#     load_dataset,
#     extract_sentiment_data,
#     convert_scores_to_sentiment,
#     get_data_statistics
# )


class TestLoadDataset:
    """Tests for load_dataset function."""
    
    def test_load_valid_dataset(self):
        """Test loading a valid CSV file."""
        # TODO: Implement test
        # df = load_dataset('data/dataset.csv')
        # assert isinstance(df, pd.DataFrame)
        # assert not df.empty
        # assert 'content' in df.columns
        # assert 'score' in df.columns
        pass
    
    def test_load_nonexistent_file(self):
        """Test that FileNotFoundError is raised for missing file."""
        # TODO: Implement test
        # with pytest.raises(FileNotFoundError):
        #     load_dataset('nonexistent_file.csv')
        pass
    
    def test_dataset_has_required_columns(self):
        """Test that dataset has all required columns."""
        # TODO: Implement test
        # df = load_dataset('data/dataset.csv')
        # required_columns = ['content', 'score']
        # for col in required_columns:
        #     assert col in df.columns, f"Missing required column: {col}"
        pass
    
    def test_dataset_not_empty(self):
        """Test that loaded dataset is not empty."""
        # TODO: Implement test
        # df = load_dataset('data/dataset.csv')
        # assert len(df) > 0, "Dataset should not be empty"
        pass


class TestExtractSentimentData:
    """Tests for extract_sentiment_data function."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        # TODO: Create sample data
        # return pd.DataFrame({
        #     'content': ['Great app!', 'Terrible', 'Okay', None, 'Love it'],
        #     'score': [5, 1, 3, 4, 5],
        #     'other_column': ['a', 'b', 'c', 'd', 'e']
        # })
        pass
    
    def test_extract_returns_two_series(self, sample_df):
        """Test that function returns two Series."""
        # TODO: Implement test
        # texts, labels = extract_sentiment_data(sample_df)
        # assert isinstance(texts, pd.Series)
        # assert isinstance(labels, pd.Series)
        pass
    
    def test_extract_same_length(self, sample_df):
        """Test that texts and labels have the same length."""
        # TODO: Implement test
        # texts, labels = extract_sentiment_data(sample_df)
        # assert len(texts) == len(labels)
        pass
    
    def test_extract_removes_missing_values(self, sample_df):
        """Test that rows with missing values are removed."""
        # TODO: Implement test
        # texts, labels = extract_sentiment_data(sample_df)
        # assert texts.notna().all(), "Texts should not contain NaN"
        # assert labels.notna().all(), "Labels should not contain NaN"
        # # Sample has 1 NaN in content, so should have 4 valid rows
        # assert len(texts) == 4
        pass
    
    def test_extract_correct_columns(self, sample_df):
        """Test that correct columns are extracted."""
        # TODO: Implement test
        # texts, labels = extract_sentiment_data(sample_df)
        # assert texts.name == 'content'
        # assert labels.name == 'score'
        pass


class TestConvertScoresToSentiment:
    """Tests for convert_scores_to_sentiment function."""
    
    @pytest.fixture
    def sample_scores(self):
        """Create sample scores for testing."""
        # TODO: Create sample data
        # return pd.Series([1, 2, 3, 4, 5, 1, 5])
        pass
    
    def test_binary_classification(self, sample_scores):
        """Test binary sentiment classification."""
        # TODO: Implement test
        # sentiment = convert_scores_to_sentiment(sample_scores, binary=True)
        # # Scores 1,2 -> 0 (negative), scores 4,5 -> 1 (positive), score 3 dropped
        # assert all(sentiment.isin([0, 1]))
        # assert len(sentiment) == 6  # One score (3) should be dropped
        pass
    
    def test_multiclass_classification(self, sample_scores):
        """Test 3-class sentiment classification."""
        # TODO: Implement test
        # sentiment = convert_scores_to_sentiment(sample_scores, binary=False)
        # # Scores 1,2 -> 0, score 3 -> 1, scores 4,5 -> 2
        # assert all(sentiment.isin([0, 1, 2]))
        # assert len(sentiment) == 7  # No scores dropped
        pass
    
    def test_binary_negative_mapping(self, sample_scores):
        """Test that scores 1-2 map to 0 (negative)."""
        # TODO: Implement test
        # sentiment = convert_scores_to_sentiment(sample_scores, binary=True)
        # # Find indices of scores 1 and 2 in original series
        # score_1_2_mask = sample_scores.isin([1, 2])
        # # These should map to 0
        # # (Need to align indices after dropping score 3)
        pass
    
    def test_binary_positive_mapping(self, sample_scores):
        """Test that scores 4-5 map to 1 (positive)."""
        # TODO: Implement test
        pass


class TestGetDataStatistics:
    """Tests for get_data_statistics function."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        # TODO: Create sample data
        # return pd.DataFrame({
        #     'content': ['Good app', 'Bad app', None, 'Okay'],
        #     'score': [5, 1, 3, 3]
        # })
        pass
    
    def test_statistics_returns_dict(self, sample_df):
        """Test that function returns a dictionary."""
        # TODO: Implement test
        # stats = get_data_statistics(sample_df)
        # assert isinstance(stats, dict)
        pass
    
    def test_statistics_has_required_keys(self, sample_df):
        """Test that statistics dict has all required keys."""
        # TODO: Implement test
        # stats = get_data_statistics(sample_df)
        # required_keys = ['total_reviews', 'score_distribution', 
        #                  'missing_content', 'missing_score']
        # for key in required_keys:
        #     assert key in stats, f"Missing key: {key}"
        pass
    
    def test_total_reviews_correct(self, sample_df):
        """Test that total_reviews count is correct."""
        # TODO: Implement test
        # stats = get_data_statistics(sample_df)
        # assert stats['total_reviews'] == len(sample_df)
        pass
    
    def test_score_distribution(self, sample_df):
        """Test score distribution calculation."""
        # TODO: Implement test
        # stats = get_data_statistics(sample_df)
        # assert stats['score_distribution'][1] == 1  # One score of 1
        # assert stats['score_distribution'][3] == 2  # Two scores of 3
        # assert stats['score_distribution'][5] == 1  # One score of 5
        pass
    
    def test_missing_values_count(self, sample_df):
        """Test that missing values are counted correctly."""
        # TODO: Implement test
        # stats = get_data_statistics(sample_df)
        # assert stats['missing_content'] == 1  # One None in content
        # assert stats['missing_score'] == 0    # No None in score
        pass


# TODO: Add integration tests
class TestIntegration:
    """Integration tests for the complete data extraction pipeline."""
    
    def test_full_pipeline(self):
        """Test complete data extraction pipeline."""
        # TODO: Implement integration test
        # # Load data
        # df = load_dataset('data/dataset.csv')
        # 
        # # Extract data
        # texts, labels = extract_sentiment_data(df)
        # 
        # # Convert to sentiment
        # sentiment = convert_scores_to_sentiment(labels, binary=True)
        # 
        # # Verify results
        # assert len(sentiment) > 0
        # assert all(sentiment.isin([0, 1]))
        pass


# Run tests with: pytest tests/unit/test_data_extraction.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
