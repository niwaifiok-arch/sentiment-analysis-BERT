"""
Data Extraction Module
Adeniwa Olomola (Student 1) - Primary Responsibility

This module handles loading and initial validation of the sentiment analysis dataset.
"""

import pandas as pd
import os
from typing import Optional, Tuple


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the sentiment analysis dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset
        
    Returns:
        pd.DataFrame: Loaded dataset with all columns
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or has invalid format
        pd.errors.EmptyDataError: If CSV is empty
    
    Example:
        >>> df = load_dataset('data/dataset.csv')
        >>> print(df.shape)
        (14076, 12)
    """
    # TODO: Implement error handling for missing file
    # if not os.path.exists(file_path):
    #     raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # TODO: Load CSV with pandas
    # df = pd.read_csv(file_path)
    
    # TODO: Validate that required columns exist
    # required_columns = ['content', 'score']
    # missing_cols = [col for col in required_columns if col not in df.columns]
    # if missing_cols:
    #     raise ValueError(f"Missing required columns: {missing_cols}")
    
    # TODO: Check for empty dataframe
    # if df.empty:
    #     raise ValueError("Dataset is empty")
    
    # TODO: Return the loaded dataframe
    pass


def extract_sentiment_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Extract text content and sentiment labels from the dataframe.
    
    Args:
        df (pd.DataFrame): The full dataset
        
    Returns:
        Tuple[pd.Series, pd.Series]: (texts, labels) where texts contains
            review content and labels contains sentiment scores
            
    Raises:
        KeyError: If required columns are missing
        ValueError: If data is invalid
    
    Example:
        >>> texts, labels = extract_sentiment_data(df)
        >>> print(len(texts), len(labels))
        14076 14076
    """
    # TODO: Extract 'content' column as texts
    # texts = df['content']
    
    # TODO: Extract 'score' column as labels
    # labels = df['score']
    
    # TODO: Remove any rows with missing values
    # mask = texts.notna() & labels.notna()
    # texts = texts[mask]
    # labels = labels[mask]
    
    # TODO: Return texts and labels
    pass


def convert_scores_to_sentiment(scores: pd.Series, 
                                binary: bool = True) -> pd.Series:
    """
    Convert numeric scores (1-5) to sentiment labels.
    
    Args:
        scores (pd.Series): Series of numeric scores from 1-5
        binary (bool): If True, use binary classification (negative/positive)
                      If False, use 3-class (negative/neutral/positive)
                      
    Returns:
        pd.Series: Sentiment labels as integers
            Binary: 0=negative (scores 1-2), 1=positive (scores 4-5)
            3-class: 0=negative (1-2), 1=neutral (3), 2=positive (4-5)
            
    Example:
        >>> scores = pd.Series([1, 2, 3, 4, 5])
        >>> sentiment = convert_scores_to_sentiment(scores, binary=True)
        >>> print(sentiment.tolist())
        [0, 0, 1, 1]  # Note: score 3 is dropped in binary mode
    """
    # TODO: Implement conversion logic
    # if binary:
    #     # Map 1-2 to 0 (negative), 4-5 to 1 (positive), drop 3
    #     mapping = {1: 0, 2: 0, 3: None, 4: 1, 5: 1}
    #     sentiment = scores.map(mapping)
    #     # Remove None values (score 3)
    #     sentiment = sentiment.dropna()
    # else:
    #     # Map 1-2 to 0, 3 to 1, 4-5 to 2
    #     mapping = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    #     sentiment = scores.map(mapping)
    
    # TODO: Return sentiment labels
    pass


def get_data_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics about the dataset.
    
    Args:
        df (pd.DataFrame): The dataset
        
    Returns:
        dict: Dictionary containing statistics like:
            - total_reviews: Total number of reviews
            - score_distribution: Count of each score (1-5)
            - avg_review_length: Average number of words per review
            - missing_content: Number of reviews with missing content
            
    Example:
        >>> stats = get_data_statistics(df)
        >>> print(stats['total_reviews'])
        14076
    """
    # TODO: Calculate statistics
    # stats = {
    #     'total_reviews': len(df),
    #     'score_distribution': df['score'].value_counts().to_dict(),
    #     'missing_content': df['content'].isna().sum(),
    #     'missing_score': df['score'].isna().sum(),
    # }
    
    # if 'content' in df.columns:
    #     # Calculate average review length
    #     df['word_count'] = df['content'].fillna('').str.split().str.len()
    #     stats['avg_review_length'] = df['word_count'].mean()
    
    # TODO: Return statistics dictionary
    pass


def main():
    """
    Example usage of the data extraction module.
    """
    # TODO: Load the dataset
    # df = load_dataset('data/dataset.csv')
    
    # TODO: Print basic statistics
    # stats = get_data_statistics(df)
    # print("Dataset Statistics:")
    # print(f"Total reviews: {stats['total_reviews']}")
    # print(f"Score distribution: {stats['score_distribution']}")
    
    # TODO: Extract texts and labels
    # texts, labels = extract_sentiment_data(df)
    
    # TODO: Convert to sentiment labels
    # sentiment = convert_scores_to_sentiment(labels, binary=True)
    
    # TODO: Print sample
    # print("\nSample reviews:")
    # for i in range(3):
    #     print(f"Review: {texts.iloc[i][:100]}...")
    #     print(f"Sentiment: {sentiment.iloc[i]}\n")


if __name__ == "__main__":
    main()
