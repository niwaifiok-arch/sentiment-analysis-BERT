"""
Data Processing Module
Both Students - Collaborative Responsibility
- Adeniwa Olomola: TextPreprocessor class
- Adelewa Olomola: BERTTokenizer class

This module handles text preprocessing, cleaning, and tokenization for BERT.
"""

import re
import pandas as pd
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer


class TextPreprocessor:
    """
    Handles text cleaning and preprocessing.
    
    Adeniwa can focus on this class.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> cleaned = preprocessor.clean_text("GREAT APP!!! 😊")
            >>> print(cleaned)
            great app
        """
        # TODO: Handle None/NaN values
        # if pd.isna(text) or not isinstance(text, str):
        #     return ""
        
        # TODO: Convert to lowercase
        # text = text.lower()
        
        # TODO: Remove URLs
        # text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # TODO: Remove email addresses
        # text = re.sub(r'\S+@\S+', '', text)
        
        # TODO: Remove HTML tags if any
        # text = re.sub(r'<.*?>', '', text)
        
        # TODO: Remove emojis/special characters (keep basic punctuation)
        # text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # TODO: Remove extra whitespace
        # text = re.sub(r'\s+', ' ', text).strip()
        
        # TODO: Return cleaned text
        pass
    
    def clean_texts(self, texts: List[str]) -> List[str]:
        """
        Clean a list of texts.
        
        Args:
            texts (List[str]): List of raw texts
            
        Returns:
            List[str]: List of cleaned texts
        """
        # TODO: Apply clean_text to each text in the list
        # return [self.clean_text(text) for text in texts]
        pass


class BERTTokenizer:
    """
    Handles BERT tokenization.
    
    Adelewa can focus on this class.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        """
        Initialize the BERT tokenizer.
        
        Args:
            model_name (str): Name of the pretrained BERT model
            max_length (int): Maximum sequence length for tokenization
        """
        # TODO: Load the tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.max_length = max_length
        pass
    
    def tokenize(self, texts: List[str], 
                 labels: List[int] = None) -> Dict[str, Any]:
        """
        Tokenize texts for BERT input.
        
        Args:
            texts (List[str]): List of texts to tokenize
            labels (List[int], optional): Corresponding labels
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - input_ids: Token IDs
                - attention_mask: Attention masks
                - labels: Labels (if provided)
                
        Example:
            >>> tokenizer = BERTTokenizer()
            >>> texts = ["great app", "terrible service"]
            >>> labels = [1, 0]
            >>> encoded = tokenizer.tokenize(texts, labels)
            >>> print(encoded.keys())
            dict_keys(['input_ids', 'attention_mask', 'labels'])
        """
        # TODO: Tokenize the texts
        # encoding = self.tokenizer(
        #     texts,
        #     padding='max_length',
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors='pt'
        # )
        
        # TODO: Add labels if provided
        # if labels is not None:
        #     encoding['labels'] = torch.tensor(labels)
        
        # TODO: Return the encoding
        pass
    
    def tokenize_batch(self, texts: List[str], 
                      labels: List[int] = None,
                      batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Tokenize texts in batches (for large datasets).
        
        Args:
            texts (List[str]): List of texts to tokenize
            labels (List[int], optional): Corresponding labels
            batch_size (int): Size of each batch
            
        Returns:
            List[Dict[str, Any]]: List of encoded batches
        """
        # TODO: Split into batches and tokenize each
        pass


def prepare_data_for_bert(texts: pd.Series, 
                          labels: pd.Series,
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[Dict, Dict]:
    """
    Complete preprocessing pipeline: clean, tokenize, and split data.
    
    Both students should collaborate on this function.
    
    Args:
        texts (pd.Series): Raw text data
        labels (pd.Series): Sentiment labels
        test_size (float): Proportion of data for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[Dict, Dict]: (train_data, val_data) where each dict contains
            input_ids, attention_mask, and labels
            
    Example:
        >>> train_data, val_data = prepare_data_for_bert(texts, labels)
        >>> print(f"Train size: {len(train_data['labels'])}")
        >>> print(f"Val size: {len(val_data['labels'])}")
    """
    # TODO: Import train_test_split
    # from sklearn.model_selection import train_test_split
    
    # TODO: Split data into train and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(
    #     texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    # )
    
    # TODO: Initialize preprocessor and clean texts
    # preprocessor = TextPreprocessor()
    # X_train_clean = preprocessor.clean_texts(X_train.tolist())
    # X_val_clean = preprocessor.clean_texts(X_val.tolist())
    
    # TODO: Initialize tokenizer and tokenize
    # tokenizer = BERTTokenizer()
    # train_data = tokenizer.tokenize(X_train_clean, y_train.tolist())
    # val_data = tokenizer.tokenize(X_val_clean, y_val.tolist())
    
    # TODO: Return train and validation data
    pass


def analyze_text_lengths(texts: List[str]) -> Dict[str, float]:
    """
    Analyze the distribution of text lengths to help choose max_length.
    
    Args:
        texts (List[str]): List of texts to analyze
        
    Returns:
        Dict[str, float]: Statistics including mean, median, max, percentiles
    """
    # TODO: Calculate word lengths
    # lengths = [len(text.split()) for text in texts]
    
    # TODO: Calculate statistics
    # import numpy as np
    # stats = {
    #     'mean': np.mean(lengths),
    #     'median': np.median(lengths),
    #     'max': np.max(lengths),
    #     'min': np.min(lengths),
    #     'p90': np.percentile(lengths, 90),
    #     'p95': np.percentile(lengths, 95),
    #     'p99': np.percentile(lengths, 99),
    # }
    
    # TODO: Return statistics
    pass


def main():
    """
    Example usage of the data processing module.
    """
    # TODO: Sample data for testing
    # sample_texts = [
    #     "This app is AMAZING!!! 😊😊😊",
    #     "Worst app ever. Doesn't work.",
    #     "It's okay, nothing special.",
    #     "LOVE IT! Best purchase ever!",
    #     "Terrible experience. Would not recommend."
    # ]
    # sample_labels = [1, 0, 1, 1, 0]
    
    # TODO: Test text cleaning
    # print("Testing Text Cleaning:")
    # preprocessor = TextPreprocessor()
    # for text in sample_texts:
    #     cleaned = preprocessor.clean_text(text)
    #     print(f"Original: {text}")
    #     print(f"Cleaned:  {cleaned}\n")
    
    # TODO: Test tokenization
    # print("\nTesting Tokenization:")
    # tokenizer = BERTTokenizer(max_length=32)
    # encoded = tokenizer.tokenize(
    #     preprocessor.clean_texts(sample_texts),
    #     sample_labels
    # )
    # print(f"Input IDs shape: {encoded['input_ids'].shape}")
    # print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    # print(f"Labels: {encoded['labels']}")


if __name__ == "__main__":
    main()
