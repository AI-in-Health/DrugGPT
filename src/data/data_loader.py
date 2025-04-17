import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging
import torch
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_LOADER_CONFIGS = {
    'batch_size': 1,
    'max_length': 512,
    'val_split': 0.2,
    'shuffle': True,
    'random_state': 42
}


class QADataset(Dataset):
    """Custom Dataset for handling QA data."""

    def __init__(self, tokenizer, data_path, config):
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = config.get('max_length', 512)

    def __len__(self):
        return len(self.data)

    def tokenize_text(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        ).input_ids.squeeze(0)

    def __getitem__(self, idx):
        question = self.data.iloc[idx]['question']
        answer = self.data.iloc[idx]['answer']

        question_tokens = self.tokenize_text(question)
        answer_tokens = self.tokenize_text(answer)

        return question_tokens, answer_tokens


def get_data_loaders(tokenizer, data_path, config=None):
    """Create data loaders from the dataset."""

    if config is None:
        config = DATA_LOADER_CONFIGS

    logger.info("Creating Data Loaders...")
    dataset = QADataset(tokenizer, data_path, config)
    train_data, val_data = train_test_split(
        dataset,
        test_size=config['val_split'],
        shuffle=config['shuffle'],
        random_state=config['random_state']
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=config['shuffle']
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config['batch_size'],
        shuffle=False
    )

    return train_loader, val_loader


class SoftPromptDataset(Dataset):
    """Dataset for soft prompt tuning"""
    
    def __init__(self, data_path, tokenizer=None, max_length=512, max_samples=None):
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        
        
        if max_samples and max_samples < len(self.data):
            self.data = self.data.sample(max_samples, random_state=42)
            
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        question = self.data.iloc[idx].get('question', '')
        answer = self.data.iloc[idx].get('answer', '')
        
        
        text = f"Question: {question}\nAnswer: {answer}"
        
        
        if self.tokenizer:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                'input_ids': tokens.input_ids[0],
                'labels': tokens.input_ids[0].clone(),
                'prompt': question
            }
        else:
            
            return {
                'text': text,
                'prompt': question
            }


def load_training_data(train_file, val_file=None, batch_size=4, max_samples=None, tokenizer=None):
    """
    Load data specifically for soft prompt tuning.
    
    Args:
        train_file (str): Path to training data CSV
        val_file (str): Path to validation data CSV (optional)
        batch_size (int): Batch size for DataLoader
        max_samples (int): Maximum number of samples to use per dataset
        tokenizer: Optional tokenizer for processing text
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    logger.info(f"Loading training data from {train_file}")
    train_dataset = SoftPromptDataset(train_file, tokenizer, max_samples=max_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_file:
        logger.info(f"Loading validation data from {val_file}")
        val_dataset = SoftPromptDataset(val_file, tokenizer, max_samples=max_samples//2 if max_samples else None)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
