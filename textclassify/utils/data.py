"""Data handling utilities for text classification."""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import Counter, defaultdict

from ..core.types import TrainingData, ClassificationType


class DataLoader:
    """Utility class for loading and handling text classification data."""
    
    @staticmethod
    def from_csv(
        file_path: str,
        text_column: str = 'text',
        label_column: str = 'label',
        classification_type: ClassificationType = ClassificationType.MULTI_CLASS,
        delimiter: str = ',',
        encoding: str = 'utf-8'
    ) -> TrainingData:
        """Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            classification_type: Type of classification
            delimiter: CSV delimiter
            encoding: File encoding
            
        Returns:
            TrainingData instance
        """
        texts = []
        labels = []
        
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                if text_column not in row or label_column not in row:
                    continue
                
                text = row[text_column].strip()
                if not text:
                    continue
                
                if classification_type == ClassificationType.MULTI_CLASS:
                    label = row[label_column].strip()
                    labels.append(label)
                else:
                    # Multi-label: assume labels are separated by semicolons or commas
                    label_str = row[label_column].strip()
                    if ';' in label_str:
                        label_list = [l.strip() for l in label_str.split(';') if l.strip()]
                    else:
                        label_list = [l.strip() for l in label_str.split(',') if l.strip()]
                    labels.append(label_list)
                
                texts.append(text)
        
        return TrainingData(
            texts=texts,
            labels=labels,
            classification_type=classification_type
        )
    
    @staticmethod
    def from_json(
        file_path: str,
        text_field: str = 'text',
        label_field: str = 'label',
        classification_type: ClassificationType = ClassificationType.MULTI_CLASS,
        encoding: str = 'utf-8'
    ) -> TrainingData:
        """Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            text_field: Name of text field
            label_field: Name of label field
            classification_type: Type of classification
            encoding: File encoding
            
        Returns:
            TrainingData instance
        """
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        
        texts = []
        labels = []
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            if text_field not in item or label_field not in item:
                continue
            
            text = str(item[text_field]).strip()
            if not text:
                continue
            
            if classification_type == ClassificationType.MULTI_CLASS:
                label = str(item[label_field]).strip()
                labels.append(label)
            else:
                # Multi-label: expect list or string
                label_data = item[label_field]
                if isinstance(label_data, list):
                    label_list = [str(l).strip() for l in label_data if str(l).strip()]
                else:
                    label_str = str(label_data).strip()
                    if ';' in label_str:
                        label_list = [l.strip() for l in label_str.split(';') if l.strip()]
                    else:
                        label_list = [l.strip() for l in label_str.split(',') if l.strip()]
                labels.append(label_list)
            
            texts.append(text)
        
        return TrainingData(
            texts=texts,
            labels=labels,
            classification_type=classification_type
        )
    
    @staticmethod
    def from_lists(
        texts: List[str],
        labels: Union[List[str], List[List[str]]],
        classification_type: ClassificationType
    ) -> TrainingData:
        """Create TrainingData from lists.
        
        Args:
            texts: List of texts
            labels: List of labels
            classification_type: Type of classification
            
        Returns:
            TrainingData instance
        """
        return TrainingData(
            texts=texts,
            labels=labels,
            classification_type=classification_type
        )
    
    @staticmethod
    def save_to_csv(
        training_data: TrainingData,
        file_path: str,
        text_column: str = 'text',
        label_column: str = 'label',
        delimiter: str = ',',
        encoding: str = 'utf-8'
    ) -> None:
        """Save training data to CSV file.
        
        Args:
            training_data: Training data to save
            file_path: Path to save CSV file
            text_column: Name of text column
            label_column: Name of label column
            delimiter: CSV delimiter
            encoding: File encoding
        """
        with open(file_path, 'w', newline='', encoding=encoding) as f:
            writer = csv.writer(f, delimiter=delimiter)
            
            # Write header
            writer.writerow([text_column, label_column])
            
            # Write data
            for text, label in zip(training_data.texts, training_data.labels):
                if training_data.classification_type == ClassificationType.MULTI_CLASS:
                    writer.writerow([text, label])
                else:
                    # Multi-label: join with semicolons
                    label_str = ';'.join(label) if isinstance(label, list) else str(label)
                    writer.writerow([text, label_str])
    
    @staticmethod
    def save_to_json(
        training_data: TrainingData,
        file_path: str,
        text_field: str = 'text',
        label_field: str = 'label',
        encoding: str = 'utf-8'
    ) -> None:
        """Save training data to JSON file.
        
        Args:
            training_data: Training data to save
            file_path: Path to save JSON file
            text_field: Name of text field
            label_field: Name of label field
            encoding: File encoding
        """
        data = []
        
        for text, label in zip(training_data.texts, training_data.labels):
            item = {
                text_field: text,
                label_field: label
            }
            data.append(item)
        
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def split_data(
    training_data: TrainingData,
    train_ratio: float = 0.8,
    random_seed: Optional[int] = None,
    stratify: bool = True
) -> Tuple[TrainingData, TrainingData]:
    """Split training data into train and validation sets.
    
    Args:
        training_data: Original training data
        train_ratio: Ratio of data to use for training
        random_seed: Random seed for reproducibility
        stratify: Whether to stratify split by labels
        
    Returns:
        Tuple of (train_data, val_data)
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    indices = list(range(len(training_data.texts)))
    
    if stratify and training_data.classification_type == ClassificationType.MULTI_CLASS:
        # Stratified split for multi-class
        label_indices = defaultdict(list)
        for i, label in enumerate(training_data.labels):
            label_indices[label].append(i)
        
        train_indices = []
        val_indices = []
        
        for label, label_idx_list in label_indices.items():
            random.shuffle(label_idx_list)
            split_point = int(len(label_idx_list) * train_ratio)
            train_indices.extend(label_idx_list[:split_point])
            val_indices.extend(label_idx_list[split_point:])
    
    else:
        # Random split
        random.shuffle(indices)
        split_point = int(len(indices) * train_ratio)
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
    
    # Create training data splits
    train_texts = [training_data.texts[i] for i in train_indices]
    train_labels = [training_data.labels[i] for i in train_indices]
    
    val_texts = [training_data.texts[i] for i in val_indices]
    val_labels = [training_data.labels[i] for i in val_indices]
    
    train_data = TrainingData(
        texts=train_texts,
        labels=train_labels,
        classification_type=training_data.classification_type
    )
    
    val_data = TrainingData(
        texts=val_texts,
        labels=val_labels,
        classification_type=training_data.classification_type
    )
    
    return train_data, val_data


def balance_data(
    training_data: TrainingData,
    method: str = 'oversample',
    random_seed: Optional[int] = None
) -> TrainingData:
    """Balance training data by class distribution.
    
    Args:
        training_data: Original training data
        method: Balancing method ('oversample', 'undersample')
        random_seed: Random seed for reproducibility
        
    Returns:
        Balanced training data
    """
    if training_data.classification_type != ClassificationType.MULTI_CLASS:
        raise ValueError("Balancing is only supported for multi-class classification")
    
    if random_seed is not None:
        random.seed(random_seed)
    
    # Count samples per class
    label_counts = Counter(training_data.labels)
    
    if method == 'oversample':
        # Oversample to match the largest class
        max_count = max(label_counts.values())
        target_count = max_count
    elif method == 'undersample':
        # Undersample to match the smallest class
        min_count = min(label_counts.values())
        target_count = min_count
    else:
        raise ValueError("Method must be 'oversample' or 'undersample'")
    
    # Group samples by label
    label_samples = defaultdict(list)
    for i, label in enumerate(training_data.labels):
        label_samples[label].append(i)
    
    # Balance each class
    balanced_indices = []
    
    for label, sample_indices in label_samples.items():
        current_count = len(sample_indices)
        
        if method == 'oversample' and current_count < target_count:
            # Oversample by repeating samples
            needed = target_count - current_count
            additional_indices = random.choices(sample_indices, k=needed)
            balanced_indices.extend(sample_indices + additional_indices)
        
        elif method == 'undersample' and current_count > target_count:
            # Undersample by randomly selecting samples
            selected_indices = random.sample(sample_indices, target_count)
            balanced_indices.extend(selected_indices)
        
        else:
            # Keep all samples
            balanced_indices.extend(sample_indices)
    
    # Shuffle the balanced indices
    random.shuffle(balanced_indices)
    
    # Create balanced training data
    balanced_texts = [training_data.texts[i] for i in balanced_indices]
    balanced_labels = [training_data.labels[i] for i in balanced_indices]
    
    return TrainingData(
        texts=balanced_texts,
        labels=balanced_labels,
        classification_type=training_data.classification_type
    )


def get_data_statistics(training_data: TrainingData) -> Dict[str, Any]:
    """Get statistics about the training data.
    
    Args:
        training_data: Training data to analyze
        
    Returns:
        Dictionary containing data statistics
    """
    stats = {
        'total_samples': len(training_data.texts),
        'classification_type': training_data.classification_type.value,
        'text_lengths': {
            'min': min(len(text) for text in training_data.texts),
            'max': max(len(text) for text in training_data.texts),
            'mean': sum(len(text) for text in training_data.texts) / len(training_data.texts),
        }
    }
    
    if training_data.classification_type == ClassificationType.MULTI_CLASS:
        # Multi-class statistics
        label_counts = Counter(training_data.labels)
        stats['num_classes'] = len(label_counts)
        stats['class_distribution'] = dict(label_counts)
        stats['class_balance'] = {
            'most_common': label_counts.most_common(1)[0],
            'least_common': label_counts.most_common()[-1],
            'balance_ratio': label_counts.most_common()[-1][1] / label_counts.most_common(1)[0][1]
        }
    
    else:
        # Multi-label statistics
        all_labels = []
        for label_list in training_data.labels:
            all_labels.extend(label_list)
        
        label_counts = Counter(all_labels)
        labels_per_sample = [len(label_list) for label_list in training_data.labels]
        
        stats['num_classes'] = len(label_counts)
        stats['class_distribution'] = dict(label_counts)
        stats['labels_per_sample'] = {
            'min': min(labels_per_sample),
            'max': max(labels_per_sample),
            'mean': sum(labels_per_sample) / len(labels_per_sample)
        }
    
    return stats

