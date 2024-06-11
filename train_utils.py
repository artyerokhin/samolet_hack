import numpy as np
import pandas as pd

from collections import Counter


def tokenize_and_align_labels(tokenizer, label_list, examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        label_ids = [label_list.index(idx) if isinstance(idx, str) else idx for idx in label_ids]

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(label_list, metric, p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def data_processor(df, text_col, target_col):
    processed_data = []
    for text, target in df[[text_col, target_col]].values:
        splitted_text = text.split(' ')
        # dummy tags
        tag_list = ['O'] * len(splitted_text)
        # change tags for real ones
        for key in target.keys():
            for item in target[key]:
                tag_list[item] = key
        processed_data.append({'tokens': splitted_text, 'tags': tag_list})
    # return funal processed list of dictionaries
    return processed_data


def most_common_words(df, text_col, label_col, filter_cnt=5):
    """Get dict of most common words for entities (to filter noised words)"""
    common_dict = {}
    
    for _, text in df.iterrows():
        splitted = text['processed_text'].split(' ')
        data_dict = text['target_labels_positions']
        for key in data_dict.keys():
            for item in data_dict[key]:
                if key not in common_dict.keys():
                    common_dict[key] = [splitted[item]]
                else:
                    common_dict[key].append(splitted[item])
    
    for key in common_dict.keys():
        most_common = Counter(common_dict[key]).most_common()
        common_dict[key] = [word for word, cnt in most_common if cnt >= filter_cnt]
    
    return common_dict


def fix_train_common(df, fix_dict, text_col, label_col):
    """Save only most common words (small fix for labels).
    
    fix_dict is {label: list_of_words}
    """
    
    df = df.copy()
    new_labels = []
    
    for _, row in df.iterrows():
        splitted = row[text_col].split(' ')
        data_dict = row[label_col]
        new_data_dict = {}
        
        for key in data_dict.keys():
            for val in data_dict[key]:
                if splitted[val] in fix_dict[key]:
                    if key in new_data_dict.keys():
                        new_data_dict[key].append(val)
                    else:
                        new_data_dict[key] = [val]
        new_labels.append(new_data_dict)
    
    df[label_col] = new_labels
    return df