#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import collections
import math
import pickle

from datasets import Dataset
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import create_optimizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
from transformers.data.data_collator import tf_default_data_collator
from transformers.keras_callbacks import PushToHubCallback

# set up model, tokenizer, and data
chunk_size = 128
wwm_probability = 0.2
model_checkpoint = "michiyasunaga/BioLinkBERT-large"

tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
model = AutoModelForMaskedLM.from_pretrained("michiyasunaga/BioLinkBERT-large")
# model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large")


def tokenize(text):
    result = tokenizer(text["abstracts"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return tf_default_data_collator(features)


def main() -> None:
    """Main function"""
    # tokenize dataset for BERT
    abstracts = "/scratch/remills_root/remills/stevesho/bio_nlp/nlp/classification/abstracts_classified_tfidf_20000.pkl"
    abstracts = pd.read_pickle(abstracts)
    abstracts = abstracts.loc[abstracts["predictions"] == 1]
    abstracts = Dataset.from_pandas(abstracts)
    tokenized_dataset = abstracts.map(
        tokenize, batched=True, num_proc=4, remove_columns=["predictions"]
    )

    with open(f"data/tokenized_classified_abstracts.pkl", "wb") as f:
        pickle.dump(tokenized_dataset, f, protocol=4)

    lm_datasets = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=512,
        num_proc=16,
    )

    # save tokenized dataset
    with open(f"data/lm_tokenized/)_classified_abstracts.pkl", "wb") as f:
        pickle.dump(lm_datasets, f)

    # # prepare to train!
    # model_name = model_checkpoint.split("/")[-1]

    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm_probability=0.15
    # )

    # train_size = 500_000
    # test_size = int(0.1 * train_size)

    # downsampled_dataset = lm_datasets["train"].train_test_split(
    #     train_size=train_size, test_size=test_size, seed=42
    # )

    # tf_train_dataset = model.prepare_tf_dataset(
    #     downsampled_dataset["train"],
    #     collate_fn=data_collator,
    #     shuffle=True,
    #     batch_size=32,
    # )

    # tf_eval_dataset = model.prepare_tf_dataset(
    #     downsampled_dataset["test"],
    #     collate_fn=data_collator,
    #     shuffle=False,
    #     batch_size=32,
    # )

    # num_train_steps = len(tf_train_dataset)
    # optimizer, schedule = create_optimizer(
    #     init_lr=2e-5,
    #     num_warmup_steps=1_000,
    #     num_train_steps=num_train_steps,
    #     weight_decay_rate=0.01,
    # )
    # model.compile(optimizer=optimizer)

    # # Train in mixed-precision float16
    # tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # model.fit(tf_train_dataset, validation_data=tf_eval_dataset)
    # eval_loss = model.evaluate(tf_eval_dataset)
    # print(f"Perplexity: {math.exp(eval_loss):.2f}")

    # model.save_pretrained(f"{model_name}-finetuned-genomics")


if __name__ == "__main__":
    main()
