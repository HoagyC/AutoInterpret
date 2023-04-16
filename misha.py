# %%
import os
import pickle
import re
import sys
from pathlib import Path
from typing import List, Tuple

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from transformer_lens import HookedTransformer


# %%
import main

# %%
model = HookedTransformer.from_pretrained(model_name=main.MODEL_ACTIVATIONS, device="cpu")

# %%
dataset = datasets.load_dataset("NeelNanda/pile-10k")

# %%
dataset["train"][0]
# %%
dataset_texts = [row["text"] for row in dataset["train"]]

# %%
LAYER_N = 31
NEURON_N = 892


def get_activations(texts: List[str]) -> Tuple[List[float], List[str]]:
    tokenization_result = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenization_result["input_ids"]
    attention_mask = tokenization_result["attention_mask"]
    out, cache = model.run_with_cache(tokens)

    next_tokens = []
    activations = []
    end_indices = attention_mask.sum(axis=-1) - 1
    neuron_activations = cache["post", LAYER_N, "mlp"][:, :, NEURON_N]
    for batch_idx in range(len(texts)):
        next_tokens.append(model.to_string(out[batch_idx, end_indices[batch_idx], :].argmax(axis=-1).item()))
        batch_activations = neuron_activations[batch_idx, : end_indices[batch_idx] + 1]
        activations.append(torch.max(batch_activations).item())
        # activations.append(batch_activations[-1].item())
    return activations, next_tokens


def get_activations_batched(texts: List[str], batch_size: int = 32) -> Tuple[List[float], List[str]]:
    activations, next_tokens = [], []
    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        batch_activations, batch_next_tokens = get_activations(texts[i : i + batch_size])
        activations.extend(batch_activations)
        next_tokens.extend(batch_next_tokens)
    return activations, next_tokens



# %%
activations, next_tokens = get_activations_batched(dataset_texts[:4], batch_size=4)
pickle.dump(activations, "activations.pickle")
pickle.dump(next_tokens, "next_tokens.pickle")

# %%
exit()

# %%
# FRUITS

# %%
fruits = [
    "apple",
    "banana",
    "cherry",
    "date",
    "elderberry",
    "fig",
    "grape",
    "honeydew",
    "ice cream",
    "jackfruit",
    "kiwi",
    "lemon",
    "mango",
    "nectarine",
    "orange",
    "pear",
    "quince",
    "raspberry",
    "strawberry",
    "tangerine",
    "ugli fruit",
    "vanilla",
    "watermelon",
    "xigua",
    "yam",
    "zucchini",
]

prompts = [
    f"In the forest, there is a pineapple tree. You shake the tree, and down falls a pineapple. In the forest, there is a {fruit} tree. You shake the tree, and down falls"
    for fruit in fruits
]
activations, next_tokens = get_activations(prompts)
print(activations)
df = pd.DataFrame(
    dict(
        fruit=fruits,
        prompt=prompts,
        activation=activations,
        next_token=next_tokens,
    )
)
df = df.sort_values("activation", ascending=False)
df

# %%
# WIKI

# %%
wiki_sentences = main.get_wiki_sentences()


# %%
aan_regex = re.compile(r"(.*)( an| a)\b")
sentences = []
labels = []
for sentence in wiki_sentences:
    for prefix, aan in aan_regex.findall(sentence):
        sentences.append(prefix + aan)
        labels.append(aan == " an")
activations, next_tokens = get_activations_batched(sentences, 32)
# %%
df = pd.DataFrame(
    dict(
        sentence=sentences,
        label=labels,
        activation=activations,
        next_token=next_tokens,
    )
)
df = df.sort_values("activation", ascending=False)
df

# %%
sns.histplot(data=df[df["label"]], x="activation", multiple="stack", bins=50)
plt.show()
sns.histplot(data=df[~df["label"]], x="activation", multiple="stack", bins=50)
plt.show()

# %%
for row in list(df.itertuples())[:10]:
    print(row.label, row.next_token, row.sentence, sep="\t")
print("===")
for row in list(df.itertuples())[-10:]:
    print(row.label, row.next_token, row.sentence, sep="\t")

# %%
plt.plot(np.array(df.sort_values("activation").reset_index()["label"].tolist(), dtype=np.float32).cumsum())