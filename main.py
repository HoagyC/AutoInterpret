import argparse
from collections import defaultdict
import copy
from dataclasses import dataclass
import itertools
import json
import os
import pickle
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import openai
import pandas as pd
import torch
import tqdm


from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from transformer_lens import HookedTransformer # pip install git+https://github.com/neelnanda-io/TransformerLens

from datasets import load_dataset

from utils import pearsonr_ci, download_from_aws

# Format is (binary_condition, text, answer)

examples = [
    ("The token is part of a year", "The baseball team won the 19[[63]] championship", "63", "YES"),
    ("The token is preceded by a proper noun", "Canada is a member of[[ the]] International", " the", "NO"),
]
MODEL_ACTIVATIONS = "gpt2-small"
MODEL_CONDITION_GENERATION = "gpt-3.5-turbo"
MODEL_SUGGESTION_GENERATION = "gpt-4"
MODEL_EMBEDDINGS = "text-embedding-ada-002"

INIT_TRIALS = 10 # Number of positive and negative cases to check each suggsted condition with
N_TOP_TEST = 5 # Number of top existing conditions to test at the end of each turn
N_EXTRA_TRIALS = 2 # Number of positive and negative cases to check each suggsted condition with
TRIALS_PER_ROUND = (INIT_TRIALS + (N_TOP_TEST * N_EXTRA_TRIALS)) * 2 # Use a positive and negative example for each suggested condition

VERBOSE = False
DISPLAY_INTERVAL = 1

model_short_name = MODEL_ACTIVATIONS.split("-")[-1]

def make_base_message(examples: List[Tuple[str, str, str, str]] = []) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Is the criterion a true statement about the token in context? Answer YES or NO. [[ ActiveToken]], meets the criterion described in the prompt."},
    ]
    if examples:
        for binary_condition, context, token, answer in examples:
            messages.append({"role": "user", "content": f"Criterion: '{binary_condition}'\nToken:'{token}'\nContext: {context}"})
            messages.append({"role": "assistant", "content": answer})
    return messages

base_message = make_base_message(examples)

# Get openai key from secrets.json
openai.api_key = json.load(open("secrets.json"))["openai_key"]

def draw_histogram(activations: List[torch.Tensor]) -> None:
    for activation in activations:
        plt.hist(activation.flatten(), bins=100, alpha=0.5)
    plt.legend()
    plt.show()

def display_chatgpt_message(messages: List[Dict[str, str]]):
    # Take a message reading to be given to chatgpt and display it in a way that makes it easy to read
    for message in messages:
        print("\n")
        if message["role"] == "user":
            print(repr(f"User: {message['content']}"))
        else:
            print(repr(f"Assistant: {message['content']}"))

def get_sentences(n: int = 5000) -> List[str]:
    sentence_list: List[str] = []
    dataset = load_dataset("NeelNanda/pile-10k")
    while len(sentence_list) < n:
        sentence = dataset["train"][random.randint(0, len(dataset["train"]))]["text"]
        # Cut off after a maximum of 20 words
        sentence = " ".join(sentence.split(" ")[:20])
        # If it contains non-ascii characters, skip it
        if not all(ord(c) < 128 for c in sentence) or len(sentence) < 20:
            continue
        sentence_list.append(sentence)

    return sentence_list

def embed_all(corpus: List[str], model) -> Tuple[List[int], List[str], List[torch.Tensor]]:
    # Get embeddings for all sentences
    sentence_token_lengths: List[int] = []
    sentence_fragments: List[str] = []
    embeddings: List[torch.Tensor] = []

    print("Embedding all sentences")
    for sentence in tqdm.tqdm(corpus):
        tokens = model.to_tokens(sentence)
        partial_sentences = []
        sentence_token_lengths.append(tokens.shape[1] - 1)
        for i in range(1, tokens.shape[1]):
            partial_sentence = model.to_string(tokens[0, 1:i + 1])
            partial_sentences.append(partial_sentence) # 1 to skip the <endoftext> token
            sentence_fragments.append(partial_sentence)

        embedding = openai.Embedding.create(input = partial_sentences, model = MODEL_EMBEDDINGS)
        embeddings.extend([torch.Tensor(e["embedding"]) for e in embedding["data"]])
    return sentence_token_lengths, sentence_fragments, embeddings


class NdxTool:
    def __init__(self, sentence_lengths, sentence_fragments):
        self.sentence_lengths = sentence_lengths # Length of each sentence in tokens
        self.sentence_fragments = sentence_fragments
        assert sum(sentence_lengths) == len(sentence_fragments)
        self.cumsum = np.cumsum(sentence_lengths)
        self.cumsum = np.insert(self.cumsum, 0, 0)
    
    def __getitem__(self, ndx):
        return self.sentence_fragments[ndx]

    def __len__(self):
        return len(self.sentence_fragments)

    def get_sentence_ndx(self, ndx):
        return np.searchsorted(self.cumsum, ndx + 1) - 1
    
    def get_token(self, ndx):
        if ndx in self.cumsum:
            return self.sentence_fragments[ndx]
        else:
            sentence_with_token = self[ndx]
            sentence_without_token = self[ndx - 1]
            return sentence_with_token[len(sentence_without_token):]
    
    def get_other_sentence_ndxs(self, ndx):
        # Get all ndxs part of the same sentence 
        sentence_ndx = self.get_sentence_ndx(ndx)
        sentence_ndxs = np.arange(self.cumsum[sentence_ndx], self.cumsum[sentence_ndx + 1])
        return sentence_ndxs

    def get_full_sentence_ndx(self, ndx):
        # Get last ndx of sentence
        sentence_ndx = self.get_sentence_ndx(ndx)
        return self.cumsum[sentence_ndx + 1] - 1

    def get_all_full_sentences(self):
        return [self[i - 1] for i in self.cumsum[1:]]
    
    def get_full_sentence(self, ndx):
        return self[self.get_full_sentence_ndx(ndx)]
    

def make_internals_func(layer_n: int, neuron_n: int):
    def compute_internals_single(model: HookedTransformer, input_txt: str) -> float:
        tokens = model.to_tokens(input_txt)
        # Remove the last token: We want the activations when the last token was predicted.
        tokens = tokens[:, :-1]
        _, cache = model.run_with_cache(tokens, return_type=None, remove_batch_dim=True)
        activations = cache["post", layer_n, "mlp"]
        return activations[-1][neuron_n].tolist()

    return compute_internals_single

def make_all_internals(
        layer_ndxs: List[int], 
        neuron_ndxs: List[int], 
        model: HookedTransformer, 
        sentence_list: List[str],
        n_ndxs: int = 1000
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[int]]]]:
    # Make dataframe with a column for each layer and neuron, with an entry for each full and partially tokenized sentence


    all_internals = []
    print(f"Computing internals for all {len(sentence_list)} sentences")
    for sentence in tqdm.tqdm(sentence_list):
        tokens = model.to_tokens(sentence)
        _, cache = model.run_with_cache(tokens, return_type=None, remove_batch_dim=True)
        for i in range(1, tokens.shape[1]):
            partial_sentence = model.to_string(tokens[0, 1 : i + 1])
            partial_str_dict: Dict[str, Any] = {}
            partial_str_dict["context"] = partial_sentence
            partial_str_dict["pre_tokens"] = tokens[0, 1 : i + 1].tolist()
            partial_str_dict["full_tokens"] = tokens[0, 1:].tolist()
            partial_str_dict["sentence"] = sentence
    
            for layer_n in layer_ndxs:
                activations = cache["post", layer_n, "mlp"]
                assert activations.shape[0] == tokens.shape[1]
                for neuron_n in neuron_ndxs:
                    partial_str_dict[f"l{layer_n}n{neuron_n}"] = activations[i][neuron_n].tolist()
                

            all_internals.append(partial_str_dict)

    # Make the dataframe
    df = pd.DataFrame(all_internals)

    # Now we want to get a list of places where there are high and low activations for each neuron
    # And return their indices so that we can access them quickly

    ndxs_dict = {}
    for layer_n in layer_ndxs:
        for neuron_n in neuron_ndxs:
            # Get the indices of the top 10% activations (so long as they are positive) and sample from zero activations
            column_title = f"l{layer_n}n{neuron_n}"
            activations = df[column_title]

            # Get the top n_ndxs activations
            top_ndxs = activations.nlargest(n_ndxs).index
            # Check if they are all positive, if not, cut off the bottom ones
            if not all(activations[top_ndxs] > 0):
                top_ndxs = activations[top_ndxs][activations[top_ndxs] > 0].index
                print(f"Cut off {n_ndxs - len(top_ndxs)} activations for layer {layer_n}, neuron {neuron_n} because they were negative, leaving {len(top_ndxs)} activations")

            # TODO: Delete carefully as made obsolete by using embeddings
            # Sample n_ndxs activations where the activation is negative
            neg_ndxs = activations[activations < 0].sample(n_ndxs).index
            if len(neg_ndxs) < n_ndxs:
                print(f"Only sampled {len(neg_ndxs)} negative activations for layer {layer_n}, neuron {neuron_n}, leaving {n_ndxs - len(neg_ndxs)} activations")
        
            # Add to the ndxs_dict
            ndxs_dict[column_title] = {"top": top_ndxs, "neg": neg_ndxs}
         
    return df, ndxs_dict

class BinaryCondition():
    def __init__(
        self,
        condition: str,
        datapoints: Optional[List[Tuple[float, float]]] = None,
        score: float = 0.,
        ucb: float = 1.,
        lcb: float = -1.,
        p_val: float = 1.
    ):
        self.condition = condition
        self.datapoints = datapoints if datapoints is not None else []
        self.score = score
        self.ucb = ucb
        self.lcb = lcb
        self.p_val = p_val
    
    def add_datapoint(self, datapoint: Tuple[float, float]):
        self.datapoints.append(datapoint)
    
    def update_score(self):
        print("current length of datapoints: ", len(self.datapoints))
        internal_vals = np.array([x[0] for x in self.datapoints])
        explanation_vals = np.array([x[1] for x in self.datapoints])
        self.score, self.lcb, self.ucb, self.p_val = pearsonr_ci(internal_vals, explanation_vals)
    
    @staticmethod
    def from_dict(d: Dict):
        return BinaryCondition(
            condition=d["condition"],
            datapoints=d["datapoints"],
            score=d["score"],
            ucb=d["ucb"],
            lcb=d["lcb"],
        )
    
    def to_dict(self) -> Dict:
        return {
            "condition": self.condition,
            "datapoints": self.datapoints,
            "score": self.score,
            "ucb": self.ucb,
            "lcb": self.lcb,
            "p_val": self.p_val
        }
    
    
def evaluate_prompt_single(
        token: str, 
        context: str,
        binary_condition: BinaryCondition,
    ) -> float:

    message = copy.deepcopy(base_message)
    message.append({"role": "user", "content": f"Criterion: '{binary_condition.condition}' Token: '{token}' Context: {context}"})
    response_status = -1
    while response_status == -1:
        display_chatgpt_message(message)
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_CONDITION_GENERATION,
                messages = message,
                max_tokens=2,
            )

        except openai.error.RateLimitError:
            print("OpenAI servers overloaded, waiting 1 second")
            time.sleep(1)
            continue

        except openai.error.APIError:
            print("API error, waiting 1 second")
            time.sleep(1)
            continue

        response_str = response["choices"][0]["message"]["content"]
        if response_str[:3] == "YES":
            return 1.
        elif response_str[:2] == "NO":
            return 0.
        else:
            print(f"Non-compliant response: '{response_str}'")
            return 0.5
        # print(f"Got response for case {ndx} of {n_test_cases}. Response status: {response_status}")
        # message.append({"role": "assistant", "content": "YES" if response_status == 1 else "NO"})
        
        # if int(" and " in text) != response_status:

        #     print(f"Got wrong response for case {input_txt}. Response status: {response_status}")
        
    return response_status
    
    
def score_condition_by_corr(
    activation_list: List[float],
    responses: List[float],
):
    assert len(activation_list) == len(responses)
    if max(responses) == min(responses):
        return 0
    return np.corrcoef(activation_list, responses)[0, 1]
    

# For scoring multiple at a time
    # else:
    #     activations_arr = np.array([activation.flatten()[activation_n] for _, activation in test_cases])
    #     correlations = [np.corrcoef(activations_arr, responses)[0, 1]]

    # return correlations


class ExplainGame():
    def __init__(
            self, 
            internals_fn: Callable[[int], Tuple[str, float]],
            embeddings: List[torch.Tensor],
            high_ndxs: List[int],
            low_ndxs: List[int],
            ndx_tool: NdxTool,
            eval_func: Callable[[str, str, BinaryCondition], float],
            agreement_func: Callable[[List[float], List[float]], float],
        ):
        print("Initializing ExplainGame")
        self.internals_fn = internals_fn
        self.eval_func = eval_func
        self.agreement_func = agreement_func

        self.high_ndxs = high_ndxs
        self.low_ndxs = low_ndxs
        self.ndx_tool = ndx_tool

        self.embeddings = embeddings
        print("Done initializing ExplainGame")
    

    def evaluate_explanation(self, explanation: BinaryCondition, n_trials: int) -> float:
        # Principle here is that if the explanation looks good initially then we should run on a larger set of examples
        # and if it looks bad then we should run on a smaller set of examples then ignore
        for _ in range(n_trials):
            high_text_ndx = random.choice(self.high_ndxs)
            high_text, high_int_val = self.internals_fn(high_text_ndx)
            context, token = annotate_high_sentence(high_text_ndx, self.ndx_tool)
            explanation_val_h = self.eval_func(token, context, explanation)
            explanation.add_datapoint((high_int_val, explanation_val_h))
            if VERBOSE:
                print(f"HIGH: Evaluated token {token}, context: {context}, bincon: {explanation.condition}, output: {explanation_val_h}")

            low_text_ndx = random.choice(self.low_ndxs)
            low_text, low_int_val = self.internals_fn(low_text_ndx)
            context, token = annotate_high_sentence(low_text_ndx, self.ndx_tool)
            explanation_val_l = self.eval_func(token, context, explanation)
            explanation.add_datapoint((low_int_val, explanation_val_l))

            if VERBOSE:
                print(f"LOW: Evaluated token {token}, context: {context}, bincon: {explanation.condition}, output: {explanation_val_l}")

        explanation.update_score()
        return explanation.score
    
    def get_high_contexts(self, n: int = 10) -> Tuple[List[int], List[str]]:
        high_ndxs = random.sample(list(self.high_ndxs), n)
        high_contexts = []
        for ndx in high_ndxs:
            context, _ = self.internals_fn(ndx)
            high_contexts.append(context)
        
        return high_ndxs, high_contexts
    
    def evaluate_high_ucbs(self, bc_list: List[BinaryCondition], n_iters: int):
        # Evaluate the high ucb conditions
        print("Evaluating high ucb conditions")
        iter = 0
        while iter < n_iters:
            iter += 1
            bc_list = sorted(bc_list, key=lambda x: x.ucb, reverse=True)
            top_bc = bc_list[0]
            self.evaluate_explanation(top_bc, n_trials=N_EXTRA_TRIALS)
        

def suggestion_generator(
    scored_conditions: List[BinaryCondition],
    high_prompts: List[Tuple[str, str]],
    # low_prompts: List[str],
) -> BinaryCondition:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    initial_instructions = "You will first be shown a series of places in text where a neuron in a model is active, which will be marked with square brackets, [[]]." + \
        " You will then be shown zero or more binary conditions which are hypotheses as to what may explain this behaviour and a " + \
        "score which judges how well the hypothesis explains the behaviour. For example \"The token is the beginning of a surname\", or \"The token comes after a sports-related noun.\". " + \
        "You will then be asked to suggest an improved binary condition in a single sentence which explains the behaviour. Some things to remember:\n" + \
        "(1) this should try to explain the behaviour for as many of the shown examples as possible.\n" + \
        "(2) the condition can be negative as well as positive.\n" + \
        "(3) try to be specific to the exact token, not just the context.\n" + \
        "(4) if there are good suggestions, say (score>0.3), you could try small variations on that theme,\n" + \
        "(5) we've tried many conditions, so don't be afraid to suggest something unusual."
    
    messages.append({"role": "user", "content": initial_instructions})
    messages.append({"role": "user", "content": "Here are sentences which have activated this neuron. Remember, tokens which activate it highly are surrounded by double square brackets, e.g. 'This is an [[ ActiveToken]]'."})
    for i, (context, token) in enumerate(high_prompts):
        messages.append({"role": "user", "content": f"Example {i + 1}\nToken: [[{token}]].\nContext: {context}"})
    
    # messages.append({"role": "user", "content": "Low prompts:"})
    # for prompt in low_prompts:
    #     messages.append({"role": "user", "content": prompt + "\n"})

    # Sample which previous conditions to show by taking a softmax over the scores
    n_conditions_to_show = 3
    raw_scores = [sc.lcb for sc in scored_conditions]
    temperature = np.std(raw_scores)
    if len(raw_scores) > n_conditions_to_show:
        scores = np.array(raw_scores)
        scores = scores - np.max(scores) # To avoid numerical issues
        scores = np.exp(scores * temperature)
        scores = scores / np.sum(scores)
        ndxs = np.random.choice(len(scores), n_conditions_to_show, p=scores)
        used_examples = [scored_conditions[ndx] for ndx in ndxs]
        
        used_scores = [bc.score for bc in used_examples]
    else:
        used_examples = scored_conditions

    for ndx, bc in enumerate(used_examples):
        messages.append({"role": "user", "content": f"Score of condition {ndx}: {bc.score}"})
        messages.append({"role": "user", "content": f"Condition {ndx}: {bc.condition}"})

    target_score = max(raw_scores) + 0.1 if len(raw_scores) > 0 else 0.0
    messages.append({"role": "user", "content": f"Score of new condition: {target_score}"})
    messages.append({"role": "user", "content": "New condition:"})

    all_conditions = [sc.condition for sc in scored_conditions]
    found_new_condition = False

    if VERBOSE:
        display_chatgpt_message(messages)

    while not found_new_condition:
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=MODEL_SUGGESTION_GENERATION,
                    messages = messages,
                )
                break

            except openai.error.RateLimitError:
                print("OpenAI servers overloaded, waiting 1 second")
                time.sleep(1)
                continue
            except openai.error.APIError:
                print("APIError, waiting 1 second")
                time.sleep(1)
                continue

        response_txt = response["choices"][0]["message"]["content"]
        # Cut off anything after the first sentence
        if (". " in response_txt and response_txt.index(". ") < len(response_txt) - 2):
            new_binary_condition = response_txt[:response_txt.index(". ")]
            print(f"Truncating multisentence response: '{response_txt}'")

        elif "\n" in response_txt and response_txt.index("\n") < len(response_txt) - 2:
            new_binary_condition = response_txt[:response_txt.index("\n")]
            print(f"Truncating multiline response: '{response_txt}'")
        else:
            new_binary_condition = response_txt
        
        # Check that the condition is not already in the list
        # TODO: some kind of fuzzy matching?
        if new_binary_condition not in all_conditions:
            found_new_condition = True

    return BinaryCondition(new_binary_condition)

def annotate_high_sentence(sentence_ndx, ndx_tool: NdxTool) -> Tuple[str, str]:
    token = ndx_tool.get_token(sentence_ndx)
    rest_of_sentence_ndxs = ndx_tool.get_other_sentence_ndxs(sentence_ndx)

    # TODO: Work in multiple high activations
    # full_sentence = ndx_tool.get_full_sentence(sentence_ndx)
    # high_ndxs = [(internals_fn(sentence_ndx)[1], sentence_ndx)]
    # for ndx in rest_of_sentence_ndxs:
    #     if ndx == sentence_ndx:
    #         continue
        
    #     sentence, activation = internals_fn(ndx)
    #     if activation > threshold:
    #         high_ndxs.append((activation, ndx))
    
    # if len(high_ndxs) > 5:
    #     high_ndxs = sorted(high_ndxs, reverse=True)[:5]
    
    # Add square brackets around the tokens that are active
    new_str = ""
    found_token = False
    for ndx in rest_of_sentence_ndxs:
        if ndx == sentence_ndx:
            new_str += "[["
            found_token = True
        
        new_str += ndx_tool.get_token(ndx)
        
        if ndx == sentence_ndx:
            new_str += "]]"
    if not found_token:
        breakpoint()
    
    return new_str, token
    

class GamePlayer():
    def __init__(
        self,
        game: ExplainGame,
        layer: int,
        neuron: int,
        load: bool = True
    ):
        self.game = game
        self.layer = layer
        self.neuron = neuron

        # Check if we have any suggestions already
        self.save_path = os.path.join(f"found_conditions_{model_short_name}", f"layer_{self.layer}", f"neuron_{self.neuron}.json")
        if os.path.exists(self.save_path) and load:
            with open(self.save_path, "r") as f:
                self.suggestions = [BinaryCondition.from_dict(bc) for bc in json.load(f)]
        else:
            self.suggestions = []


    def run_turn(self) -> Tuple[str, List[float]]:
        # Get a few low and high suggestions
        high_ndxs, high_contexts = self.game.get_high_contexts(5)
        annotated_high_sentences = [annotate_high_sentence(ndx, self.game.ndx_tool) for ndx in high_ndxs]
        # for high_ndx in high_ndxs:
            # close_low_ndxs = get_similar_low_sentences(high_ndx, self.game.internals_fn, self.game.ndx_tool, self.game.embeddings, n=10)
            # close_low_fragments = [self.game.ndx_tool[ndx] for ndx in close_low_ndxs]
        new_condition = suggestion_generator(self.suggestions, annotated_high_sentences)
        print(f"New suggested condition: {new_condition.condition}")
        init_score = self.game.evaluate_explanation(new_condition, INIT_TRIALS)
        print(f"Initial score of condition: {init_score}")
        self.suggestions.append(new_condition)
        self.game.evaluate_high_ucbs(self.suggestions, n_iters=N_TOP_TEST)
        if len(self.suggestions) % 5 == 0:
            top_3_suggestions = sorted(self.suggestions, key=lambda x: x.lcb, reverse=True)[:3]
            # Print the top 3 suggestions
            print("\n")
            print(f"Top 3 binary criteria for layer{self.layer}, neuron{self.neuron} after {len(self.suggestions)} suggestions:")
            for ndx, bc in enumerate(top_3_suggestions):
                if bc.p_val == 1.0:
                    bc.update_score() # Only needed for runs saved before p-vals, can delete soon
                print(f"Condition {ndx + 1}: {bc.condition}")
                # Suspect that the appropriate Bonferroni correction is ~ total_datapoints / len(datapoints), might be nonsense
                correction = len(self.suggestions * TRIALS_PER_ROUND) / len(bc.datapoints) # 10 for initial conditions, 9 for new conditions
                print(f"LCB: {bc.lcb:.3f}, Score: {bc.score:.3f}, P-val = {bc.p_val * correction:.6f}")
            print("\n")

            # Save the suggestions
            if not os.path.exists(os.path.dirname(self.save_path)):
                os.makedirs(os.path.dirname(self.save_path))

            with open(self.save_path, "w") as f:
                json.dump([bc.to_dict() for bc in self.suggestions], f)
        
        # Returning the new condition and the yes/no values of the datapoints
        return new_condition.condition, [dp[1] for dp in new_condition.datapoints]
    
def make_db_internals_fn(layer_n: int, neuron_n: int, internals: pd.DataFrame) -> Callable[[int], Tuple[str, float]]:
    def db_internals_fn(ndx: int) -> Tuple[str, float]:
        # Get index of text in internals
        column_name = f"l{layer_n}n{neuron_n}"
        sentence = internals.iloc[ndx]["context"]
        internal_val = internals.iloc[ndx][column_name]
        return sentence, internal_val

    return db_internals_fn

def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--remake", type=bool, default=False)
    parser.add_argument("--load", type=bool, default=True)
    parser.add_argument("--n_turns", type=int, default=40)
    # parser.add_argument("--layer", type=int, default=-1)
    # parser.add_argument("--neuron", type=int, default=-1)

    args = parser.parse_args()

    # if args.neuron == -1:
    #     args.neuron = random.randint(0, 3071)
    # if args.layer == -1:
    #     args.layer = random.randint(0, 11)

    game_players: List[GamePlayer] = []

    n_rng = (10, 20)
    l_rng = (1, 2)
    layers = list(range(*l_rng))
    neurons = list(range(*n_rng))
    n_sentences = 5000
    
    # Load the internals
    internals_str = f"{model_short_name}_internals/s{n_sentences}_l{l_rng[0]}-{l_rng[1]}_n{n_rng[0]}-{n_rng[1]}"
    full_internals_loc = internals_str + ".pkl"
    internals_ndxs_loc = internals_str + "_ndxs.pkl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL_ACTIVATIONS, device=device)     

    if not os.path.exists(f"sentencedata{n_sentences}.pkl"):
        corpus = get_sentences(n=n_sentences)       
        sentence_lengths, sentence_fragments, embeddings = embed_all(corpus, model)
        pickle.dump((sentence_lengths, sentence_fragments, embeddings), open(f"sentencedata{n_sentences}.pkl", "wb"))
    
    with open(f"sentencedata{n_sentences}.pkl", "rb") as f:
        sentence_lengths, sentence_fragments, embeddings = pickle.load(f)
    
    ndx_tool = NdxTool(sentence_lengths, sentence_fragments) # Utility for converting from index to sentence fragment etc
    corpus = ndx_tool.get_all_full_sentences()

    if not os.path.exists(full_internals_loc) or not os.path.exists(internals_ndxs_loc):
        internals, internals_ndxs = make_all_internals(layers, neurons, model, corpus)
        internals.to_pickle(full_internals_loc)
        pickle.dump(internals_ndxs, open(internals_ndxs_loc, "wb"))
    
    with open(full_internals_loc, "rb") as f:
        internals = pickle.load(f)
    
    with open(internals_ndxs_loc, "rb") as f:
        internals_ndxs = pickle.load(f)

    for layer, neuron in list(itertools.product(layers, neurons)):
        internals_fn = make_db_internals_fn(layer, neuron, internals)
        ndxs_dict = internals_ndxs[f"l{layer}n{neuron}"]
        game = ExplainGame(
            internals_fn=internals_fn,
            embeddings=embeddings,
            high_ndxs=ndxs_dict["top"],
            low_ndxs=ndxs_dict["neg"],
            ndx_tool=ndx_tool,
            eval_func=evaluate_prompt_single,
            agreement_func=score_condition_by_corr,
        )

        game_player = GamePlayer(game, layer, neuron, load=args.load)
        game_players.append(game_player)


    for turn_ndx in range(args.n_turns):
        turn_ndx += 1
        print(f"Turn {turn_ndx}")
        for game_player in game_players:
            game_player.run_turn()
        

        if turn_ndx % DISPLAY_INTERVAL == 0:
            results = []
            for game_player in game_players:
                top_condition = sorted(game_player.suggestions, key=lambda x: x.lcb, reverse=True)[0]
                p_val_correction = len(game_player.suggestions) * len(game_players) * (TRIALS_PER_ROUND / len(top_condition.datapoints))

                # Add line breaks to the condition every 40 characters
                condition_w_breaks = top_condition.condition
                if len(condition_w_breaks) > 40:
                    n_breaks = len(condition_w_breaks) // 40
                    for i in range(n_breaks):
                        condition_w_breaks = condition_w_breaks[:40 * (i + 1) + i] + "\n" + condition_w_breaks[40 * (i + 1) + i:]

                results.append({
                    "layer": game_player.layer,
                    "neuron": game_player.neuron,
                    "condition": top_condition.condition,
                    "score": top_condition.score,
                    "lcb": top_condition.lcb,
                    "ucb": top_condition.ucb,
                    "p_val": top_condition.p_val * p_val_correction,
                    "description": f"Layer: {game_player.layer}, Neuron: {game_player.neuron},\nCondition: '{condition_w_breaks}'"
                })
            
            # Plot as a horizontal bar chart in descending order of score, with error bars
            df = pd.DataFrame(results)
            df = df.sort_values(by="score", ascending=True)

            # Show only those with score > 0 and up to 6 total
            df = df[df["score"] > 0]
            df = df[-6:]

            # Make large figure
            fig, ax = plt.subplots(figsize=(10, 5), )

            # Colours is purple if p_val < 1e-6, blue if p_val < 1e-4, green if p_val < 1e-2, yellow if p_val < 1e1, red if p_val > 1e1
            try:
                colours = pd.Series(["red"] * len(df))
                colours[(df["p_val"] < 1e-1).tolist()] = "yellow"
                colours[(df["p_val"] < 1e-2).tolist()] = "green"
                colours[(df["p_val"] < 1e-4).tolist()] = "blue"
                colours[(df["p_val"] < 1e-6).tolist()] = "purple"
            except:
                breakpoint()

            # Add colour legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='p_val < 1e-6', markerfacecolor='purple', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='p_val < 1e-4', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='p_val < 1e-2', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='p_val < 1e-1', markerfacecolor='yellow', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='p_val > 1e-1', markerfacecolor='red', markersize=10),
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            ax.barh(df["description"], df["score"], xerr=(df["score"] - df["lcb"], df["ucb"] - df["score"]), color=colours)

            ax.set_xlabel("Score")
            ax.set_ylabel("Neuron/condition")
            n_conditions = len(game_players[0].suggestions)
            ax.set_title(f"Top conditions for each neuron after {n_conditions} suggestions")

            # Give lots of space for the descriptions
            # Make font smaller
            plt.rcParams.update({'font.size': 6})
            plt.subplots_adjust(left=0.5)

            plt.savefig(f"graphs/l{l_rng[0]}-{l_rng[1]}_n{n_rng[0]}-{n_rng[1]}_top_conditions_{n_conditions}.png")


def test_binary_consistency() -> None:
    # Test the degree to which the model is self-consistent it its YES/NO responses
    n_sentences = 5000
    # Load the sentence data
    with open(f"sentencedata{n_sentences}.pkl", "rb") as f:
        sentence_lengths, sentence_fragments, embeddings = pickle.load(f)

    # Load the internals

    internals_str = f"small_internals/s5000_l1-2_n0-3"
    full_internals_loc = internals_str + ".pkl"
    internals_ndxs_loc = internals_str + "_ndxs.pkl"

    with open(full_internals_loc, "rb") as f:
        internals = pickle.load(f)
    
    with open(internals_ndxs_loc, "rb") as f:
        internals_ndxs = pickle.load(f)

    internals_fn = make_db_internals_fn(1, 1, internals)
    ndx_tool = NdxTool(sentence_lengths, sentence_fragments) # Utility for converting from index to sentence fragment etc

    # Load the binary conditions
    with open("found_conditions_small/layer_1/neuron_0.json", "r") as f:
        bin_conditions: List[BinaryCondition] = [BinaryCondition.from_dict(bc) for bc in json.load(f)][:5]
    print(f"Loaded {len(bin_conditions)} binary conditions")
    ndxs_dict = internals_ndxs[f"l1n1"]
    game = ExplainGame(
        internals_fn=internals_fn,
        embeddings=None,
        high_ndxs=ndxs_dict["top"],
        low_ndxs=ndxs_dict["neg"],
        ndx_tool=ndx_tool,
        eval_func=evaluate_prompt_single,
        agreement_func=score_condition_by_corr,
    )

    results: Dict[str, int] = defaultdict(int)
    n_iters = 10
    for condition, sentence_ndx in tqdm.tqdm(itertools.product(bin_conditions, list(range(n_iters)))):
        high_text_ndx, low_text_ndx = game.high_ndxs[sentence_ndx], game.low_ndxs[sentence_ndx]
        for sentence_ndx in [high_text_ndx, low_text_ndx]:
            context, token = annotate_high_sentence(sentence_ndx, ndx_tool)
            attempt_1 = evaluate_prompt_single(token, context, condition.condition)
            attempt_2 = evaluate_prompt_single(token, context, condition.condition)
            if attempt_1 and attempt_2:
                results["yes-yes"] += 1
            elif attempt_1 and not attempt_2 or not attempt_1 and attempt_2:
                results["yes-no"] += 1
            elif not attempt_1 and not attempt_2:
                results["no-no"] += 1
            else:
                raise ValueError("This shouldn't happen")

        
    print(f"Results: {results}")
    print(f"Proportion of yes-yes: {results['yes-yes'] / (n_iters * 2 * len(bin_conditions))}")
    print(f"Proportion of yes-no: {results['yes-no'] / (n_iters * 2 * len(bin_conditions))}")
    print(f"Proportion of no-no: {results['no-no'] / (n_iters * 2 * len(bin_conditions))}")
    print(f"Proportion of matches: {(results['yes-yes'] + results['no-no']) / (n_iters * 2 * len(bin_conditions))}")

def get_top_n_sentences(layer: int, neuron: int, n_sentences: int = 10):
    neuron_str = f"l{layer}_n{neuron}"
    # TODO: check through all pickles
    internals_df = pd.read_pickle(f"large_internals/s5000_l29-31_800-900.pkl")
    strs = [str(x) for x in internals_df.sort_values(by=neuron_str)["context"][-n_sentences:]]
    return strs

    
def get_similar_low_sentences(sentence_ndx: int, internals_fn: Callable[[int], Tuple[str, float]], ndx_tool: NdxTool, embeddings, n: int = 10, same_sentence_max: int = 5):
    rest_of_sentence_ndxs = ndx_tool.get_other_sentence_ndxs(sentence_ndx)
    top_cos_sims = []

    # Try to find low-activation fragments of the same sentence
    for ndx in rest_of_sentence_ndxs:
        if ndx == sentence_ndx:
            continue
        
        sentence, activation = internals_fn(ndx)
        if activation < 0:
            cos_sim = torch.nn.functional.cosine_similarity(embeddings[ndx], embeddings[sentence_ndx], dim=0)
            top_cos_sims.append((cos_sim, ndx))
    
    # Then also try to find low-activation fragments of other sentences
    n_test = 100
    other_sentence_ndxs = np.random.choice(np.arange(len(ndx_tool)), size=n_test, replace=False)
    for other_sentence_ndx in other_sentence_ndxs:
        if other_sentence_ndx == sentence_ndx:
            continue
        
        other_sentence, other_activation = internals_fn(other_sentence_ndx)
        if other_activation < 0:
            other_sentence_ndx = ndx_tool.get_full_sentence_ndx(other_sentence_ndx)
            cos_sim = torch.nn.functional.cosine_similarity(embeddings[other_sentence_ndx], embeddings[sentence_ndx], dim=0)
            top_cos_sims.append((cos_sim, other_sentence_ndx))
        
    top_cos_sims.sort(key=lambda x: float(x[0]), reverse=True)

    # Avoiding having all fragments from the same sentence
    n_same_sentence = 0
    close_sentence_ndxs: List[int] = []
    while len(close_sentence_ndxs) < n:
        _, ndx = top_cos_sims.pop(0)
        if ndx in rest_of_sentence_ndxs and n_same_sentence >= same_sentence_max:
            continue
        close_sentence_ndxs.append(ndx)

    return close_sentence_ndxs


    

if __name__ == "__main__":
    main()

    # for high_ndx in high_ndxs:
    #     print("High activation example:", ndx_tool[high_ndx])
    #     close_low_fragments = get_similar_low_sentences(high_ndx, internals["l1n1"], ndx_tool, embeddings, n=10)
    #     print("Similar low activations:")
    #     print([ndx_tool[ndx] for ndx in close_low_fragments])


    # main()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = HookedTransformer.from_pretrained("gpt2-large", device=device)

    # n_sentences = 5000
    # l_rng = (29, 31)
    # n_rng = (800, 900)
    # layers = list(range(*l_rng))
    # neurons = list(range(*n_rng))

    # df, ndxs_dict = make_all_internals(layers, neurons, model, sentence_list=get_sentences(n=n_sentences))

    # os.makedirs("large_internals", exist_ok=True)
    # # Save the dataframe
    # df.to_pickle(f"large_internals/s{n_sentences}_l{l_rng[0]}-{l_rng[1]}_{n_rng[0]}-{n_rng[1]}.pkl")

    # # Save the ndxs_dict
    # with open(f"large_internals/s{n_sentences}_l{l_rng[0]}-{l_rng[1]}_{n_rng[0]}-{n_rng[1]}_ndxs.pkl", "wb") as f:
    #     pickle.dump(ndxs_dict, f)
