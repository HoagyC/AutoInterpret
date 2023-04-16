import argparse
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
    ("This piece of text is about sports", "The baseball team won the", "YES"),
    ("The final word of the prompt is \"and\"", "Canada is a member of the International", "NO"),
]

def make_base_message(examples: List[Tuple[str, str, str]] = []) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please respond simply with YES or NO, answering whether the given piece of text meets the given binary condition."}
    ]
    if examples:
        for binary_condition, text, answer in examples:
            messages.append({"role": "user", "content": f"Binary condition: '{binary_condition}' Text: '{text}' Answer:"})
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


def get_wiki_sentences(n: int = 5000) -> List[str]:
    sentence_list: List[str] = []
    wiki_dataset = load_dataset("NeelNanda/pile-10k")
    while len(sentence_list) < n:
        sentence = wiki_dataset["train"][random.randint(0, len(wiki_dataset["train"]))]["text"]
        # Cut off after a maximum of 20 words
        sentence = " ".join(sentence.split(" ")[:20])
        # If it contains non-ascii characters, skip it
        if not all(ord(c) < 128 for c in sentence) or len(sentence) < 20:
            continue
        sentence_list.append(sentence)

    return sentence_list

def make_internals_func(layer_n: int, neuron_n: int):
    def compute_internals_single(model: HookedTransformer, input_txt: str) -> float:
        tokens = model.to_tokens(input_txt, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, return_type=None, remove_batch_dim=True)
        activations = cache["pre", layer_n, "mlp"]
        return activations[-1][neuron_n].tolist()

    return compute_internals_single

def make_all_internals(
        layer_ndxs: List[int], 
        neuron_ndxs: List[int], 
        model: HookedTransformer, 
        sentence_list: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[int]]]]:
    # Make dataframe with a column for each layer and neuron, with an entry for each full and partially tokenized sentenc


    all_internals = []
    print(f"Computing internals for all {len(sentence_list)} sentences")
    for sentence in tqdm.tqdm(sentence_list):
        tokens = model.to_tokens(sentence)
        _, cache = model.run_with_cache(tokens, return_type=None, remove_batch_dim=True)
        for i in range(tokens.shape[1]):
            partial_sentence = model.to_string(tokens[0,:i + 1])
            partial_str_dict: Dict[str, Any] = {}
            partial_str_dict["context"] = partial_sentence
            partial_str_dict["pre_tokens"] = tokens[0, :i + 1].tolist()
            partial_str_dict["full_tokens"] = tokens[0, :].tolist()
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
    n_ndxs = 1000
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
        input_txt: str, 
        binary_condition: BinaryCondition,
    ) -> float:

    message = copy.deepcopy(base_message)
    response_status = -1
    while response_status == -1:
        try:
            message.append({"role": "user", "content": f"Binary condition: '{binary_condition.condition}' Text: '{input_txt}' Answer:"})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = message,
                max_tokens=2,
            )

        except openai.error.RateLimitError:
            print("OpenAI servers overloaded, waiting 1 second")
            time.sleep(1)
            continue

        response_str = response["choices"][0]["message"]["content"]
        if response_str[:3] == "YES":
            return 1
        elif response_str[:2] == "NO":
            return 0
        else:
            print(f"Non-compliant response: '{response_str}'")
            return -1
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
            high_ndxs: List[int],
            low_ndxs: List[int],
            eval_func: Callable[[str, BinaryCondition], float],
            agreement_func: Callable[[List[float], List[float]], float],
            n_eval_trials = 5,
        ):
        print("Initializing ExplainGame")
        self.internals_fn = internals_fn
        self.eval_func = eval_func
        self.agreement_func = agreement_func

        self.high_ndxs = high_ndxs
        self.low_ndxs = low_ndxs

        self.n_eval_trials = n_eval_trials
        print("Done initializing ExplainGame")
    

    def evaluate_explanation(self, explanation: BinaryCondition, n_trials: Optional[int] = None) -> float:
        # Principle here is that if the explanation looks good initially then we should run on a larger set of examples
        # and if it looks bad then we should run on a smaller set of examples then ignore
    
        if n_trials is None:
            n_trials = self.n_eval_trials

        for _ in range(n_trials):
            high_text_ndx = random.choice(self.high_ndxs)
            high_text, high_int_val = self.internals_fn(high_text_ndx)
            explanation_val_h = self.eval_func(high_text, explanation)
            explanation.add_datapoint((high_int_val, explanation_val_h))

            low_text_ndx = random.choice(self.low_ndxs)
            low_text, low_int_val = self.internals_fn(low_text_ndx)
            explanation_val_l = self.eval_func(low_text, explanation)
            explanation.add_datapoint((low_int_val, explanation_val_l))

        explanation.update_score()
        return explanation.score
    
    def get_high_context(self, n: int = 10) -> List[str]:
        ndxs = random.sample(list(self.high_ndxs), n)
        contexts = []
        for ndx in ndxs:
            context, _ = self.internals_fn(ndx)
            contexts.append(context)
        
        return contexts

    def get_low_context(self, n: int = 10) -> List[str]:
        ndxs = random.sample(list(self.low_ndxs), n)
        contexts = []
        for ndx in ndxs:
            context, _ = self.internals_fn(ndx)
            contexts.append(context)
        
        return contexts

    
    def evaluate_high_ucbs(self, bc_list: List[BinaryCondition], n_iters: int = 10):
        # Evaluate the high ucb conditions
        print("Evaluating high ucb conditions")
        iter = 0
        while iter < n_iters:
            iter += 1
            bc_list = sorted(bc_list, key=lambda x: x.ucb, reverse=True)
            top_bc = bc_list[0]
            self.evaluate_explanation(top_bc, n_trials=2)
        

def suggestion_generator(
    scored_conditions: List[BinaryCondition],
    high_prompts: List[str],
    low_prompts: List[str],
) -> BinaryCondition:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    initial_instructions = "You will first be shown a series of prompts under which a neuron in a langauge model is active, " + \
        "and a series of prompts under which the neuron is inactive. You will then be shown zero or more binary conditions which are hypotheses as to what may explain this behaviour and a " + \
        "score which judges how well the hypothesis explains the behaviour. For example \"The text ends in an incomplete word\", or \"The text is about sports.\". " + \
        "You will then be asked to suggest an improved binary condition in a single sentence which explains the behaviour. Some things to remember:\n" + \
        "(1) this should try to explain the behaviour over all examples, not just one.\n" + \
        "(2) the neuron will predict the next token, so it will often relate to what likely to come next, e.g. \"the next word will be preposition\"\n" + \
        "(3) the condition can be negative as well as positive.\n" + \
        "(4) if there are good suggestions, say (score>0.3), you could try small variations on that theme,\n" + \
        "(5) we've tried many conditions, so don't be afraid to suggest something unusual."
    
    messages.append({"role": "user", "content": initial_instructions})
    messages.append({"role": "user", "content": "High prompts:"})
    for prompt in high_prompts:
        messages.append({"role": "user", "content": prompt + "\n"})
    
    messages.append({"role": "user", "content": "Low prompts:"})
    for prompt in low_prompts:
        messages.append({"role": "user", "content": prompt + "\n"})

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
        messages.append({"role": "user", "content": f"Condition {ndx}: {bc.score}"})

    target_score = max(raw_scores) + 0.1 if len(raw_scores) > 0 else 0.0
    messages.append({"role": "user", "content": f"Score of new condition: {target_score}"})
    messages.append({"role": "user", "content": "New condition:"})

    all_conditions = [sc.condition for sc in scored_conditions]
    found_new_condition = False

    while not found_new_condition:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = messages,
        )

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
        self.save_path = os.path.join("found_conditions", f"layer_{self.layer}", f"neuron_{self.neuron}.json")
        if os.path.exists(self.save_path) and load:
            with open(self.save_path, "r") as f:
                self.suggestions = [BinaryCondition.from_dict(bc) for bc in json.load(f)]
        else:
            self.suggestions = []


    def run_turn(self) -> Tuple[str, List[float]]:
        # Get a few low and high suggestions
        low_contexts = self.game.get_low_context(n=3)
        high_contexts = self.game.get_high_context(n=3)
        new_condition = suggestion_generator(self.suggestions, high_contexts, low_contexts)
        print(f"New suggested condition: {new_condition.condition}")
        init_score = self.game.evaluate_explanation(new_condition)
        print(f"Initial score of condition: {init_score}")
        self.suggestions.append(new_condition)
        self.game.evaluate_high_ucbs(self.suggestions, n_iters=3)
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
                correction = len(self.suggestions * 22) / len(bc.datapoints) # 10 for initial conditions, 9 for new conditions
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

    model_size = "large"

    game_players: List[GamePlayer] = []

    n_rng = (800, 900)
    l_rng = (29, 31)
    layers = list(range(*l_rng))
    neurons = list(range(*n_rng))
    n_sentences = 5000
    
    # Load the internals
    internals_str = f"{model_size}_internals/s{n_sentences}_l{l_rng[0]}-{l_rng[1]}_n{n_rng[0]}-{n_rng[1]}"
    full_internals_loc = internals_str + ".pkl"
    internals_ndxs_loc = internals_str + "_ndxs.pkl"
    
    if not os.path.exists(full_internals_loc) or not os.path.exists(internals_ndxs_loc):
        # loaded = download_from_aws([full_internals_loc, internals_ndxs_loc])
        # print("Loaded from AWS: ", loaded)
        # if not loaded:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = HookedTransformer.from_pretrained(f"gpt2-{model_size}", device=device)            
        make_all_internals(layers, neurons, model, get_wiki_sentences(n=n_sentences))
    
    with open(full_internals_loc, "rb") as f:
        internals = pickle.load(f)
    
    with open(internals_ndxs_loc, "rb") as f:
        internals_ndxs = pickle.load(f)

    for layer, neuron in list(itertools.product(layers, neurons))[90:100]:
        internals_fn = make_db_internals_fn(layer, neuron, internals)
        ndxs_dict = internals_ndxs[f"l{layer}n{neuron}"]
        game = ExplainGame(
            internals_fn=internals_fn,
            high_ndxs=ndxs_dict["top"],
            low_ndxs=ndxs_dict["neg"],
            eval_func=evaluate_prompt_single,
            agreement_func=score_condition_by_corr,
        )

        game_player = GamePlayer(game, layer, neuron, load=args.load)
        game_players.append(game_player)


    for turn_ndx in range(args.n_turns):
        turn_ndx += 1
        print(f"Turn {turn_ndx}")
        for game_player in game_players:
            new_condition, yesno_list = game_player.run_turn()
        

        if turn_ndx % 5 == 0:
            results = []
            for game_player in game_players:
                top_condition = sorted(game_player.suggestions, key=lambda x: x.lcb, reverse=True)[0]
                p_val_correction = len(game_player.suggestions) * len(game_players) * (22 / len(top_condition.datapoints))

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
            plt.savefig(f"graphs/top_conditions_{n_conditions}.png")


if __name__ == "__main__":
    main()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = HookedTransformer.from_pretrained("gpt2-large", device=device)

    # n_sentences = 5000
    # l_rng = (29, 31)
    # n_rng = (800, 900)
    # layers = list(range(*l_rng))
    # neurons = list(range(*n_rng))

    # df, ndxs_dict = make_all_internals(layers, neurons, model, sentence_list=get_wiki_sentences(n=n_sentences))

    # os.makedirs("large_internals", exist_ok=True)
    # # Save the dataframe
    # df.to_pickle(f"large_internals/s{n_sentences}_l{l_rng[0]}-{l_rng[1]}_{n_rng[0]}-{n_rng[1]}.pkl")

    # # Save the ndxs_dict
    # with open(f"large_internals/s{n_sentences}_l{l_rng[0]}-{l_rng[1]}_{n_rng[0]}-{n_rng[1]}_ndxs.pkl", "wb") as f:
    #     pickle.dump(ndxs_dict, f)
