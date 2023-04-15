import argparse
import copy
from dataclasses import dataclass
import json
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import openai
import pandas as pd
import pickle
import torch

from matplotlib import pyplot as plt

from transformer_lens import HookedTransformer

from datasets import load_dataset

from utils import pearsonr_ci

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
    wiki_dataset = load_dataset('wikitext', 'wikitext-103-v1')
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
    layer_n = 6 # Must be between 0 and 11 (inclusive)
    neuron_n = 0 # Must be between 0 and 3071 (inclusive)

    def compute_internals_single(model: HookedTransformer, input_txt: str) -> float:
        tokens = model.to_tokens(input_txt)
        _, cache = model.run_with_cache(tokens, return_type=None, remove_batch_dim=True)
        activations = cache["pre", layer_n, "mlp"]
        return activations[-1][neuron_n].tolist()

    return compute_internals_single

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
            model: HookedTransformer, 
            input_corpus: List[str], 
            model_internals_func: Callable[[HookedTransformer, str], float], # Need a more flexible version which gets activations for each token
            eval_func: Callable[[str, BinaryCondition], float],
            agreement_func: Callable[[List[float], List[float]], float],
            internals_loc: str,
            n_eval_trials = 5,
            load_internals: bool = True,
        ):
        print("Initializing ExplainGame")
        self.model = model
        self.input_corpus = input_corpus
        self.model_internals_func = model_internals_func
        self.eval_func = eval_func
        self.agreement_func = agreement_func

        self.n_eval_trials = n_eval_trials

        self.precompute_internals(internals_loc, load_internals)
        self.low_prompts, self.high_prompts = self.get_low_high_prompts()
        print("Done initializing ExplainGame")
    
    def precompute_internals(self, internals_loc: str, load_internals: bool):
        if load_internals and os.path.exists(internals_loc):
            print("Loading internals from file")
            with open(internals_loc, "rb") as f:
                self.internals = pickle.load(f)
            
            self.input_corpus = list(self.internals.keys())
            return
        
        print("Computing internals")
        self.internals = {}
        n_duplicates = 0
        for ndx, text in enumerate(self.input_corpus):
            if text not in self.internals:
                self.internals[text] = self.model_internals_func(self.model, text)
            else:
                n_duplicates += 1
            
            if ndx % 100 == 0:
                print(f"Computed internals for {ndx} of {len(self.input_corpus)}")

        print(f"Found {n_duplicates} duplicates")
        print(f"Saving internals to {internals_loc}")
        os.makedirs(os.path.dirname(internals_loc), exist_ok=True)
        with open(internals_loc, "wb") as f:
            pickle.dump(self.internals, f)

        print("Done computing internals")
        
    
    def evaluate_explanation(self, explanation: BinaryCondition, n_trials: Optional[int] = None) -> float:
        # Principle here is that if the explanation looks good initially then we should run on a larger set of examples
        # and if it looks bad then we should run on a smaller set of examples then ignore
    
        if n_trials is None:
            n_trials = self.n_eval_trials

        for _ in range(n_trials):
            high_text = random.choice(self.high_prompts)
            internal_val_h = self.internals[high_text]
            explanation_val_h = self.eval_func(high_text, explanation)
            explanation.add_datapoint((internal_val_h, explanation_val_h))

            low_text = random.choice(self.low_prompts)
            internal_val_l = self.internals[low_text]
            explanation_val_l = self.eval_func(low_text, explanation)
            explanation.add_datapoint((internal_val_l, explanation_val_l))

        explanation.update_score()
        return explanation.score
    
    def get_low_high_prompts(self, max_prompts: int = 2000) -> Tuple[List[str], List[str]]:
        # Get a list of prompts under which the model_internal_func returns a high value
        # by creating a structure which stores the top n values found so far and removing the lowest value if a new value is found which is higher
        
        sorted_vals = sorted(self.internals.items(), key=lambda x: x[1])
        total_above_zero = sum([1 for x in sorted_vals if x[1] > 0])
        n_high = min(total_above_zero // 2, max_prompts)
        high_vals = sorted_vals[-n_high:]
        print(f"Cutoff for high values is {high_vals[0][1]}, {len(high_vals)} values above this")
        high_prompts = [x[0] for x in high_vals]

        total_n_low = len(self.internals) - total_above_zero # Number of values below the cutoff
        all_low_vals = sorted_vals[:total_n_low] # All values below the cutoff
        n_low = min(total_n_low, max_prompts) # Number of values to use
        low_vals = random.sample(all_low_vals, n_low) # Sample n_low values
        low_prompts = [x[0] for x in low_vals]

        return low_prompts, high_prompts
    
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
        {"role": "user", "content": "You will first be shown a series of prompts under which a neuron in a langauge model is active, \
         and a series of prompts under which the neuron is inactive. You will then be shown zero or more binary conditions which are hypotheses as to what may explain this behaviour and a \
         score which judges how well the hypothesis explains the behaviour. For example \" The text ends in an incomplete word\", or \"The text is about sports.\". \
         You will then be asked to suggest an improved binary condition in a single sentence which explains the behaviour. Some things to remember:,\n\
         (1) this should try to explain the behaviour over all examples, not just one.\n\
         (2) the neuron will predict the next token, so it will often relate to what likely to come next, e.g. \"the next word will be preposition\"\
         (3) the condition can be negative as well as positive.\n\
         (4) if there are good suggestions, say (score>0.3), you could try small variations on that theme,\n\
         (5) we've tried many conditions, so don't be afraid to suggest something unusual."},
    ]

    messages.append({"role": "user", "content": "High prompts:"})
    if len(high_prompts) > 3:
        high_prompts = random.sample(high_prompts, 10)
    for prompt in high_prompts:
        messages.append({"role": "user", "content": prompt + "\n"})
    
    if len(low_prompts) > 3:
        low_prompts = random.sample(low_prompts, 10)
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

    
    def run_turn(self):
        new_condition = suggestion_generator(self.suggestions, self.game.high_prompts, self.game.low_prompts)
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
                correction = len(self.suggestions * 19) / len(bc.datapoints) # 10 for initial conditions, 9 for new conditions
                print(f"LCB: {bc.lcb:.3f}, Score: {bc.score:.3f}, P-val = {bc.p_val * correction:.6f}")
            print("\n")

            # Save the suggestions
            if not os.path.exists(os.path.dirname(self.save_path)):
                os.makedirs(os.path.dirname(self.save_path))

            with open(self.save_path, "w") as f:
                json.dump([bc.to_dict() for bc in self.suggestions], f)

def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--remake", type=bool, default=False)
    parser.add_argument("--load", type=bool, default=True)
    parser.add_argument("--n_turns", type=int, default=10)
    # parser.add_argument("--layer", type=int, default=-1)
    # parser.add_argument("--neuron", type=int, default=-1)

    args = parser.parse_args()

    # if args.neuron == -1:
    #     args.neuron = random.randint(0, 3071)
    # if args.layer == -1:
    #     args.layer = random.randint(0, 11)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    corpus = get_wiki_sentences()

    game_players: List[GamePlayer] = []
    neurons = list(range(3))
    layers = [5] * len(neurons)

    for layer, neuron in zip(layers, neurons):
        internals_func = make_internals_func(layer, neuron)
        internals_loc = os.path.join("internals", f"layer_{layer}", f"neuron_{neuron}.pkl")

        game = ExplainGame(
            model,
            corpus,
            model_internals_func=internals_func,
            eval_func=evaluate_prompt_single,
            agreement_func=score_condition_by_corr,
            internals_loc=internals_loc,
            load_internals= not args.remake
        )

        game_player = GamePlayer(game, layer, neuron, load=args.load)
        game_players.append(game_player)
    
    for turn_ndx in range(args.n_turns):
        turn_ndx += 1
        print(f"Turn {turn_ndx}")
        for game_player in game_players:
            game_player.run_turn()
        
        if turn_ndx % 3 == 0:
            results = []
            for game_player in game_players:
                top_condition = sorted(game_player.suggestions, key=lambda x: x.lcb, reverse=True)[0]
                results.append({
                    "layer": game_player.layer,
                    "neuron": game_player.neuron,
                    "condition": top_condition.condition,
                    "score": top_condition.score,
                    "lcb": top_condition.lcb,
                    "ucb": top_condition.ucb,
                    "p_val": top_condition.p_val,
                    "description": f"Layer: {game_player.layer}, Neuron: {game_player.neuron},\nCondition: '{top_condition.condition}'"
                })
            
            # Plot as a horizontal bar chart in descending order of score, with error bars
            df = pd.DataFrame(results)
            df = df.sort_values(by="score", ascending=True)

            # Show only those with score > 0 and up to 6 total
            df = df[df["score"] > 0]
            df = df[:6]

            # Make large figure
            fig, ax = plt.subplots(figsize=(10, 5), )

            # Colour the bars by the log of the p-value
            colours = np.log(df["p_val"])
            colours = (colours - colours.min()) / (colours.max() - colours.min())
            colours = plt.cm.viridis(colours)
            ax.barh(df["description"], df["score"], xerr=(df["score"] - df["lcb"], df["ucb"] - df["score"]), color=colours)

            ax.set_xlabel("Score")
            ax.set_ylabel("Neuron/condition")
            ax.set_title("Top conditions for each neuron")

            # Give lots of space for the descriptions
            # Make font smaller
            plt.rcParams.update({'font.size': 6})
            plt.subplots_adjust(left=0.5)
            plt.savefig(f"graphs/top_conditions_{turn_ndx}.png")





if __name__ == "__main__":
    main()