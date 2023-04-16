# AutoInterpret

**Hoagy Cunningham and Misha Wagner**

## Summary

In this project we used OpenAI's `gpt-3.5-turbo` model to build a system which autonomously proposes hypotheses for what a neuron in GPT2-large might 'mean', and evaluates them, repeating this process to find the best possible hypothesis. Our hypotheses are binary conditions over the recent context window of the transformer, and we evaluate the YES/NO answers of this condition by correlation with the neurons's activation.

We find that our current system is able to generate hypotheses for a reasonable fraction where the automatic evaluation of the binary condition of the text has a significant correlation with the neuron's activation, when (we hope correctly!) correcting for the prescence of many hypotheses.

![](./public_graphs/top_conditions_40.png)

Nonetheless the hypotheses generated are usually bland and not particularly informative. We hypothesize that this is because our current strategy of understanding neurons simply in terms of when they are on and off is too simplistic and and the next major step (along with many minor ones) should be to enable the system to suggest and evaluate richer forms of hypothesis about how a model is working.

## Our Work

Previous work has shown that LLMs are capable of finding unexpected connections, and doing labelling of models. We want to combine these capabilities into a single system, so that the system can take in various datapoints around an aspect of the models internals, propose a hypothesis for what this part of the model might mean.

When permanently running, the system consumes about 1.4M tokens in requests to the server by hour, at least on my laptop. `gpt-3.5-turbo` costs $1 per 500K tokens, so this system costs about $3 an hour, in which time it generates and evaluates a few hundred hypotheses. This could be made much more efficient, but part of the idea here is to probe what can be done if basic intellectual tasks can be treated as essentially free.

## Running the system

To run this system you will need quite a few GB of memory and RAM to download GPT-2-Large and run the activations. If you don't have access to a GPU or powerful computer, set `model_size="small"` in main.py and set `n_sentences` to something smaller if getting the activations is slow.

To run you will also need an OpenAI paid account key saved in `secrets.json` like so:

```
{
    "openai_key": "sk-YOUR-KEY-HERE",
}
```

Then run the following to get the system running. 
```
python -m venv .env 
pip install -r requirements.txt
python main.py
```

## Trying to find the 'an' neuron

One of our goals for this system, and the main reason we used GPT2-large in particular, was that we hoped (and still do!) that this kind of system could in theory independently rediscover the main result in [We Found An Neuron in GPT-2](https://www.lesswrong.com/posts/cgqh99SHsCv3jJYDS/we-found-an-neuron-in-gpt-2), where they found a neuron that 

What we found was that we could replicate the results on contrived examples - neuron 892 of layer 29 was usually more active on sentences like `'A pear fell from a pear tree, a {fruit} fell from..'`, when the fruit in question began with a vowel. 

However, when we took contexts which led to high activations of that neuron, the pattern was barely noticeable, and would be impossible for either us or GPT-4 to see any connection.

We're still not sure if this is because we didn't use a large enough corpus to find examples where 'an' was highly likely, or because our method of getting sentences was confused or wrong somehow, but we think it highlights the difficultly of locating the meaning of a neuron from just the contexts in which it is active.

## Theoretical Description
The particular system here is a single, rather slow and weak example of a wider pattern of turning interpretability into a game, based on the strategy of 

The basic pattern is the following:

- the model one wishes to understand (in this case GPT2-Large), $M$ 
- a set of inputs $i\in I$ which forms the domain on which we want to understand the model's behaviour
- some function of the models behaviour which we think represents some important component of understanding of the models behaviour (e.g. the activations of particular neurons, or directions in activation space) $f: (M, i) \rightarrow \mathbb{R}^n$
- a function (model) for evaluating whether a condition $C$ is met by input $i$, $g: (C, i)\rightarrow \mathbb{R}$ , (e.g. few-shot prompting gpt-3.5-turbo with "does text input {text} meet binary condition {bin_con}?")
- an agreement function $h: (f(M, i), g(C, i)) \rightarrow \mathbb{R}$ (e.g. the correlation between the responses (boolean interpreted as an int) and the activation float). 

## Building on this work

We'd greatly welcome anyone who wants to collaborate or fork off from this project. 

The most important form of improvement is, I think, more sophisticated understanding of what it means to understand a model, looking at directions, including model patching. 

I think we're only scratching the surface of the kind of models, and in fact as the nature of 'evaluating a hypothesis about a model' becomes more involved and complex, the payoffs of automating that evaluation will only grow.

There are also more limited imporvements which are possible such as:

- Better suggestion generation, such as by:
    - Using stronger models
    - Fine-tuning the models, or using RL on the scores
    - Better prompting of the existing models, especially regarding variation in suggestions - it can be very repetitive.
    - Giving the model better information, including:
        - Showing the token that the model eventually predicted, or the true completion, or the activation throughout a sequence of tokens
        - Showing sentences that have high semantic similarity to cases where the activation is high, but where the activation is low.
- Better scoring, such as by:
    - Better understand of how to handle negative activations with the GELU function
    - Using a flexible binary threshold for neuron being 'on' instead of using correlation.

Basically, I hope that where there is theoretical progress about how we should understand what a model is doing, and the kind of tests that would confirm this, we are able to 
