import pickle 

import torch
import tqdm
from transformer_lens import HookedTransformer

from main import NdxTool

MODEL_ACTIVATIONS = "gpt2-small"
layer_n = 1
representation_penalty = 0.1
neurons_per_feature_penalty = 0.1

with open("sentencedata200.pkl", "rb") as f:
    sentence_lengths, sentence_fragments, embeddings = pickle.load(f)

ndx_tool = NdxTool(sentence_lengths, sentence_fragments) 
# Recover tyhe full sentences using the ndx_tool
full_sentences = ndx_tool.get_all_full_sentences()

mlp_dim = 768 * 4
total_dimensions = mlp_dim * 10

encoder = torch.nn.Linear(mlp_dim, total_dimensions)
decoder = torch.nn.Sequential(
    torch.nn.Linear(total_dimensions, mlp_dim),
    torch.nn.ReLU(),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained(MODEL_ACTIVATIONS, device=device)     

all_activations = []
for sentence in tqdm.tqdm(full_sentences):
    tokens = model.to_tokens(sentence)
    _, cache = model.run_with_cache(tokens, return_type=None, remove_batch_dim=True)
    activations = cache["post", layer_n, "mlp"]
    for activation in activations:
        all_activations.append(activation)

class ActivationDataset(torch.utils.data.Dataset):
    def __init__(self, activations):
        self.activations = activations

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, i):
        return self.activations[i]

dataset = ActivationDataset(all_activations)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

for epoch in range(1000):
    for batch in tqdm.tqdm(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        sparse_representation = encoder(batch)
        reconstruction = decoder(sparse_representation)
        loss = torch.nn.functional.mse_loss(reconstruction, batch)
        loss += representation_penalty * torch.norm(sparse_representation, p=1, dim=1).mean() # representation penalty
        loss += neurons_per_feature_penalty * torch.norm(sparse_representation, p=1, dim=0).mean() # neurons per feature penalty

        loss.backward()
        optimizer.step()

    print(loss.item())