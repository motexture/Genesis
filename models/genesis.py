import torch
import random
import torch.nn.functional as F

from torch import nn
from transformers import AutoTokenizer
from .config import GenesisConfig
from dataclasses import dataclass
from typing import Optional
from transformers.utils import ModelOutput

@dataclass
class GenesisOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    outputs: torch.FloatTensor = None
    neuron_activation_count: int = None

class Genesis(nn.Module):
    def __init__(self, config: GenesisConfig, tokenizer: AutoTokenizer = None):
        super(Genesis, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.n_dim = self.config.n_dim
        self.n_inner_dim =  self.n_dim // self.config.n_sections if self.config.n_section == "split" else self.n_dim
        
        self.input = None
        if self.config.n_input == "reading":
            self.input = Reading(self.config.n_dim, self.config.n_vocab_size, self.config.n_pos_size)
        elif self.config.n_input == "bridge":
            self.input = Bridge(self.config.n_dim)

        self.sections = nn.ModuleList([
            Section(self.n_inner_dim, self.config.n_neurons, self.config.n_attentive_neurons, self.config.n_attentive_neuron_heads, self.config.n_dendritics, self.config.n_synapses, self.config.n_pos_size, self.config.n_neighbors) for _ in range(self.config.n_sections)
        ])
        if self.config.n_section == "split":
            self.a_splits = nn.ModuleList([
                nn.Linear(self.n_dim, self.n_inner_dim) for _ in range(self.config.n_sections)
            ])
            self.combiner = Combiner(self.config.n_dim)

            self.c_splits = None
            if self.config.n_cross_attention:
                self.c_splits = nn.ModuleList([
                    nn.Linear(self.n_dim, self.n_inner_dim) for _ in range(self.config.n_sections)
                ])
        
        self.output = None
        if self.config.n_output == "causal_lm":
            self.output = CausalLM(self.config.n_dim, self.config.n_vocab_size)
        elif self.config.n_output == "sequence_classifier":
            self.output = SequenceClassifier(self.config.n_dim, self.config.n_outputs)
        elif self.config.n_output == "bridge":
            self.output = Bridge(self.config.n_dim)

    def __repr__(self):
        return f"Genesis(input={self.input}, sections={self.sections}, output={self.output})"

    def assign_neighbors(self):
        random.seed(self.config.n_seed)
        for k, section in enumerate(self.sections):
            for z, neuron in enumerate(section.neurons):
                potential_neighbors = [n for j, n in enumerate(section.neurons) if j != z]
                neuron.set_neighbors(random.sample(potential_neighbors, self.config.n_neighbors))

    def forward(self, inputs, attention_mask=None, y=None, labels=None):
        total_neuron_activation_count = 0

        if self.input:
            hidden_states = self.input(inputs)
        else:
            hidden_states = inputs

        if self.config.n_section == "split":
            splitted = []
            for i in range(len(self.sections)):
                a_split = self.a_splits[i](hidden_states)
                if y is not None and self.config.n_cross_attention:
                    c_split = self.c_splits[i](y)
                else:
                    c_split = None
                splitted_hidden_states, neuron_activation_count = self.sections[i](a_split, attention_mask, c_split)
                total_neuron_activation_count += neuron_activation_count
                splitted.append(splitted_hidden_states)
            hidden_states = self.combiner(torch.cat(splitted, dim=-1))
        else:
            for section in self.sections:
                hidden_states, neuron_activation_count = section(hidden_states, attention_mask, y)
                total_neuron_activation_count += neuron_activation_count

        if self.output:
            outputs = self.output(hidden_states)
        else:
            outputs = hidden_states

        loss = None
        if labels is not None:
            if self.config.n_output == "causal_lm":
                labels = labels.to(outputs.device)

                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            elif self.config.n_output == "sequence_classifier":
                labels = labels.to(outputs.device)

                if self.config.n_outputs == 1:
                    loss_fct = nn.BCELoss()
                    outputs = outputs.squeeze(-1)
                    loss = loss_fct(outputs, labels.float())
                else:
                    loss_fct = nn.NLLLoss()
                    outputs = outputs.squeeze(-1)
                    loss = loss_fct(outputs, labels)

        return GenesisOutput(
            loss=loss,
            outputs=outputs,
            neuron_activation_count=total_neuron_activation_count
        )

    def build_instruction_prompt(self, prompt):
        return f"### Instruction:\n\n{prompt.strip()}\n\n### Response: \n\n"

    def generate(self, prompt, device, max_length=128, temperature=0.9, top_p=0.9, build_instruction=True, print_prompt=False):
        prompt = self.build_instruction_prompt(prompt) if build_instruction else prompt
        if print_prompt:
            print(prompt, end="")

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            for _ in range(max_length):
                output = self.forward(input_ids)
                logits = output.outputs[:, -1, :] / temperature

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = torch.finfo(torch.float32).min

                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

                if next_token == self.tokenizer.eos_token:
                    break
                    
                print(self.tokenizer.decode(next_token[0], return_tensors='pt'), end='')

                input_ids = torch.cat([input_ids, next_token], dim=-1)

class Section(nn.Module):
    def __init__(self, n_dim, n_neurons, n_attentive_neurons, n_attentive_neuron_heads, n_dendritics, n_synapses, n_pos_size, n_neighbors):
        super(Section, self).__init__()
        self.n_dim = n_dim
        self.n_neurons = n_neurons
        self.n_attentive_neurons = n_attentive_neurons
        self.n_attentive_neuron_heads = n_attentive_neuron_heads
        self.n_dendritics = n_dendritics
        self.n_synapses = n_synapses
        self.n_pos_size = n_pos_size
        self.n_neighbors = n_neighbors
        
        self.neurons = nn.ModuleList([
            Neuron(self.n_dim, True, self.n_attentive_neuron_heads, self.n_dendritics, self.n_synapses, self.n_pos_size, self.n_neighbors) for _ in range(self.n_attentive_neurons)
        ])
        self.neurons += nn.ModuleList([
            Neuron(self.n_dim, False, self.n_attentive_neuron_heads, self.n_dendritics, self.n_synapses, self.n_pos_size, self.n_neighbors) for _ in range(self.n_neurons - self.n_attentive_neurons)
        ])
        
        self.indexer = Indexer(self.n_dim, self.neurons, self.n_attentive_neurons)

        self.neuron_activation_count = 0

    def forward(self, x, attention_mask=None, y=None):
        self.neuron_activation_count = 0

        neuron = self.indexer(x)     
        x = neuron.forward(x, lambda: self.increment_neuron_count(), attention_mask, y)

        self.reset_neurons()

        return x, self.neuron_activation_count
    
    def increment_neuron_count(self):
        self.neuron_activation_count += 1

    def get_neuron_activation_count(self):
        return self.neuron_activation_count
    
    def reset_neurons(self):
        for neuron in self.neurons:
            neuron.reset()

    def __repr__(self):
        return f"Section(neurons={self.neurons})"

class Reading(nn.Module):
    def __init__(self, n_embed_size, n_vocab_size, n_pos_size):
        super(Reading, self).__init__()
        self.n_embed_size = n_embed_size
        self.n_vocab_size = n_vocab_size
        self.n_pos_size = n_pos_size
        
        self.wte = nn.Embedding(self.n_vocab_size, self.n_embed_size)
        self.wpe = nn.Embedding(self.n_pos_size, self.n_embed_size)

        self.fc_out = nn.Linear(self.n_embed_size, self.n_embed_size)
        self.ln = nn.LayerNorm(self.n_embed_size)
        self.act = nn.SiLU()

    def forward(self, input_ids):
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.ln(hidden_states)
        hidden_states = self.act(hidden_states)
        
        return hidden_states
    
    def __repr__(self):
        return f"Reading(n_dim={self.n_embed_size}, n_vocab_size={self.n_vocab_size}, n_pos_size={self.n_pos_size})"
    
class CausalLM(nn.Module):
    def __init__(self, n_dim, n_vocab_size):
        super(CausalLM, self).__init__()
        self.n_dim = n_dim
        self.n_vocab_size = n_vocab_size
        
        self.lm_head = nn.Linear(self.n_dim, self.n_vocab_size)

    def forward(self, x):
        return self.lm_head(x)
    
    def __repr__(self):
        return f"Writing(n_dim={self.n_dim}, n_vocab_size={self.n_vocab_size})"
    
class SequenceClassifier(nn.Module):
    def __init__(self, n_dim, n_outputs):
        super(SequenceClassifier, self).__init__()
        self.n_dim = n_dim
        self.n_outputs = n_outputs
        
        self.fc_out = nn.Linear(self.n_dim, self.n_outputs)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.fc_out(x)
        x = torch.softmax(x, dim=-1) if self.n_outputs > 1 else torch.sigmoid(x)

        return x
    
class Combiner(nn.Module):
    def __init__(self, n_dim):
        super(Combiner, self).__init__()
        self.n_dim = n_dim
        
        self.fc_out = nn.Linear(self.n_dim, self.n_dim)
        self.ln = nn.LayerNorm(self.n_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.fc_out(x)
        x = self.ln(x)
        x = self.act(x)

        return x
    
    def __repr__(self):
        return f"Combiner(n_dim={self.n_dim}"
    
class Bridge(nn.Module):
    def __init__(self, n_dim):
        super(Bridge, self).__init__()
        self.n_dim = n_dim
        
        self.fc_out = nn.Linear(self.n_dim, self.n_dim)
        self.ln = nn.LayerNorm(self.n_dim)

    def forward(self, x):
        x = self.fc_out(x)
        x = self.ln(x)

        return x
    
    def __repr__(self):
        return f"Bridge(n_dim={self.n_dim}"
    
class Indexer(nn.Module):
    def __init__(self, n_dim, neurons, n_attentive_neurons):
        super(Indexer, self).__init__()
        self.n_dim = n_dim
        self.neurons = neurons
        self.n_attentive_neurons = n_attentive_neurons

    def forward(self, x):
        targets = []
        for neuron in self.neurons:
            if not neuron.attentive_neuron and self.n_attentive_neurons > 0:
                continue
            voltages, means = neuron.interact(x)
            targets.append((neuron, voltages, means))

        if targets:
            neuron, _, _ = max(targets, key=lambda item: item[-1])
            return neuron

        return random.choice(self.neurons)
    
    def __repr__(self):
        return f"Indexer()"
    
class Dendritic(nn.Module):
    def __init__(self, n_dim):
        super(Dendritic, self).__init__()
        self.n_dim = n_dim

        self.fc_out = nn.Linear(self.n_dim, self.n_dim, bias=False)
        self.ln = nn.LayerNorm(self.n_dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_out(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.dropout(x)

        return x
    
class Soma(nn.Module):
    def __init__(self, n_dim):
        super(Soma, self).__init__()
        self.n_dim = n_dim

        self.fc_out = nn.Linear(self.n_dim, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.fc_out(x)
        x = self.act(x)

        return x, torch.mean(x, dim=0)

class Axon(nn.Module):
    def __init__(self, n_dim, attentive_neuron=False, heads=8):
        super(Axon, self).__init__()
        self.n_dim = n_dim
        self.attentive_neuron = attentive_neuron
        self.heads = heads

        if self.attentive_neuron:
            self.to_q = nn.Linear(self.n_dim, self.n_dim, bias=False)
            self.to_k = nn.Linear(self.n_dim, self.n_dim, bias=False)
            self.to_v = nn.Linear(self.n_dim, self.n_dim, bias=False)

        self.fc_out = nn.Linear(self.n_dim, self.n_dim, bias=False)
        self.ln = nn.LayerNorm(self.n_dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None, y=None):
        if self.attentive_neuron:
            y = x if y is None else y
            
            x_shape = x.shape
            y_shape = y.shape

            q = self.to_q(x)
            k = self.to_k(y)
            v = self.to_v(y)

            q = q.reshape(x_shape[0], x_shape[1], self.heads, x_shape[2] // self.heads).transpose(1, 2)
            k = k.reshape(y_shape[0], y_shape[1], self.heads, y_shape[2] // self.heads).transpose(1, 2)
            v = v.reshape(y_shape[0], y_shape[1], self.heads, y_shape[2] // self.heads).transpose(1, 2)

            is_causal = True if attention_mask is not None else False
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=is_causal)
            x = x.transpose(1, 2).contiguous().view(x_shape[0], x_shape[1], x_shape[2])

        x = self.fc_out(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.dropout(x)

        return x
    
class Synapse(nn.Module):
    def __init__(self, n_dim):
        super(Synapse, self).__init__()
        self.n_dim = n_dim

        self.fc_out = nn.Linear(self.n_dim, self.n_dim, bias=False)
        self.ln = nn.LayerNorm(self.n_dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_out(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.dropout(x)

        return x
    
class Neuron(nn.Module):
    def __init__(self, n_dim, attentive_neuron, n_attentive_neuron_heads, n_dendritics, n_synapses, n_pos_size, n_neighbors):
        super(Neuron, self).__init__()
        self.n_dim = n_dim
        self.attentive_neuron = attentive_neuron
        self.n_attentive_neuron_heads = n_attentive_neuron_heads
        self.n_dendritics = n_dendritics
        self.n_synapses = n_synapses
        self.n_pos_size = n_pos_size
        self.n_neighbors = n_neighbors

        self.n_threshold = nn.Parameter(torch.tensor(0.5))
        self.activated = False 

        self.dendritics = nn.ModuleList([
            Dendritic(self.n_dim) for _ in range(self.n_dendritics)
        ])
        self.soma = Soma(self.n_dim)
        self.axon = Axon(self.n_dim, self.attentive_neuron, self.n_attentive_neuron_heads)
        self.synapses = nn.ModuleList([
            Synapse(self.n_dim) for _ in range(self.n_synapses)
        ])
        self.ln = nn.LayerNorm(self.n_dim, bias=False)
        
        self.neighbors = []

    def __repr__(self):
        return f"Neuron(dendritics={self.dendritics}, soma={self.soma}, axon={self.axon}, synapses={self.synapses})"

    def reset(self):
        self.activated = False

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors
    
    def interact(self, x):
        hidden_states = x
        for dendritic in self.dendritics:
            hidden_states = dendritic(hidden_states)

        return self.soma(hidden_states)
    
    def process(self, x, attention_mask=None, y=None):
        residual = x
        x = self.axon(x, attention_mask, y)
        x = x + residual

        return x

    def propagate(self, x):
        residual = x
        for synapse in self.synapses:
            x = synapse(x)
        x = x + residual
        
        return x
    
    def connect(self, x, increment_neuron_count_func):
        hidden_states = x

        targets = []
        for neighbor in self.neighbors:
            if not neighbor.activated and not neighbor.attentive_neuron:
                voltages, means = neighbor.interact(hidden_states)
                targets.append((neighbor, voltages, means))

        if targets:
            neighbor, voltages, _ = max(targets, key=lambda item: item[-1])
            mask = (voltages > neighbor.n_threshold).squeeze()

            if mask.any():
                masked_hidden_states = hidden_states.clone()
                masked_hidden_states[~mask] = 0
                processed_hidden_states = neighbor.forward(masked_hidden_states, increment_neuron_count_func)
                processed_hidden_states = self.ln(processed_hidden_states)
                masked_hidden_states = masked_hidden_states + processed_hidden_states
                hidden_states[mask] = masked_hidden_states[mask]

        return hidden_states
            
    def forward(self, x, increment_neuron_count_func, attention_mask=None, y=None):
        if self.activated:
            return x
        self.activated = True
        increment_neuron_count_func()

        x = self.process(x, attention_mask, y)
        x = self.propagate(x)
        x = self.connect(x, increment_neuron_count_func)

        return x
