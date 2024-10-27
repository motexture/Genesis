import json
import os

class GenesisConfig:
    def __init__(self, n_dim=1280, n_input="reading", n_section="split", n_sections=2, n_neurons=16, n_attentive_neurons=4, n_attentive_neuron_heads=16, n_cross_attention=False, n_dendritics=2, n_synapses=1, n_vocab_size=50257, n_pos_size=2048, n_neighbors=4, n_output="causal_lm", n_outputs=1, n_seed=42):
        self.n_dim = n_dim
        self.n_input = n_input
        self.n_section = n_section
        self.n_sections = n_sections
        self.n_neurons = n_neurons
        self.n_attentive_neurons = n_attentive_neurons
        self.n_attentive_neuron_heads = n_attentive_neuron_heads
        self.n_cross_attention = n_cross_attention
        self.n_dendritics = n_dendritics
        self.n_synapses = n_synapses
        self.n_vocab_size = n_vocab_size
        self.n_pos_size = n_pos_size
        self.n_neighbors = n_neighbors
        self.n_output = n_output
        self.n_outputs = n_outputs
        self.n_seed = n_seed

    def save_to_json(self, file_path):
        with open(os.path.join(file_path, "config.json"), 'w') as file:
            json.dump(self.__dict__, file, indent=4)

    @classmethod
    def load_from_json(cls, file_path):
        try:
            with open(os.path.join(file_path, "config.json"), 'r') as file:
                config_dict = json.load(file)
            return cls(**config_dict)
        except:
            return cls()

    def __repr__(self):
        return f"GenesisConfig(n_dim={self.n_dim}, n_sections={self.n_sections}, n_neurons={self.n_neurons}, n_attentive_neurons={self.n_attentive_neurons}, n_attentive_neuron_heads={self.n_attentive_neuron_heads}, n_cross_attention={self.n_cross_attention}, n_dentritics={self.n_dendritics}, n_vocab_size={self.n_vocab_size}, n_pos_size={self.n_pos_size}, n_synapses={self.n_synapses}, n_neighbors={self.n_neighbors})"