import torch
import torch.nn.functional as F
import torch.nn as nn

class MetanetworkTransformer(nn.Module):
    def __init__(self, tar_model_layer, num_memtoken, in_size, cfg):
        super().__init__()
        self.tar_model_layer = tar_model_layer
        self.num_mem_token = num_mem_token
        self.in_size = in_size

        self.layer_pe = nn.Parameter(torch.zeros((self.tar_model_layer, self.in_size)), requires_grad=True)
        self.token_pe = nn.Parameter(torch.zeros((self.num_mem_token, self.in_size)), requires_grad=True)
        
        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(in_size, 16, dim_feedforward=2*in_size, dropout=0.0, activation=nn.SiLU(), batch_first=True, norm_first=True, bias=True) for _ in range(4)])


    def forward(self, memory_states:torch.Tensor) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        memory_states = memory_states + self.layer_pe.unsqueeze(-2) + self.token_pe # apply PE
        batch_size = memory_states.shape[0]

        memory_states = self.transformer_layers[0](memory_states.transpose(1, 2).flatten(0, 1)).unflatten(0, (batch_size, self.num_mem_token)).transpose(1, 2) # exchange information among layers
        
        memory_states = self.transformer_layers[1](memory_states.flatten(0, 1)).unflatten(0, (batch_size, self.tar_model_layer)) # exchange information among tokens

        memory_states = self.transformer_layers[2](memory_states.transpose(1, 2).flatten(0, 1)).unflatten(0, (batch_size, self.num_mem_token)).transpose(1, 2) # exchange information among layers

        memory_states = self.transformer_layers[3](memory_states.flatten(0, 1)).unflatten(0, (batch_size, self.tar_model_layer)) # exchange information among tokens
        return memory_states

class Metanetwork(nn.Module):
    def __init__(self, lora_model:nn.Module, cfg):
        super().__init__()
        self.tarmodel_layer = len(lora_model.model.layers)
        self.in_size = cfg.hidden_size
        self.lora_r = cfg.model.lora_r
        self.output_dim = lora_model.lora_params_numel(self.lora_r)
        self.lora_model = lora_model

        assert cfg.num_mem_token * cfg.in_size * self.tarmodel_layer == self.output_dim, f"num_memtoken should = {self.output_dim/(cfg.in_size * self.tarmodel_layer)}"

        if cfg.metanetwork.type == "transformer":
            self.metanetwork = MetanetworkTransformer(self.output_dim, cfg)
            self.scale = cfg.metanetwork.transformer_cfg.scale
        else:
            raise ValueError(f"Unknown metanetwork type: {cfg.metanetwork.type}")

    def forward(self, memory_states:torch.Tensor) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        plain_output = self.metanetwork(memory_states).flatten(1, -1)  # (batch_size, output_dim)
        loradict = self.lora_model.generate_lora_dict(self.lora_r, scale=self.scale, plain_tensor=plain_output)
        return loradict