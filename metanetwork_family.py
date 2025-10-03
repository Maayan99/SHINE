import torch
import torch.nn.functional as F
import torch.nn as nn
import weakref

class MetanetworkTransformer(nn.Module):
    def __init__(self, output_dim, cfg):
        super().__init__()
        transformer_cfg = cfg.metanetwork.transformer_cfg
        self.fc_in = nn.Linear(cfg.hidden_size * cfg.num_mem_token, transformer_cfg.encoder_cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(**transformer_cfg.encoder_cfg)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_cfg.num_layers)
        self.fc_out1 = nn.Linear(transformer_cfg.encoder_cfg.d_model * cfg.num_layers, transformer_cfg.output_bottleneck)
        self.fc_out2 = nn.Linear(transformer_cfg.output_bottleneck, output_dim)
        
        self.num_layers = cfg.num_layers
        self.num_mem_token = cfg.num_mem_token
        self.hidden_size = cfg.hidden_size

    def forward(self, memory_states:torch.Tensor) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        batch_size = memory_states.shape[0]
        x = memory_states.view(batch_size, self.num_layers, self.num_mem_token * self.hidden_size)  
        x = F.gelu(self.fc_in(x))  # (batch_size, num_layer, d_model)    
        x = self.transformer_encoder(x)
        x = x.contiguous().view(batch_size, -1)  # (batch_size, num_layer * d_model) 
        x = F.gelu(self.fc_out1(x))
        x = self.fc_out2(x)  # (batch_size, output_dim)
        return x

class Metanetwork(nn.Module):
    def __init__(self, lora_model:nn.Module, cfg):
        super().__init__()
        self.lora_r = cfg.model.lora_r
        self.output_dim = lora_model.lora_params_numel(self.lora_r)
        self.lora_model = lora_model
        if cfg.metanetwork.type == "transformer":
            self.metanetwork = MetanetworkTransformer(self.output_dim, cfg)
            self.scale = cfg.metanetwork.transformer_cfg.scale
        else:
            raise ValueError(f"Unknown metanetwork type: {cfg.metanetwork.type}")

    def forward(self, memory_states:torch.Tensor) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        plain_output = self.metanetwork(memory_states)  # (batch_size, output_dim)
        loradict = self.lora_model.generate_lora_dict(self.lora_r, scale=self.scale, plain_tensor=plain_output)
        return loradict
        
    
