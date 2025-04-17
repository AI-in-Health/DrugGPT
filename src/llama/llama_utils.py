import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import numpy as np
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
import dataclasses
from typing import Optional
import math
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

class SoftEmbedding(nn.Module):
    """
    A simple soft embedding module that prepends learned embeddings to input embeddings.
    
    Attributes:
        n_tokens (int): Number of soft prompt tokens to use
        learned_embedding (nn.Parameter): Trainable soft prompt embeddings
    """
    
    def __init__(self, wte, n_tokens=20, initialize_from_vocab=True):
        """
        Initialize soft embeddings.
        
        Args:
            wte (nn.Embedding): The word token embedding layer from the model
            n_tokens (int): Number of soft prompt tokens
            initialize_from_vocab (bool): Whether to initialize from vocabulary or randomly
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        
        
        if initialize_from_vocab:
            self.learned_embedding = nn.Parameter(self.wte.weight[:n_tokens].clone())
        else:
            self.learned_embedding = nn.Parameter(torch.randn(n_tokens, self.wte.weight.size(1)) * 0.02)
    
    def forward(self, input_ids):
        """
        Forward pass that prepends soft prompt embeddings to input embeddings.
        
        Args:
            input_ids (torch.LongTensor): Input token IDs
            
        Returns:
            torch.FloatTensor: Combined embeddings with soft prompts
        """
        input_embeddings = self.wte(input_ids)
        
        
        batch_size = input_embeddings.shape[0]
        learned_embeddings = self.learned_embedding.repeat(batch_size, 1, 1)
        
        
        return torch.cat([learned_embeddings, input_embeddings], dim=1)


class LLaMAUtils:
    """
    Utility class for LLaMA models that supports soft prompt tuning.
    
    This class provides a simplified interface for working with LLaMA models,
    with support for soft prompt embeddings.
    """
    
    def __init__(self, configs, soft_prompt_checkpoint=None):
        """
        Initialize LLaMA utilities.
        
        Args:
            configs (dict): Configuration dictionary
            soft_prompt_checkpoint (str, optional): Path to soft prompt checkpoint
        """
        self.configs = configs
        self.model = None
        self.tokenizer = None
        self.soft_embedding = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.initialize_model()
        self.initialize_tokenizer()
        
        
        self.max_length = configs.get('max_length', 512)
        
        
        self.initialize_soft_embedding()
        if soft_prompt_checkpoint:
            self.load_soft_prompt(soft_prompt_checkpoint)
    
    def initialize_model(self):
        """Initialize the LLaMA model from HuggingFace."""
        logger.info(f"Loading model: {self.configs['model_name']}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.configs['model_name'],
                use_auth_token=self.configs.get('use_auth_token', False),
                trust_remote_code=True
            )
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def initialize_tokenizer(self):
        """Initialize the tokenizer for the model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.configs['model_name'],
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer initialized")
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {e}")
            raise
    
    def initialize_soft_embedding(self):
        """Initialize the soft embedding module."""
        try:
            wte = self.model.get_input_embeddings()
            self.soft_embedding = SoftEmbedding(
                wte,
                n_tokens=self.configs.get('soft_prompt_tokens', 20),
                initialize_from_vocab=True
            )
            logger.info("Soft embedding initialized")
        except Exception as e:
            logger.error(f"Error initializing soft embedding: {e}")
            self.soft_embedding = None
    
    def load_soft_prompt(self, checkpoint_file):
        """
        Load soft prompt parameters from a checkpoint file.
        
        Args:
            checkpoint_file (str): Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            if 'soft_prompt_params' in checkpoint:
                
                if self.soft_embedding is None:
                    self.initialize_soft_embedding()
                
                self.soft_embedding.load_state_dict(checkpoint['soft_prompt_params'])
                logger.info("Soft prompt loaded from checkpoint")
            else:
                logger.warning("No soft prompt parameters found in checkpoint")
        except Exception as e:
            logger.error(f"Error loading soft prompt: {e}")
    
    def save_soft_prompt(self, save_path):
        """
        Save soft prompt parameters to a checkpoint file.
        
        Args:
            save_path (str): Path to save checkpoint
        """
        if self.soft_embedding is not None:
            try:
                torch.save(
                    {'soft_prompt_params': self.soft_embedding.state_dict()},
                    save_path
                )
                logger.info(f"Soft prompt saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving soft prompt: {e}")
    
    def llama_inference(self, prompt, use_soft_prompt=True):
        """
        Run inference with the LLaMA model.
        
        Args:
            prompt (str): Input prompt
            use_soft_prompt (bool): Whether to use soft prompts
            
        Returns:
            str: Generated text
        """
        try:
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            if use_soft_prompt and self.soft_embedding is not None:
                
                inputs_embeds = self.soft_embedding(inputs.input_ids)
                
                
                attention_mask = torch.cat([
                    torch.ones(
                        (inputs.attention_mask.shape[0], self.soft_embedding.n_tokens),
                        device=self.device
                    ),
                    inputs.attention_mask
                ], dim=1)
                
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        max_length=self.max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                
                return self.tokenizer.decode(outputs[0, self.soft_embedding.n_tokens:], skip_special_tokens=True)
            else:
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return f"Error generating response: {str(e)}"
    
    def apply_soft_prompt(self, prompt, soft_prompt_prefix):
        """
        Update soft embeddings with embeddings derived from GCN output.
        
        Args:
            prompt (str): The input prompt
            soft_prompt_prefix (torch.Tensor): Tensor from GCN to use for soft prompt
            
        Returns:
            str: The prompt (unchanged, as updating happens internally)
        """
        try:
            if self.soft_embedding is None:
                logger.warning("Soft embedding not initialized")
                self.initialize_soft_embedding()
            
            
            with torch.no_grad():
                
                soft_prompt_prefix = soft_prompt_prefix.to(self.device)
                
                
                n_tokens = min(self.soft_embedding.n_tokens, soft_prompt_prefix.size(1))
                
                
                for i in range(n_tokens):
                    self.soft_embedding.learned_embedding.data[i] = soft_prompt_prefix[0, i]
                
            logger.info(f"Updated {n_tokens} soft prompt tokens")
            return prompt
        except Exception as e:
            logger.error(f"Error applying soft prompt: {e}")
            return prompt
    
    def llama_training_forward(self, input_ids, soft_prompt_prefix=None):
        """
        Forward pass for training purposes.
        
        Args:
            input_ids (torch.LongTensor): Input token IDs
            soft_prompt_prefix (torch.Tensor, optional): GCN-generated prefix
            
        Returns:
            torch.FloatTensor: Model outputs
        """
        try:
            
            input_ids = input_ids.to(self.device)
            
            
            if soft_prompt_prefix is not None:
                self.apply_soft_prompt(None, soft_prompt_prefix)
            
            
            if self.soft_embedding is not None:
                inputs_embeds = self.soft_embedding(input_ids)
                
                
                batch_size = input_ids.size(0)
                attention_mask = torch.cat([
                    torch.ones((batch_size, self.soft_embedding.n_tokens), device=self.device),
                    torch.ones_like(input_ids)
                ], dim=1)
                
                
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            else:
                
                outputs = self.model(input_ids=input_ids, return_dict=True)
            
            return outputs.logits
        except Exception as e:
            logger.error(f"Error in training forward pass: {e}")
            raise
    
    def freeze_llama_weights(self):
        """Freeze the weights of the LLaMA model to prevent them from being updated during training."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("LLaMA model weights frozen")


class LlamaConfig:
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    max_sequence_length: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    position_embedding_base: int

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    initializer_range: float
    model_type: str
    torch_dtype: str
    num_key_value_heads: int = 0

    def __post_init__(self):
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0

    @classmethod
    def from_dict(cls, d: dict) -> "LlamaConfig":
        field_names = (field.name for field in dataclasses.fields(cls))
        return cls(**{k: v for k, v in d.items() if k in field_names})


class RotaryEmbedding(nn.Module):
    """
        Implements rotary positional embeddings.

        Attributes:
            dim (int): The dimensionality of the embeddings.
            max_seq_len (int): The maximum sequence length.

        Methods:
            forward(x, offset=0): Applies rotary embeddings to the input tensor.
        """
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('sin', freqs.sin())
        self.register_buffer('cos', freqs.cos())

    def forward(self, x, offset=0):
        dim = self.dim
        sin, cos = self.sin[offset:offset + x.shape[1]], self.cos[offset:offset + x.shape[1]]
        sin, cos = sin.unsqueeze(-1), cos.unsqueeze(-1)
        return torch.cat((x[..., :dim // 2] * cos + x[..., dim // 2:] * sin,
                          x[..., :dim // 2] * sin - x[..., dim // 2:] * cos), dim=-1)


class LlamaFFN(nn.Module):
    """
        Implements the feed-forward network (FFN) for the LLaMA architecture.

        Attributes:
            gate_proj (nn.Linear): Linear layer for gating.
            up_proj (nn.Linear): Linear layer for up-projection.
            down_proj (nn.Linear): Linear layer for down-projection.

        Methods:
            forward(x): Passes the input through the feed-forward network.
        """
    def __init__(self, hidden_size, intermediate_size):
        super(LlamaFFN, self).__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        x1 = self.gate_proj(x)
        x2 = self.up_proj(x)
        return self.down_proj(F.silu(x1) * x2)


class LlamaAttention(nn.Module):
    """
        Implements the attention mechanism for the LLaMA architecture.

        Attributes:
            hidden_size (int): Size of the hidden layer.
            num_attention_heads (int): Number of attention heads.
            num_key_value_heads (int): Number of key/value heads.
            head_dim (int): Dimension of each attention head.
            rotary_embedding (RotaryEmbedding): Rotary embedding module.
            q_proj, k_proj, v_proj, o_proj (nn.Linear): Linear projection layers for query, key, value, and output.
            k_cache, v_cache (torch.Tensor): Caches for key and value tensors.

        Methods:
            forward(hidden_states, attention_mask, total_seq_len): Computes the attention mechanism.
        """
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, max_sequence_length, rotary_embedding):
        super(LlamaAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.rotary_embedding = rotary_embedding

        
        assert self.hidden_size % self.num_attention_heads == 0

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.k_cache = torch.zeros(max_sequence_length, self.num_key_value_heads, self.head_dim)
        self.v_cache = torch.zeros(max_sequence_length, self.num_key_value_heads, self.head_dim)

    def forward(self, hidden_states, attention_mask, total_seq_len):
        batch_size, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        
        q, k = self.rotary_embedding(q, k, total_seq_len - seq_len)

        
        self.k_cache = torch.cat((self.k_cache, k.squeeze(0)), dim=0)[-total_seq_len:]
        self.v_cache = torch.cat((self.v_cache, v.squeeze(0)), dim=0)[-total_seq_len:]

        
        attn_output, _ = F.multi_head_attention_forward(
            query=q,
            key=self.k_cache.unsqueeze(0).expand(batch_size, -1, -1, -1),
            value=self.v_cache.unsqueeze(0).expand(batch_size, -1, -1, -1),
            embed_dim_to_check=self.hidden_size,
            num_heads=self.num_attention_heads,
            dropout_p=0,
            attention_mask=attention_mask,
            need_weights=False,
        )

        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayer(nn.Module):
    """
        Represents a single decoder layer in the LLaMA model.

        Attributes:
            attn (LlamaAttention): Attention mechanism for the layer.
            ffn (LlamaFFN): Feed-forward network for the layer.
            input_norm, post_attention_norm (nn.LayerNorm): Layer normalization modules.

        Methods:
            forward(hidden_states, attention_mask, total_seq_len): Processes input through the decoder layer.
        """
    def __init__(self, config, rotary_embedding):
        super(LlamaDecoderLayer, self).__init__()
        self.attn = LlamaAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                                   config.max_sequence_length, rotary_embedding)
        self.ffn = LlamaFFN(config.hidden_size, config.intermediate_size)
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, total_seq_len):
        
        attn_output = self.attn(self.input_norm(hidden_states), attention_mask, total_seq_len)
        hidden_states = attn_output + hidden_states

        
        ffn_output = self.ffn(self.post_attention_norm(hidden_states))
        hidden_states = ffn_output + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    """
        The core LLaMA model comprising multiple decoder layers.

        Attributes:
            embed_tokens (nn.Embedding): Token embedding layer.
            layers (nn.ModuleList): List of LLaMA decoder layers.
            norm (nn.LayerNorm): Layer normalization for the final output.

        Methods:
            forward(inputs, total_seq_len, attention_mask): Processes input through the entire model.
        """
    def __init__(self, config):
        super(LlamaModel, self).__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        rotary_embedding = RotaryEmbedding(config.position_embedding_base, config.max_sequence_length,
                                           config.hidden_size // config.num_attention_heads)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, rotary_embedding)
                                     for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs, total_seq_len, attention_mask):
        hidden_states = self.embed_tokens(inputs)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCasualLM(nn.Module):
    """
        LLaMA model for causal language modeling.

        Attributes:
            model (LlamaModel): The core LLaMA model.
            lm_head (nn.Linear): Linear layer for language modeling predictions.
            vocab_size (int): Size of the vocabulary.
            dtype (str): Data type of the model.

        Methods:
            forward(inputs, attention_mask=None, total_seq_len=None, inputs_embeds=None): Processes input for language modeling.
            save_pretrained(save_path): Saves the model's state dictionary to a file.
            from_pretrained(config, load_path): Class method to load a pretrained model from a file.
        """
    def __init__(self, config, dtype='float32'):
        super(LlamaForCasualLM, self).__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.dtype = dtype

    def forward(self, inputs, attention_mask=None, total_seq_len=None, inputs_embeds=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.model.embed_tokens(inputs)

        if attention_mask is None:
            attention_mask = torch.ones((hidden_states.shape[0], 1, hidden_states.shape[1], total_seq_len),
                                        dtype=torch.float32, device=hidden_states.device)
            attention_mask = (attention_mask - 1) * 1e9  

        for layer in self.model.layers:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def save_pretrained(self, save_path):
        torch.save(self.state_dict(), save_path)

    @classmethod
    def from_pretrained(cls, config, load_path):
        
        model = cls(config)
        
        model.load_state_dict(torch.load(load_path))
        return model
