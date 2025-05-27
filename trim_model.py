from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import argparse
import os

def layer_removal(model, prune_list):
    model = model.model
    layers = model.layers
    sorted_list = sorted(prune_list, reverse=True)
    
    for prune_layer in sorted_list:
        del layers[prune_layer]

    for i, layer in enumerate(layers):
        if hasattr(layer, 'self_attn'):
            layer.self_attn.layer_idx = i
        
    model.config.num_hidden_layers -= len(prune_list)
    if hasattr(model.config, 'max_window_layers') and model.config.max_window_layers:
        model.config.max_window_layers -= len(prune_list)

    state_dict = model.state_dict()
    updated_state_dict = {k: v for k, v in state_dict.items()
                          if "layers" not in k or int(k.split('.')[1]) not in prune_list}
    model.load_state_dict(updated_state_dict, strict=False)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune_layers", nargs="+", type=int, required=True,)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--save_path", type=str, required=True,)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=512)

    print(f"Deleting layers: {args.prune_layers}")
    layer_removal(model, args.prune_layers)

    print(f"Saving trimmed model to {args.save_path}...")
    print(model.config)

    directory = os.path.dirname(args.save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)      

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print(F"Parameters: {sum(p.numel() for p in model.parameters())}")
