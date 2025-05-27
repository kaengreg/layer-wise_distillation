from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse 
import os
import json

def load_local_dataset(file_path, max_lines=1000):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            try:
                record = json.loads(line)
                if 'text' in record:
                    documents.append(record['text'].strip())
            except json.JSONDecodeError:
                continue
    return documents

def calculate_mean_block_importance(X_i, X_i_plus_1):
    BI_scores = []
    X_i = X_i.squeeze(0)
    X_i_plus_1 = X_i_plus_1.squeeze(0)

    for row in range(X_i.shape[0]):
        row_X_i = X_i[row]
        row_X_i_plus_1 = X_i_plus_1[row]
        numerator = row_X_i.T @ row_X_i_plus_1
        denominator = np.linalg.norm(row_X_i) * np.linalg.norm(row_X_i_plus_1)
        BI_scores.append(numerator / denominator)

    mean_BI = np.mean(BI_scores)
    BI_i = 1 - mean_BI
    return BI_i

def calculate_mean_mse_importance(X_i, X_i_plus_1):
    mse_scores = []
    X_i = X_i.squeeze(0)
    X_i_plus_1 = X_i_plus_1.squeeze(0)

    for row in range(X_i.shape[0]):
        row_X_i = X_i[row]
        row_X_i_plus_1 = X_i_plus_1[row]
        mse_value = np.mean((row_X_i - row_X_i_plus_1) ** 2)
        mse_scores.append(mse_value)

    mean_mse = np.mean(mse_scores)
    return mean_mse

def calculate_BI(model, tokenizer, dataset):
    tokenizer.pad_token = tokenizer.eos_token
    BI_per_text = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for text in dataset:
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            BI_scores = {}
            for i in range(model.config.num_hidden_layers):
                hs_i = outputs.hidden_states[i].detach().cpu().numpy().astype(np.float64)
                hs_i_plus_1 = outputs.hidden_states[i + 1].detach().cpu().numpy().astype(np.float64)
                mean_BI = calculate_mean_block_importance(hs_i, hs_i_plus_1)
                BI_scores[i] = mean_BI
            BI_per_text.append(BI_scores)
            del inputs, outputs
        
    sum_BI_scores = {}
    mean_BI_scores = {}

    for bi_dict in BI_per_text:
        for layer, bi in bi_dict.items():
            if layer not in sum_BI_scores:
                sum_BI_scores[layer] = bi
            else:
                sum_BI_scores[layer] += bi

    num_texts = len(BI_per_text)
    for layer, sum_bi in sum_BI_scores.items():
        mean_BI_scores[layer] = sum_bi / num_texts

    return mean_BI_scores

def calculate_mse_importance(model, tokenizer, dataset):
    mse_scores_accum = {}
    num_texts = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for text in dataset:
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  
            for i in range(model.config.num_hidden_layers):
                hs_current = hidden_states[i].detach().cpu().numpy().astype(np.float64)
                hs_next = hidden_states[i+1].detach().cpu().numpy().astype(np.float64)
                mse = calculate_mean_mse_importance(hs_current, hs_next)
                mse_scores_accum[i] = mse_scores_accum.get(i, 0) + mse
            num_texts += 1
            del inputs, outputs

    if num_texts == 0:
        return {}
    
    mean_mse_scores = {layer: total / num_texts for layer, total in mse_scores_accum.items()}
    return mean_mse_scores


def query_hook(module, input, output):
    query_activation_data[module.layer_id] = {
         'input': input[0].detach(),      
         'output': output.detach(),         
         'module': module                   
    }

def register_query_hooks(model):
    for i, layer in enumerate(model.layers):
        query_layer = layer.self_attn.q_proj
        query_layer.layer_id = f"layer_{i}_q_proj"
        query_layer.register_forward_hook(query_hook)

def propagate_epsilon_linear(module, input_activations, output_relevance, epsilon=1e-6):
 
    x = input_activations
    W = module.weight  
    z = torch.matmul(x, W.t()) 
    sign_z = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    z_stable = z + epsilon * sign_z
    s = output_relevance / (z_stable + 1e-9)  
    R_in = x * torch.matmul(s, W)
    return R_in

def calculate_LRP_importance(model, tokenizer, dataset, epsilon=1e-6):
    global query_activation_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    register_query_hooks(model)
    
    total_layer_relevance = {}
    n_texts = 0

    with torch.no_grad():
        for text in dataset:
            query_activation_data = {}
            
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
            _ = model(**inputs)
            del inputs
            
            for layer_id, data in query_activation_data.items():
                R_out = data['output'].abs()
                module_ref = data['module']
                R_in = propagate_epsilon_linear(module_ref, data['input'], R_out, epsilon)
                relevance_score = R_in.abs().sum().item()
                total_layer_relevance[layer_id] = total_layer_relevance.get(layer_id, 0) + relevance_score
            
            n_texts += 1

    average_layer_relevance = {layer: total / n_texts for layer, total in total_layer_relevance.items()}
    return average_layer_relevance

def graph_BI(mean_BI_scores, title, filepath="BI_graph.png"):
    layers = list(mean_BI_scores.keys())
    scores = [mean_BI_scores[layer] for layer in layers]

    plt.figure(figsize=(10, 10))
    sns.barplot(x=[f"Layer {layer}" for layer in layers], y=scores, palette="viridis")
    plt.xlabel('Layers')
    plt.ylabel('BI')
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(filepath)
    plt.close()
    print(f"График важности слоев сохранен в: {filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True, help='Path to local dataset file')
    parser.add_argument("--lines", type=int, default=10000)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--graph_name", type=str, required=True)
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B", help="Model local path or identifier")
    parser.add_argument("--loss_type", type=str, default="BI", choices=["BI", "MSE", "LRP"])

    args = parser.parse_args()

    dataset_cut = load_local_dataset(args.dataset_file, max_lines=args.lines)
    
    #dataset = load_dataset("ccdv/govreport-summarization", trust_remote_code=True)
    #dataset_train = dataset['train'][slice(None, 10, None)]
    #dataset_cut = dataset_train['report']
    
    print(f"Загружено {len(dataset_cut)} из {args.dataset_file}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.maxlen)

    if args.loss_type == "BI":
        mean_scores = calculate_BI(model, tokenizer, dataset_cut)
        title = "Block Importance per Layer"
    elif args.loss_type == "MSE":
        mean_scores = calculate_mse_importance(model, tokenizer, dataset_cut)
        title = "MSE Loss between Layers"
    elif args.loss_type == "LRP":
        mean_scores = calculate_LRP_importance(model, tokenizer, dataset_cut, epsilon=1e-6)
        title = "LRP-based Importance (ε-rule) for Query Projections"
    
    sorted_layers = sorted(mean_scores.items(), key=lambda x: x[1])
    print("Слои в порядке увеличения значений:")
    for layer, score in sorted_layers:
        print(f"Layer {layer}: {score:.4f}")

    if not os.path.exists(args.graph_path):
         os.makedirs(args.graph_path)
    
    output_file = os.path.join(args.graph_path, f"{args.loss_type}_ranking.txt")
    with open(output_file, "w", encoding="utf-8") as f:
         f.write("Слои в порядке увеличения значений:\n")
         for layer, score in sorted_layers:
             f.write(f"Layer {layer}: {score:.4f}\n")
    print(f"Результаты сохранены в файл: {output_file}")
    
    graph_file = os.path.join(args.graph_path, args.graph_name)
    graph_BI(mean_scores, "Block Importance per Layer", filepath=graph_file)
        
