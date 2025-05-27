import os
import torch
import torch.nn.functional as F
import argparse
import subprocess

from torch import nn
from typing import List, Optional
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    Qwen2ForCausalLM
)
from transformers.modeling_outputs import CausalLMOutputWithPast

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import json

from torch.utils.data import Dataset  


class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_length)


#local_rank = int(os.environ.get("LOCAL_RANK", 0))
#device = torch.device(f"cuda:{local_rank}")

class Qwen2ForDistillation(Qwen2ForCausalLM):
    def __init__(
        self,
        teacher_model_path: str,
        student_model_path: str,
        config,
        norm_factor,
        distil_layers: List[int],
        removed_layers_iterations: List[int]
    ):
        super().__init__(config)

        self.norm_factor = norm_factor
        self.distil_layers = distil_layers
        self.removed_layers_iterations = removed_layers_iterations

        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.student = AutoModelForCausalLM.from_pretrained(
            student_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        
        for param in self.student.parameters():
            param.requires_grad = False

        for layer_idx in self.distil_layers:
            for param in self.student.model.layers[layer_idx].parameters():
                param.requires_grad = True

        self.student.train()
        self.hidden_loss_fn = nn.MSELoss()

        self.val_student_hs_losses = []
        self.val_student_hs_losses_all_batches = []
        self.val_logits_losses = []
        self.val_logits_losses_all_batches = []
        self.val_last_layer_losses = []
        self.val_last_layer_losses_all_batches = []

    def get_student_to_teacher_mapping(self, removed_teacher_layers: list, total_teacher_layers: int) -> dict:
            teacher_layers_remaining = [i for i in range(total_teacher_layers) if i not in removed_teacher_layers]
            mapping = {student_idx: teacher_idx for student_idx, teacher_idx in enumerate(teacher_layers_remaining)}
            return mapping

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )

        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
    
        teacher_hidden_states = teacher_outputs.hidden_states
        student_hidden_states = student_outputs.hidden_states
        total_teacher_layers = self.teacher.config.num_hidden_layers

        hidden_loss = 0.0
        
        teacher_mapping = list(range(total_teacher_layers))
        for iteration in self.removed_layers_iterations:
            teacher_mapping = [tid for i, tid in enumerate(teacher_mapping) if i not in set(iteration)]
            
        mapping = {student_idx: teacher_id for student_idx, teacher_id in enumerate(teacher_mapping)}
        print(mapping)

        
        loss_layers = [12]
        for student_layer in loss_layers:
            teacher_layer = mapping[student_layer]
            hidden_loss += self.hidden_loss_fn(
                student_hidden_states[student_layer],
                teacher_hidden_states[teacher_layer]
            )
        
        hidden_loss += self.hidden_loss_fn(
            student_hidden_states[-1],
            teacher_hidden_states[-1]
        )
        
        
        #loss = hidden_loss

        student_log_probs = F.log_softmax(student_outputs.logits, dim=-1)
        #student_probs = F.softmax(student_outputs.logits, dim=-1)
        
        teacher_probs = F.softmax(teacher_outputs.logits, dim=-1)
        #teacher_log_probs = F.log_softmax(teacher_outputs.logits, dim=-1)

        #kl_div_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        kl_div_loss = torch.sum(F.kl_div(student_log_probs, teacher_probs, reduction="none"))
        
        loss = kl_div_loss

        #loss = self.norm_factor * kl_div_loss + hidden_loss / self.norm_factor
        
        loss = kl_div_loss / 100 + hidden_loss

        #print(f"Sum loss: {loss}; logits_normalized: {self.norm_factor * kl_div_loss}; hs_normalized: {(1 - self.norm_factor) * hidden_loss}")
        #print(f"logits_loss: {kl_div_loss}; hs_loss: {hidden_loss}")
        #print()


        if not self.training:

            #hs_loss_3 = self.hidden_loss_fn(student_hidden_states[3], teacher_hidden_states[4])
            #hs_loss_28 = self.hidden_loss_fn(student_hidden_states[27], teacher_hidden_states[29])
            #hs_loss_val = hs_loss_3 + hs_loss_28 

            #hs_loss_val = self.hidden_loss_fn(student_hidden_states[12], teacher_hidden_states[mapping[12]])
            #print(f"Mapping: studnent's 3 to teacher's {mapping[3]}")

            #self.val_student_hs_losses_all_batches.append(hs_loss_val.detach().cpu().item())
            
            logits_loss = torch.sum(F.kl_div(
                F.log_softmax(student_outputs.logits, dim=-1),
                F.softmax(teacher_outputs.logits, dim=-1),
                reduction='none'
            ))

            last_layer_loss = self.hidden_loss_fn(student_hidden_states[-1], teacher_hidden_states[-1])

            self.val_logits_losses_all_batches.append(logits_loss.detach().cpu().item())
            self.val_last_layer_losses_all_batches.append(last_layer_loss.detach().cpu().item())
    

        return CausalLMOutputWithPast(
            loss=loss,
            logits=student_outputs.logits,
            past_key_values=student_outputs.past_key_values,
            hidden_states=student_outputs.hidden_states,
            attentions=student_outputs.attentions,
        )


class LossAverageCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        model = kwargs.get('model')
        if model is not None:
            if hasattr(model, 'val_student_hs_losses_all_batches') and model.val_student_hs_losses_all_batches:
                avg_loss = sum(model.val_student_hs_losses_all_batches) / len(model.val_student_hs_losses_all_batches)
                model.val_student_hs_losses.append(avg_loss)
                model.val_student_hs_losses_all_batches = []
            if hasattr(model, 'val_logits_losses_all_batches') and model.val_logits_losses_all_batches:
                avg_logits_loss = sum(model.val_logits_losses_all_batches) / len(model.val_logits_losses_all_batches)
                model.val_logits_losses.append(avg_logits_loss)
                model.val_logits_losses_all_batches = []
            if hasattr(model, 'val_last_layer_losses_all_batches') and model.val_last_layer_losses_all_batches:
                avg_last_layer_loss = sum(model.val_last_layer_losses_all_batches) / len(model.val_last_layer_losses_all_batches)
                model.val_last_layer_losses.append(avg_last_layer_loss)
                model.val_last_layer_losses_all_batches = []
        return control


def parse_args():
    parser = argparse.ArgumentParser(description="Distillation script for Qwen2")

    parser.add_argument("--teacher_model_path", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--student_model_path", type=str, default="kngrg/Qwen2.5-3B-trimmed2")
    parser.add_argument("--distil_layers", nargs="+", type=int, required=True)
    parser.add_argument("--removed_layers_iterations", action="append", type=int, nargs="+", required=True,)

    parser.add_argument("--train_dataset", type=str, default="kngrg/ru-miracl-cleaned")

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--bf16", action="store_true", default=True) 
    parser.add_argument("--fp16", action="store_true", default=False)

    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--ds_frac", type=int, default=3)
    parser.add_argument("--use_local_data", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--norm_factor", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="./qwen2.5-3b-trimmed2-logits+hs")
    parser.add_argument("--logging_dir", type=str, default="./logs-logits+hs")

    return parser.parse_args()


def main():
    args = parse_args()

    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #torch.cuda.set_device(local_rank)
    #device = torch.device(f"cuda:{local_rank}")

    config = AutoConfig.from_pretrained(args.teacher_model_path)

    print("Removed_layers: ", args.removed_layers_iterations)

    distillation_model = Qwen2ForDistillation(
        teacher_model_path=args.teacher_model_path,
        student_model_path=args.student_model_path,
        config=config,
        norm_factor=args.norm_factor, 
        distil_layers=args.distil_layers, 
        removed_layers_iterations=args.removed_layers_iterations
    )
    #distillation_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)

    if args.use_local_data:
         
        train_file = "/scratch/s02210660/diploma-llm/data/train.json"
        val_file = "/scratch/s02210660/diploma-llm/data/val.json"

        def load_local_dataset(file_path, max_lines=10000000):
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

        #train = load_local_dataset(train_file)
        #len_train =  len(train)
        train = load_local_dataset(train_file, max_lines=args.ds_frac)
        val = load_local_dataset(val_file, max_lines=3000)
        train_dataset = TokenizedDataset(train, tokenizer, args.maxlen)
        val_dataset = TokenizedDataset(val, tokenizer, args.maxlen)

    else:
        dataset = load_dataset(args.train_dataset)
        dataset = dataset['train']
        ds_size = len(dataset)//args.ds_frac
        dataset = dataset.select(range(ds_size))

        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True,
                max_length=args.maxlen
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        train_dataset = tokenized_dataset['train']
        val_dataset = tokenized_dataset['test'].select(range(3000))


    data_collator = default_data_collator

    training_args_dict = {
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": "cosine_with_restarts",
        "warmup_steps": args.warmup_steps,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "optim": "adamw_torch",
        "save_total_limit": 1,
        "seed": 1337,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
        'report_to': 'tensorboard',
        'logging_dir': args.logging_dir,
        'ddp_find_unused_parameters': True 
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        **training_args_dict
    )

    trainer = Trainer(
        model=distillation_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[LossAverageCallback()],
    )

    #subprocess.run(["nvidia-smi"])    
    
    trainer.train()

    

    plt.figure(figsize=(10, 5))
    plt.plot(distillation_model.val_student_hs_losses, label="HS Loss (Specified Layers)")
    plt.xlabel("Validation Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Hidden State Loss (Specified Layers)")
    plt.savefig(f"{args.output_dir}/hidden_state_losses_specified.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(distillation_model.val_last_layer_losses, label="HS Loss (Last Layers)")
    plt.xlabel("Validation Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Hidden State Loss (Last Layers)")
    plt.savefig(f"{args.output_dir}/hidden_state_losses_last.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(distillation_model.val_logits_losses, label="Logits Loss")
    plt.xlabel("Validation Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Logits Loss")
    plt.savefig(f"{args.output_dir}/logits_losses.png")
    plt.close()



   # if local_rank == 0:
    distillation_model.student.save_pretrained(
        os.path.join(args.output_dir, "student"), 
        torch_dtype=torch.bfloat16
    )
    tokenizer.save_pretrained(os.path.join(args.output_dir, "student"))

    print(F"Student parameters: {sum(p.numel() for p in distillation_model.student.parameters())}")



if __name__ == "__main__":
    main()