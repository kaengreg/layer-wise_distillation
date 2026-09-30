import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from layerwise_distillation.losses import distillation_loss
from layerwise_distillation.pipeline import _forward_with_block_outputs
from layerwise_distillation.pruning import prune_layers
from layerwise_distillation.repair import set_trainable_parameters


def tiny_qwen(depth=4):
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=depth, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32, tie_word_embeddings=False)
    return Qwen2ForCausalLM(config)


def test_tiny_qwen_prune_save_reload_and_forward(tmp_path):
    torch.manual_seed(7)
    student = tiny_qwen()
    mapping, removed = prune_layers(student, [1, 2], list(range(4)))
    assert mapping == [0, 3]
    assert removed == [1, 2]
    student.save_pretrained(tmp_path, safe_serialization=True)

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True)
    output = reloaded(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
    assert output.logits.shape == (1, 3, 32)
    assert reloaded.config.num_hidden_layers == 2
    assert [layer.self_attn.layer_idx for layer in reloaded.model.layers] == [0, 1]


def test_one_finite_backward_and_optimizer_step():
    torch.manual_seed(11)
    teacher = tiny_qwen()
    student = tiny_qwen()
    student.load_state_dict(teacher.state_dict())
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    mapping, _ = prune_layers(student, [1], list(range(4)))
    trainable = set_trainable_parameters(student, [0, 1], train_final_norm=True, train_lm_head=True)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    with torch.no_grad():
        teacher_outputs, teacher_block_outputs = _forward_with_block_outputs(teacher, {"input_ids": input_ids, "attention_mask": mask})
    student_outputs, student_block_outputs = _forward_with_block_outputs(student, {"input_ids": input_ids, "attention_mask": mask})
    loss, _ = distillation_loss(student_outputs, teacher_outputs, mapping, mask, input_ids, [0, 1], temperature=2.0, kl_weight=1.0, hidden_weight=1.0, lm_weight=0.1, student_block_outputs=student_block_outputs, teacher_block_outputs=teacher_block_outputs)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in student.parameters())
    optimizer.step()


def test_only_requested_student_components_are_trainable():
    student = tiny_qwen()
    set_trainable_parameters(student, [1], train_final_norm=False, train_lm_head=False)
    trainable_names = {name for name, parameter in student.named_parameters() if parameter.requires_grad}
    assert trainable_names
    assert all(name.startswith("model.layers.1.") for name in trainable_names)

    set_trainable_parameters(student, [2], train_final_norm=True, train_lm_head=True)
    trainable_names = {name for name, parameter in student.named_parameters() if parameter.requires_grad}
    assert any(name.startswith("model.layers.2.") for name in trainable_names)
    assert any(name.startswith("model.norm.") for name in trainable_names)
    assert any(name.startswith("lm_head.") for name in trainable_names)
    assert not any(name.startswith("model.layers.0.") for name in trainable_names)
