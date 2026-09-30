from types import SimpleNamespace

import pytest
import torch
from torch import nn

from layerwise_distillation.importance import block_influence_from_hidden_states, evaluate_block_influence, rank_layers
from layerwise_distillation.losses import causal_lm_loss, mapped_block_output_mse, token_kl_divergence
from layerwise_distillation.pipeline import PipelineConfig, _evaluate, _prepare_output_dir, validate_pipeline_config, validate_single_process_environment
from layerwise_distillation.pruning import prune_layers
from layerwise_distillation.repair import select_repair_layers


class FakeLayer(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(layer_idx)))
        self.self_attn = SimpleNamespace(layer_idx=layer_idx)


class FakeModel(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=depth, layer_types=[f"type-{index}" for index in range(depth)], max_window_layers=3)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer(index) for index in range(depth)])
        self.model.config = self.config


def test_block_influence_averages_sequences_and_ignores_padding():
    inputs = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]])
    outputs = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]], [[0.0, 1.0], [-1.0, 0.0]]])
    mask = torch.tensor([[1, 0], [1, 1]])
    scores = block_influence_from_hidden_states((inputs, outputs), mask)
    assert scores == pytest.approx([0.75])


def test_model_importance_uses_raw_block_output_before_final_norm():
    class SignedLayer(nn.Module):
        def __init__(self, sign):
            super().__init__()
            self.sign = sign

        def forward(self, hidden_states):
            return (hidden_states * self.sign,)

    class HookModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([SignedLayer(1), SignedLayer(-1)])

        def forward(self, input_ids, attention_mask, **kwargs):
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=2).float()
            for layer in self.model.layers:
                hidden = layer(hidden)[0]
            return SimpleNamespace(logits=hidden.abs())

    batch = {"input_ids": torch.tensor([[0, 1]]), "attention_mask": torch.tensor([[1, 1]])}
    assert evaluate_block_influence(HookModel(), [batch], torch.device("cpu")) == pytest.approx([0.0, 2.0])


def test_ranking_is_deterministic_and_respects_protected_ranges():
    assert rank_layers([0.0, 0.2, 0.1, 0.1, 0.0], 5, protect_first=1, protect_last=1) == [2, 3, 1]
    with pytest.raises(ValueError, match="finite"):
        rank_layers([0.0, float("nan")], 2)


def test_prune_one_layer_updates_mapping_config_and_attention_indices():
    model = FakeModel(5)
    remaining, removed = prune_layers(model, [1], [0, 1, 2, 3, 4])
    assert remaining == [0, 2, 3, 4]
    assert removed == [1]
    assert model.config.num_hidden_layers == 4
    assert model.config.layer_types == ["type-0", "type-2", "type-3", "type-4"]
    assert model.config.max_window_layers == 2
    assert [layer.self_attn.layer_idx for layer in model.model.layers] == [0, 1, 2, 3]


def test_prune_multiple_layers_preserves_original_mapping_across_iterations():
    model = FakeModel(6)
    mapping, first_removed = prune_layers(model, [1, 4], list(range(6)))
    mapping, second_removed = prune_layers(model, [2], mapping)
    assert first_removed == [1, 4]
    assert second_removed == [3]
    assert mapping == [0, 2, 5]


@pytest.mark.parametrize("indices", ([0, 1, 2], [-1], [3], [1, 1]))
def test_prune_rejects_invalid_indices(indices):
    with pytest.raises(ValueError):
        prune_layers(FakeModel(3), indices, [0, 1, 2])


def test_repair_layers_around_isolated_and_contiguous_removals():
    assert select_repair_layers(6, [2], radius=1) == [1, 2]
    assert select_repair_layers(7, [2, 3], radius=1) == [1, 2]
    assert select_repair_layers(8, [1, 5], radius=1) == [0, 1, 3, 4]
    assert select_repair_layers(5, [0], radius=2) == [0, 1]


def test_kl_masks_padding_and_applies_temperature_square():
    student = torch.tensor([[[2.0, 0.0], [100.0, -100.0]]])
    teacher = torch.tensor([[[0.0, 2.0], [-100.0, 100.0]]])
    mask = torch.tensor([[1, 0]])
    actual = token_kl_divergence(student, teacher, mask, temperature=2.0)
    expected = torch.nn.functional.kl_div(torch.log_softmax(student[:, :1] / 2.0, -1), torch.softmax(teacher[:, :1] / 2.0, -1), reduction="sum") * 4.0
    assert actual == pytest.approx(expected.item())


def test_hidden_and_lm_losses_ignore_padding():
    student_hidden = (torch.zeros(1, 3, 2), torch.tensor([[[1.0, 1.0], [2.0, 2.0], [99.0, 99.0]]]))
    teacher_hidden = (torch.zeros(1, 3, 2), torch.tensor([[[0.0, 0.0], [1.0, 1.0], [-99.0, -99.0]]]))
    mask = torch.tensor([[1, 1, 0]])
    assert mapped_block_output_mse(student_hidden[1:], teacher_hidden[1:], [0], mask).item() == pytest.approx(1.0)

    logits = torch.tensor([[[0.0, 0.0, 5.0], [0.0, 5.0, 0.0], [20.0, 0.0, 0.0]]])
    labels = torch.tensor([[2, 1, 0]])
    actual = causal_lm_loss(logits, labels, mask)
    expected = torch.nn.functional.cross_entropy(logits[:, 0], labels[:, 1])
    assert actual == pytest.approx(expected.item())


def test_causal_lm_loss_is_invariant_to_left_or_right_padding():
    logits = torch.tensor([[[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [5.0, 0.0, 0.0]]])
    right_loss = causal_lm_loss(logits, torch.tensor([[1, 2, 0]]), torch.tensor([[1, 1, 0]]))
    left_loss = causal_lm_loss(logits, torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 1]]))
    expected = torch.nn.functional.cross_entropy(logits[:, 0], torch.tensor([2]))
    assert right_loss == pytest.approx(expected.item())
    assert left_loss == pytest.approx(expected.item())


def test_evaluation_uses_exact_valid_transition_count_for_left_padding(tmp_path):
    class UniformModel(nn.Module):
        def forward(self, input_ids, attention_mask, **kwargs):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 3))

    config = PipelineConfig(teacher_model_path="unused", dataset="unused", text_column="text", target_num_layers=1, layers_per_iteration=1, output_dir=str(tmp_path), dtype="float32")
    model = UniformModel()
    right = [{"input_ids": torch.tensor([[1, 2, 0]]), "attention_mask": torch.tensor([[1, 1, 0]])}]
    left = [{"input_ids": torch.tensor([[0, 1, 2]]), "attention_mask": torch.tensor([[0, 1, 1]])}]
    right_metrics = _evaluate(model, model, right, config, torch.device("cpu"))
    left_metrics = _evaluate(model, model, left, config, torch.device("cpu"))
    assert right_metrics["validation_loss"] == pytest.approx(torch.log(torch.tensor(3.0)).item())
    assert left_metrics["validation_loss"] == pytest.approx(right_metrics["validation_loss"])


def test_hidden_mapping_uses_original_teacher_layer_ids():
    student_outputs = [torch.zeros(1, 2, 1), torch.full((1, 2, 1), 7.0)]
    teacher_outputs = [torch.zeros(1, 2, 1), torch.ones(1, 2, 1), torch.full((1, 2, 1), 7.0)]
    loss = mapped_block_output_mse(student_outputs, teacher_outputs, [0, 2], torch.ones(1, 2), layer_indices=[1])
    assert loss.item() == 0.0


def test_invalid_target_depth_is_rejected_before_model_loading(tmp_path):
    base = dict(teacher_model_path="unused", dataset="unused", text_column="text", layers_per_iteration=1, output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="smaller than the teacher depth"):
        validate_pipeline_config(PipelineConfig(target_num_layers=4, **base), initial_num_layers=4)
    with pytest.raises(ValueError, match="protected layer count"):
        validate_pipeline_config(PipelineConfig(target_num_layers=1, protect_first_layers=1, protect_last_layers=1, **base), initial_num_layers=4)
    with pytest.raises(ValueError, match="hidden-state loss requires"):
        validate_pipeline_config(PipelineConfig(target_num_layers=2, repair_radius=0, kl_weight=0.0, hidden_weight=1.0, lm_weight=0.0, **base), initial_num_layers=4)


def test_distributed_launch_and_nonempty_output_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="one process"):
        validate_single_process_environment({"WORLD_SIZE": "2"})
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        _prepare_output_dir(output_dir)
