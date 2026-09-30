"""Command-line entry point for the iterative pipeline."""

import argparse

from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iterative Block Influence pruning and repair distillation")
    parser.add_argument("--teacher_model_path", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dataset", default="kngrg/ru-miracl-cleaned")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--target_num_layers", type=int, required=True)
    parser.add_argument("--layers_per_iteration", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_importance_samples", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=20_000)
    parser.add_argument("--max_eval_samples", type=int, default=1_000)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--protect_first_layers", type=int, default=1)
    parser.add_argument("--protect_last_layers", type=int, default=1)
    parser.add_argument("--repair_radius", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--hidden_weight", type=float, default=1.0)
    parser.add_argument("--lm_weight", type=float, default=0.0)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_final_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_lm_head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attention_implementation", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--report_to", choices=("none", "json"), default="json")
    return parser


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    return PipelineConfig(**vars(build_parser().parse_args(argv)))


def main() -> None:
    try:
        run_pipeline(parse_args())
    except ValueError as error:
        raise SystemExit(f"configuration error: {error}") from error


if __name__ == "__main__":
    main()
