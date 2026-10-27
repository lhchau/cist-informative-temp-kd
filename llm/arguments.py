import argparse

import deepspeed

DISTILL_TYPES = ["fkl", "rkl", "symkl", "jsd", "srkl", "akl", "ab", "cist"]


def get_args():
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("model")
    g.add_argument("--model-path", type=str, required=True, help="student (or SFT model) path / HF id")
    g.add_argument("--model-type", type=str, default="gpt2", choices=["gpt2", "opt"])
    g.add_argument("--teacher-model-path", type=str, default=None)
    g.add_argument("--gradient-checkpointing", action="store_true")

    g = parser.add_argument_group("runtime")
    g.add_argument("--type", type=str, required=True, choices=["sft", "eval"] + DISTILL_TYPES,
                   help="sft: fine-tune with CE; eval: generate + ROUGE-L; otherwise the distillation loss")
    g.add_argument("--save", type=str, required=True)
    g.add_argument("--log-interval", type=int, default=4)
    g.add_argument("--eval-interval", type=int, default=-1, help="-1 = once per epoch")
    g.add_argument("--seed", type=int, default=10)
    g.add_argument("--local_rank", type=int, default=None, help="set by the distributed launcher")

    g = parser.add_argument_group("data")
    g.add_argument("--data-dir", type=str, required=True)
    g.add_argument("--data-names", type=str, default=None, help="benchmark name (eval only)")
    g.add_argument("--train-num", type=int, default=-1)
    g.add_argument("--dev-num", type=int, default=-1)
    g.add_argument("--num-workers", type=int, default=1)
    g.add_argument("--max-prompt-length", type=int, default=256)
    g.add_argument("--eval-gen", action="store_true", help="generate on the dev set and track ROUGE-L during training")

    g = parser.add_argument_group("optimization")
    g.add_argument("--batch-size", type=int, default=8, help="per-GPU micro batch size")
    g.add_argument("--eval-batch-size", type=int, default=32)
    g.add_argument("--gradient-accumulation-steps", type=int, default=1)
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--total-iters", type=int, default=None)
    g.add_argument("--max-length", type=int, default=512)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--lr-min", type=float, default=1e-7)
    g.add_argument("--weight-decay", type=float, default=1e-2)
    g.add_argument("--clip-grad", type=float, default=1.0)
    g.add_argument("--warmup-iters", type=int, default=0, help="constant schedule only")
    g.add_argument("--lr-decay-style", type=str, default="cosine", choices=["constant", "cosine"])
    g.add_argument("--kd-ratio", type=float, default=1.0, help="loss = (1 - r) * CE + r * distillation")

    g = parser.add_argument_group("distillation")
    g.add_argument("--rho", type=float, default=None, help="CIST entropy-control constant")
    g.add_argument("--skew-alpha", type=float, default=0.1, help="SRKL skew coefficient")
    g.add_argument("--ab-alpha", type=float, default=0.2)
    g.add_argument("--ab-beta", type=float, default=0.7)

    g = parser.add_argument_group("generation")
    g.add_argument("--do-sample", action="store_true")
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--no-repeat-ngram-size", type=int, default=6)
    g.add_argument("--repetition-penalty", type=float, default=None)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    if args.type == "cist":
        assert args.rho is not None, "--type cist requires --rho"
    if args.type in DISTILL_TYPES:
        assert args.teacher_model_path, f"--type {args.type} requires --teacher-model-path"
    return args
