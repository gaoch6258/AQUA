

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from trl import GRPOConfig

from grpo_data import (
    _build_prompt_to_paths,
    build_prompt_to_qa,
    load_qa_dataset,
    resolve_dataset_paths,
)
from grpo_rewards import (
    reward_answer_accuracy,
    reward_format_correctness,
    reward_qclr_progressive,
    reward_qclr_terminal_soft,
    reward_qclr_query_penalty,
)
from grpo_rollout import create_rollout_func
from grpo_tools import FLUID_TOOLS
from grpo_trainer import AlignedGRPOTrainer

os.environ.setdefault("VLLM_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")


CONFIG = {

    "gaussians_dir": "/data/luozhemeng/Agent_RL/real_fluid_grpo_copy/dataset/boxTurb_tzxyc/gs",

    "qa_dir": "/data/luozhemeng/Agent_RL/real_fluid_grpo_copy/dataset/boxTurb_tzxyc/QA/cases",

    "output_dir": "./outputs/real_fluid_vlm_agent",

    "model_name": "/share/users/luozhemeng/Qwen3/Qwen3-VL-4B-Instruct",

    "device": "cuda",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实3DGS流体分析GRPO训练")
    parser.add_argument("--gaussians-dir", default=CONFIG["gaussians_dir"])
    parser.add_argument("--qa-dir", default=CONFIG["qa_dir"])
    parser.add_argument(
        "--dataset",
        default=None,
        help="数据集名或路径（包含 QA/cases、gs），设置后自动覆盖 qa/gaussians 目录",
    )
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument(
        "--no-timestamp-output-dir",
        action="store_true",
        help="禁用输出目录时间戳（默认每次训练创建带时间戳的输出目录）",
    )
    parser.add_argument("--model-name", default=CONFIG["model_name"])
    parser.add_argument("--device", default=CONFIG["device"])
    parser.add_argument("--limit", type=int, default=None, help="限制加载的问题数量")
    parser.add_argument("--eval-ratio", type=float, default=0.3, help="验证集比例(0表示不划分)")
    parser.add_argument("--eval-seed", type=int, default=42, help="验证集划分随机种子")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度(增大提升多样性)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p采样")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k采样")
    parser.add_argument("--min-p", type=float, default=0.0, help="最小概率截断")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="重复惩罚")
    parser.add_argument("--sampling-seed", type=int, default=42, help="采样基础随机种子")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="按案例名过滤，可重复传参",
    )
    return parser.parse_args()


def _append_timestamp(output_dir: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir)
    if path.name:
        return str(path.with_name(f"{path.name}_{stamp}"))
    return f"{output_dir}_{stamp}"


def _get_rank_index() -> int:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    try:
        return int(rank)
    except ValueError:
        return 0


def _resolve_output_dir(output_dir: str, use_timestamp: bool) -> str:
    if not use_timestamp:
        return output_dir
    base_path = Path(output_dir)
    base_name = base_path.name if base_path.name not in {"", ".", ".."} else "run"
    sync_file = base_path.parent / f".{base_name}.run_dir"
    run_id = (
        os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("ACCELERATE_RUN_ID")
        or os.environ.get("ACCELERATE_PROCESS_UUID")
    )
    rank = _get_rank_index()
    if rank == 0:
        resolved = _append_timestamp(output_dir)
        sync_file.parent.mkdir(parents=True, exist_ok=True)
        meta = run_id if run_id else str(time.time_ns())
        sync_file.write_text(f"{resolved}\n{meta}\n", encoding="utf-8")
        return resolved

    max_wait = 300  # 30s
    for _ in range(max_wait):
        if sync_file.exists():
            content = sync_file.read_text(encoding="utf-8")
            if content:
                lines = content.splitlines()
                resolved = lines[0].strip() if lines else ""
                file_run_id = lines[1].strip() if len(lines) > 1 else ""
                if resolved and (not run_id or not file_run_id or file_run_id == run_id):
                    return resolved
        time.sleep(0.1)
    raise SystemExit(f"等待主进程写入输出目录超时: {sync_file}")


def main():
    args = _parse_args()
    if args.dataset:
        try:
            gaussians_dir, qa_dir = resolve_dataset_paths(args.dataset)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        args.gaussians_dir = gaussians_dir
        args.qa_dir = qa_dir

    if not args.gaussians_dir:
        raise SystemExit("训练需要 gs 目录，请检查数据集结构或改用包含 gs 的数据集")

    args.output_dir = _resolve_output_dir(
        args.output_dir,
        use_timestamp=not args.no_timestamp_output_dir,
    )

    runtime_config = {
        "gaussians_dir": args.gaussians_dir,
        "qa_dir": args.qa_dir,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "device": args.device,
    }

    def _is_main_process() -> bool:
        return _get_rank_index() == 0

    is_main = _is_main_process()

    log_dir = runtime_config["output_dir"]
    os.makedirs(log_dir, exist_ok=True)
    rank_id = _get_rank_index()
    log_path = os.path.join(log_dir, "train.log" if rank_id == 0 else f"train_rank{rank_id}.log")

    class Tee:
        def __init__(self, *files):
            self.files = files
            self.primary = files[0] if files else None

        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

        def fileno(self):
            return self.primary.fileno() if self.primary else -1

        def isatty(self):
            return self.primary.isatty() if self.primary else False

    if is_main:
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    else:

        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    print("=" * 80)
    print("真实 3DGS 流体分析 GRPO 训练")
    print("=" * 80)


    print("\n[1/5] 检查数据目录...")
    qa_dir = Path(runtime_config["qa_dir"])
    gaussians_dir = Path(runtime_config["gaussians_dir"])
    gaussians_files = list(gaussians_dir.rglob("*.h5"))
    gaussians_files += list(gaussians_dir.rglob("*.hdf5"))
    gaussians_files = sorted(gaussians_files)
    if not gaussians_files:
        print(f"错误: 在 {gaussians_dir} 中没有找到Gaussian模型文件")
        return
    print(f"   高斯模型目录: {gaussians_dir}")
    print(f"   QA数据目录: {qa_dir}")
    print(f"   找到 {len(gaussians_files)} 个高斯模型文件（递归搜索）")


    print("\n[2/5] 准备数据集...")
    full_dataset = load_qa_dataset(
        qa_dir=runtime_config["qa_dir"],
        gaussians_dir=runtime_config["gaussians_dir"],
        limit=args.limit,
        case_names=set(args.case) if args.case else None,
    )
    print(f"   数据集大小: {len(full_dataset)} 条")
    if len(full_dataset) == 0:
        print("错误: 数据集为空，请检查过滤条件和数据目录")
        return

    train_cases = None
    eval_cases = None

    if args.eval_ratio and args.eval_ratio > 0:
        if not 0 < args.eval_ratio < 1:
            raise ValueError("--eval-ratio 必须在 (0, 1) 之间")
        case_names = sorted(set(full_dataset["case_name"]))
        if len(case_names) < 2:
            print("警告: case 数量不足，无法按 case 划分验证集，已禁用 eval。")
            train_dataset = full_dataset
            eval_dataset = None
            train_cases = set(case_names)
            eval_cases = set()
        else:
            rng = random.Random(args.eval_seed)
            rng.shuffle(case_names)
            eval_case_count = max(1, int(len(case_names) * args.eval_ratio))
            eval_case_count = min(eval_case_count, len(case_names) - 1)
            eval_cases = set(case_names[:eval_case_count])
            train_cases = set(case_names[eval_case_count:])
            train_dataset = full_dataset.filter(lambda x: x["case_name"] in train_cases)
            eval_dataset = full_dataset.filter(lambda x: x["case_name"] in eval_cases)
            print(
                f"   按 case 划分: 训练 {len(train_cases)} 个case, 验证 {len(eval_cases)} 个case"
            )
    else:
        train_dataset = full_dataset
        eval_dataset = None
    print(f"   训练集大小: {len(train_dataset)} 条")
    if eval_dataset is not None:
        print(f"   验证集大小: {len(eval_dataset)} 条")
    if train_cases is not None:
        train_case_list = sorted(train_cases)
        print(f"   训练case列表({len(train_case_list)}): {', '.join(train_case_list)}")
    if eval_cases:
        eval_case_list = sorted(eval_cases)
        print(f"   验证case列表({len(eval_case_list)}): {', '.join(eval_case_list)}")


    prompt_to_paths = _build_prompt_to_paths(full_dataset)
    prompt_to_qa = build_prompt_to_qa(full_dataset)


    print("\n[3/5] 创建 rollout 函数...")
    rollout_func = create_rollout_func(
        device=runtime_config["device"],
        prompt_to_paths=prompt_to_paths,
        output_dir=runtime_config["output_dir"],
        prompt_to_qa=prompt_to_qa,
    )


    print("\n[4/5] 配置训练参数...")
    evaluation_strategy = "epoch" if eval_dataset is not None else "no"
    train_config = GRPOConfig(
        output_dir=runtime_config["output_dir"],
        num_train_epochs=100,
        per_device_train_batch_size=3,  
        gradient_accumulation_steps=1,


        num_generations=3,  
        max_completion_length=2368,


        use_vllm=True,
        vllm_mode="colocate",
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="token_truncate",
        mask_truncated_completions=False,

        vllm_max_model_length=26384,  
        vllm_gpu_memory_utilization=0.3,

        learning_rate=1e-5,


        logging_steps=1,
        save_strategy="epoch",
        save_steps=100,
        report_to=["tensorboard"],
        logging_dir=os.path.join(runtime_config["output_dir"], "tb"),
        do_eval=eval_dataset is not None,
        eval_strategy=evaluation_strategy,
        per_device_eval_batch_size=4,
        num_generations_eval=1,


        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs=None,
    )


    from peft import LoraConfig

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"""
   训练配置:
   - Epochs: {train_config.num_train_epochs}
   - Batch size: {train_config.per_device_train_batch_size}
   - Generations per prompt: {train_config.num_generations}
   - Learning rate: {train_config.learning_rate}
   - vLLM加速: {train_config.use_vllm} (mode: {train_config.vllm_mode})
   - LoRA: r={peft_config.r}, alpha={peft_config.lora_alpha}
   - Sampling: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, min_p={args.min_p}, rep_penalty={args.repetition_penalty}
    """)


    print("\n[5/5] 创建 GRPO Trainer...")
    print(f"   使用模型: {runtime_config['model_name']}")

    trainer = AlignedGRPOTrainer(
        model=runtime_config["model_name"],
        reward_funcs=[
            reward_answer_accuracy,
            reward_format_correctness,
            reward_qclr_progressive,
            reward_qclr_terminal_soft,
            reward_qclr_query_penalty,
        ],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        rollout_func=rollout_func,
        args=train_config,
        peft_config=peft_config,
    )
    trainer.tools = FLUID_TOOLS
    trainer.temperature = args.temperature
    trainer.top_p = args.top_p
    trainer.top_k = args.top_k
    trainer.min_p = args.min_p
    trainer.repetition_penalty = args.repetition_penalty
    trainer.sampling_seed = args.sampling_seed


    print("\n[LoRA] 验证可训练参数...")
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    trainable_ratio = (trainable_params / total_params) if total_params else 0.0
    print(f"  - Trainable params: {trainable_params} / {total_params} ({trainable_ratio:.4%})")
    trainable_names = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
    print(f"  - Trainable modules (head): {trainable_names[:5]}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()


    print("\n" + "=" * 80)
    print("开始 GRPO 训练...")
    print("=" * 80)
    print("\n重要提醒:")
    print("  - 训练的是 VLM 模型，使用真实3DGS渲染器")
    print("  - 渲染器提供真实流体场图像")
    print("  - QA数据提供Ground Truth答案\n")

    trainer.train()


    print("\n[训练完成] 保存模型...")
    trainer.save_model(f"{runtime_config['output_dir']}/final")
    print(f"模型已保存到: {runtime_config['output_dir']}/final")

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
