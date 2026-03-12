

import builtins
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from trl import GRPOTrainer

import re

from grpo_parsing import parse_answer, parse_final_answer, parse_query_plan, parse_tool_calls
from grpo_rewards import compute_answer_similarity
from grpo_tools import execute_tool
from real_fluid_renderer import RealFluidRenderer


DEFAULT_OUTPUT_DIR = "./outputs/real_fluid_vlm_agent"


def generate_with_vllm(
    trainer,
    prompt_token_ids: list[int],
    max_new_tokens: int = 128,  
    multi_modal_data: dict | None = None,
    mm_processor_kwargs: dict | None = None,
    seed: int | None = None,
    allowed_token_ids: list[int] | None = None,
) -> dict:


    processor = trainer.processing_class
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    rt_tokenizer = tokenizer
    if hasattr(trainer, "llm") and trainer.llm is not None:
        try:
            rt_tokenizer = trainer.llm.get_tokenizer()
        except Exception:
            rt_tokenizer = tokenizer
    temperature = getattr(trainer, "temperature", 0.7)
    top_p = getattr(trainer, "top_p", 1.0)
    top_k = getattr(trainer, "top_k", -1)
    min_p = getattr(trainer, "min_p", 0.0)
    repetition_penalty = getattr(trainer, "repetition_penalty", 1.0)

    from vllm import SamplingParams

    llm = getattr(trainer, "llm", None)
    if llm is None:
        raise ValueError("vLLM 未初始化，无法生成")
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids 为空，无法生成")

    probe_logprobs = 1
    if allowed_token_ids:
        probe_logprobs = max(1, len(set(int(x) for x in allowed_token_ids)))


    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=0.0 if min_p is None else min_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_new_tokens,
        logprobs=probe_logprobs,
        allowed_token_ids=allowed_token_ids,
    )
    if seed is not None:
        sampling_params.seed = int(seed)

    prompt_payload = {"prompt_token_ids": prompt_token_ids}
    if multi_modal_data is not None:
        prompt_payload["multi_modal_data"] = multi_modal_data
    if mm_processor_kwargs is not None:
        prompt_payload["mm_processor_kwargs"] = mm_processor_kwargs


    outputs = llm.generate(
        [prompt_payload],
        sampling_params=sampling_params,
    )

    if not outputs or not outputs[0].outputs:
        return {
            "prompt_ids": prompt_token_ids,
            "completion_ids": [],
            "logprobs": [],
            "text": "",
        }

    output = outputs[0]
    completion = output.outputs[0]


    prompt_ids = list(output.prompt_token_ids) if output.prompt_token_ids else prompt_token_ids
    completion_ids = list(completion.token_ids) if completion.token_ids else []
    first_token_logprobs: dict[int, float] = {}


    logprobs = []
    if completion.logprobs is None:
        if allowed_token_ids:
            logprobs = [0.0 for _ in completion_ids]
        else:
            raise ValueError("vLLM 未返回 logprobs，无法校验采样token")
    else:
        if completion.logprobs:
            first_lp = completion.logprobs[0]
            if isinstance(first_lp, dict):
                for tid, obj in first_lp.items():
                    try:
                        first_token_logprobs[int(tid)] = float(obj.logprob)
                    except Exception:
                        continue
        if len(completion.logprobs) != len(completion_ids):
            raise ValueError(
                "vLLM logprobs 长度与采样 token 数不一致，无法校验对齐"
            )
        for token_id, lp in zip(completion_ids, completion.logprobs):
            if not lp or token_id not in lp:
                if allowed_token_ids:
                    logprobs.append(0.0)
                    continue
                raise ValueError("采样 token 未包含在 logprobs 中，拒绝继续训练")
            logprobs.append(lp[token_id].logprob)

    def _strip_special_suffix(ids: list[int], logps: list[float]) -> tuple[list[int], list[float]]:
        if not ids:
            return ids, logps
        eos_id = getattr(rt_tokenizer, "eos_token_id", None)
        pad_id = getattr(rt_tokenizer, "pad_token_id", None)
        while ids and (ids[-1] == eos_id or ids[-1] == pad_id):
            ids.pop()
            if logps:
                logps.pop()
        return ids, logps

    completion_ids, logprobs = _strip_special_suffix(completion_ids, logprobs)

    def _encode_tokens(text: str) -> list[int]:
        try:
            return rt_tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return rt_tokenizer.encode(text)
        except Exception:
            try:
                return rt_tokenizer(text, add_special_tokens=False).get("input_ids", [])
            except Exception:
                return []

    def _decode_for_round_trip(token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        candidates = []
        candidates.append(
            rt_tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        try:
            tokens = rt_tokenizer.convert_ids_to_tokens(token_ids)
            candidates.append(rt_tokenizer.convert_tokens_to_string(tokens))
        except Exception:
            pass
        for cand in candidates:
            re_ids = _encode_tokens(cand)
            if re_ids == token_ids:
                return cand
        return candidates[0]

    if completion_ids:
        raw_text = _decode_for_round_trip(completion_ids)
        try:
            text = rt_tokenizer.decode(completion_ids, skip_special_tokens=True)
        except Exception:
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    else:
        raw_text = completion.text
        text = completion.text

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "text": text,
        "raw_text": raw_text,
        "first_token_logprobs": first_token_logprobs,
    }


def create_rollout_func(
    device: str = "cuda",
    prompt_to_paths: dict | None = None,
    prompt_to_qa: dict | None = None,
    output_dir: str | None = None,
):


    renderer_cache = {}
    sys_prompt_printed = False
    prompt_to_paths = prompt_to_paths or {}
    prompt_to_qa = prompt_to_qa or {}
    timing_enabled = os.environ.get("GRPO_TIMING", "0") == "1"
    dump_context_enabled = os.environ.get("GRPO_DUMP_CONTEXT", "0") == "1"
    try:
        timing_every = max(1, int(os.environ.get("GRPO_TIMING_EVERY", "1")))
    except ValueError:
        timing_every = 1
    timing_counter = {"count": 0}

    def _safe_float_env(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            return float(default)

    qclr_enabled = os.environ.get("GRPO_QCLR_ENABLE", "1") == "1"
    qclr_lambda_p = max(0.0, _safe_float_env("GRPO_QCLR_LAMBDA_P", 0.15))

    def _resolve_output_dir() -> Path:
        if output_dir:
            return Path(output_dir)
        return Path(DEFAULT_OUTPUT_DIR)

    def get_renderer(gaussians_path: str, qa_path: str) -> RealFluidRenderer:

        if gaussians_path not in renderer_cache:
            print(f"    加载渲染器: {Path(gaussians_path).name}")
            renderer_cache[gaussians_path] = RealFluidRenderer(
                gaussians_hdf5_path=gaussians_path,
                qa_json_path=qa_path,
                device=device,
            )
        return renderer_cache[gaussians_path]

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:


        nonlocal sys_prompt_printed
        accel = getattr(trainer, "accelerator", None)
        is_main_process = accel.is_main_process if accel is not None else True

        def _print(*args, **kwargs):
            if is_main_process:
                builtins.print(*args, **kwargs)

        print = _print
        probe_fail_once = {"reported": False}

        def _should_time() -> bool:
            if not timing_enabled:
                return False
            accel = getattr(trainer, "accelerator", None)
            if accel is not None and not accel.is_main_process:
                return False
            timing_counter["count"] += 1
            return timing_counter["count"] % timing_every == 0

        def _sampling_seed(prompt_idx: int, generation_idx: int, step_idx: int) -> int | None:
            base_seed = getattr(trainer, "sampling_seed", None)
            if base_seed is None:
                return None
            return int(base_seed + prompt_idx * 1000 + generation_idx * 10 + step_idx)

        def _extract_tool_calls_from_plan(plan_obj: dict) -> list[dict]:
            tools = plan_obj.get("tools") or []
            if not isinstance(tools, list):
                tools = []
            if not tools:
                legacy_steps = plan_obj.get("steps") or plan_obj.get("plan") or []
                if isinstance(legacy_steps, list):
                    tools = legacy_steps
            tool_calls = []
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                name = tool.get("name") or tool.get("tool")
                args = tool.get("args") or tool.get("arguments") or tool.get("parameters") or {}
                if not name:
                    continue
                if not isinstance(args, dict):
                    args = {}
                tool_calls.append({"name": str(name), "arguments": args})
            return tool_calls

        def _extract_reasoning_text_before_json(text: str) -> str:
            if not text:
                return ""
            cleaned = text.replace("<|im_end|>", "")
            brace_idx = cleaned.find("{")
            if brace_idx == -1:
                candidate = cleaned
            else:
                candidate = cleaned[:brace_idx]
            candidate = re.sub(r"```(?:json)?", "", candidate, flags=re.IGNORECASE)
            candidate = candidate.replace("```", "")
            return candidate.strip()

        def _validate_query_plan_for_qid(qid: str | None, plan_obj: dict) -> str | None:
            if not qid:
                return None
            qid = str(qid).strip().upper()
            tools = plan_obj.get("tools")
            if not isinstance(tools, list) or not tools:
                return "query_plan.tools must not be empty"

            for tool in tools:
                if not isinstance(tool, dict):
                    return "query_plan.tools must be an array of objects"
                name = tool.get("name") or tool.get("tool")
                if not name:
                    return "each query_plan tool must include name"

            tool_calls = _extract_tool_calls_from_plan(plan_obj)
            if not tool_calls:
                return "query_plan must include at least one tool"


            return None

        def _dump_context_tokens(history_ids: list[int], prefix: str, step: int) -> None:
            if not dump_context_enabled or tokenizer is None:
                return
            try:
                decoded = tokenizer.decode(history_ids, skip_special_tokens=False)
            except Exception:
                decoded = ""
            dump_dir = _resolve_output_dir() / "context_dumps"
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / f"{prefix}_step_{step:02d}.txt"
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(f"[len={len(history_ids)}]\n")
                f.write(decoded)

        def _get_tokenizer():
            if hasattr(trainer, "llm") and trainer.llm is not None:
                try:
                    return trainer.llm.get_tokenizer()
                except Exception:
                    return None
            return (
                trainer.processing_class.tokenizer
                if hasattr(trainer.processing_class, "tokenizer")
                else trainer.processing_class
            )

        tokenizer = _get_tokenizer()
        processor = getattr(trainer, "processing_class", None)
        use_vllm_image_placeholders = getattr(trainer, "llm", None) is not None
        mm_processor_kwargs: dict[str, object] = {}
        if processor is not None:
            image_processor = getattr(processor, "image_processor", None)
            if image_processor is not None:
                for key in ("size", "min_pixels", "max_pixels"):
                    value = getattr(image_processor, key, None)
                    if value is not None:
                        mm_processor_kwargs[key] = value

        def _get_special_token_id(token_name: str) -> int | None:
            if tokenizer is None:
                return None
            token_id = None
            try:
                token_id = tokenizer.convert_tokens_to_ids(token_name)
            except Exception:
                token_id = None
            if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
                ids = _encode_text(token_name)
                if len(ids) == 1:
                    token_id = ids[0]
            return token_id

        vision_start_id = getattr(tokenizer, "vision_start_token_id", None) or _get_special_token_id("<|vision_start|>")
        vision_end_id = getattr(tokenizer, "vision_end_token_id", None) or _get_special_token_id("<|vision_end|>")
        image_pad_id = (
            getattr(tokenizer, "image_pad_token_id", None)
            or getattr(tokenizer, "image_token_id", None)
            or _get_special_token_id("<|image_pad|>")
        )

        def _encode_text(text: str) -> list[int]:
            if tokenizer is None:
                return []
            try:
                return tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                return tokenizer.encode(text)
            except Exception:
                return []

        def _encode_role(role: str, content: str) -> list[int]:
            return _encode_text(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        assistant_prefix_ids = _encode_text("<|im_start|>assistant\n")
        assistant_suffix_ids = _encode_text("<|im_end|>\n")

        def _extend_history(
            history_ids: list[int],
            tokens: list[int],
            completion_ids: list[int],
            tool_mask: list[int],
            logprobs: list[float],
        ) -> None:
            if not tokens:
                return
            history_ids.extend(tokens)
            completion_ids.extend(tokens)
            tool_mask.extend([0] * len(tokens))
            logprobs.extend([0.0] * len(tokens))

        def _sync_completion_with_prompt_ids(
            base_prompt_ids: list[int],
            local_prompt_ids: list[int],
            vllm_prompt_ids: list[int],
            completion_ids: list[int],
            tool_mask: list[int],
            logprobs: list[float],
        ) -> tuple[list[int], list[int], list[float]]:
            prefix_len = len(base_prompt_ids)
            if prefix_len > len(local_prompt_ids) or prefix_len > len(vllm_prompt_ids):
                return completion_ids, tool_mask, logprobs
            vllm_suffix = vllm_prompt_ids[prefix_len:]
            if completion_ids == vllm_suffix:
                return completion_ids, tool_mask, logprobs

            allowed_insert_tokens = {
                token_id
                for token_id in (image_pad_id, vision_start_id, vision_end_id)
                if token_id is not None
            }
            if not allowed_insert_tokens:
                raise ValueError("无法校验插入token类型：未解析到图像相关特殊token id")

            base = completion_ids
            if len(tool_mask) != len(base):
                tool_mask = (tool_mask + [0] * len(base))[: len(base)]
            if len(logprobs) != len(base):
                logprobs = (logprobs + [0.0] * len(base))[: len(base)]

            new_completion: list[int] = []
            new_tool_mask: list[int] = []
            new_logprobs: list[float] = []

            i = 0
            for token in vllm_suffix:
                if i < len(base) and base[i] == token:
                    new_completion.append(token)
                    new_tool_mask.append(tool_mask[i])
                    new_logprobs.append(logprobs[i])
                    i += 1
                else:
                    if token not in allowed_insert_tokens:
                        raise ValueError(
                            f"检测到非图像占位token插入: token_id={token}"
                        )
                    new_completion.append(token)
                    new_tool_mask.append(0)
                    new_logprobs.append(0.0)

            return new_completion, new_tool_mask, new_logprobs

        def _is_subsequence(needle: list[int], haystack: list[int]) -> bool:
            if not needle:
                return True
            idx = 0
            for token in haystack:
                if token == needle[idx]:
                    idx += 1
                    if idx == len(needle):
                        return True
            return False

        def _assert_history_alignment(
            history_ids: list[int],
            prompt_ids: list[int],
            completion_ids: list[int],
            label: str,
        ) -> None:
            expected = prompt_ids + completion_ids
            if history_ids != expected:
                raise ValueError(
                    f"{label}历史与prompt+completion不一致 "
                    f"(history={len(history_ids)}, prompt={len(prompt_ids)}, completion={len(completion_ids)})"
                )

        def _calc_image_token_counts(images: list[Image.Image]) -> list[int]:
            if not images:
                return []
            if processor is None:
                return [1] * len(images)
            try:
                merge_size = 1
                image_processor = getattr(processor, "image_processor", None)
                if image_processor is not None:
                    merge_size = getattr(image_processor, "merge_size", None) or 1
                merge_div = max(1, int(merge_size) ** 2)
                prompt_inputs = processor(
                    images=images,
                    text=[""] * len(images),
                    padding=True,
                    return_tensors="pt",
                    **mm_processor_kwargs,
                )
                grid = prompt_inputs.get("image_grid_thw")
                if grid is None:
                    return [1] * len(images)
                return [max(1, int(g.prod().item()) // merge_div) for g in grid]
            except Exception:
                return [1] * len(images)

        def _build_tool_response_block(tool_items: list[tuple[str, object]]) -> tuple[str, list[Image.Image], list[int]]:
            blocks = []
            blocks_ids: list[int] = []
            images: list[Image.Image] = []
            image_items = [item for item in tool_items if isinstance(item[1], Image.Image)]
            if use_vllm_image_placeholders:

                image_counts = [1] * len(image_items)
            else:
                image_counts = _calc_image_token_counts([item[1] for item in image_items])
            count_iter = iter(image_counts)
            for tool_name, tool_result in tool_items:
                if isinstance(tool_result, Image.Image):
                    images.append(tool_result)
                    pad_count = max(1, next(count_iter, 1))
                    pad_tokens = "<|image_pad|>" * pad_count
                    content = "<|vision_start|>" + pad_tokens + "<|vision_end|>\n" + f"{tool_name} completed"
                    if vision_start_id is not None and vision_end_id is not None and image_pad_id is not None:
                        content_ids = (
                            [vision_start_id]
                            + [image_pad_id] * pad_count
                            + [vision_end_id]
                            + _encode_text("\n" + f"{tool_name} completed")
                        )
                    else:
                        content_ids = _encode_text(content)
                else:
                    content = json.dumps(tool_result, ensure_ascii=False)
                    content_ids = _encode_text(content)
                blocks.append("<tool_response>\n" + content + "\n</tool_response>")
                blocks_ids.extend(_encode_text("<tool_response>\n"))
                blocks_ids.extend(content_ids)
                blocks_ids.extend(_encode_text("\n</tool_response>"))
            return "\n".join(blocks), images, blocks_ids

        def _estimate_image_tokens(images: list[Image.Image]) -> tuple[int | None, list[tuple[int, int, int]]]:
            if not images:
                return 0, []
            counts = _calc_image_token_counts(images)
            if not counts:
                return None, []
            details = []
            total = 0
            for img, tokens in zip(images, counts, strict=False):
                try:
                    width, height = img.size
                except Exception:
                    continue
                total += tokens
                details.append((width, height, tokens))
            return total, details

        def _count_image_pad(tokens: list[int]) -> int:
            if image_pad_id is None:
                return 0
            return sum(1 for t in tokens if t == image_pad_id)

        def _assert_image_token_alignment(
            history_ids: list[int],
            images: list[Image.Image],
            label: str,
        ) -> None:
            if not images:
                return
            if image_pad_id is None:
                raise ValueError("无法获取 image_pad_token_id，无法校验图像token对齐")
            expected_counts = _calc_image_token_counts(images)
            expected = sum(expected_counts)
            actual = _count_image_pad(history_ids)
            if actual != expected:
                raise ValueError(
                    f"{label}图像token数量不一致 (actual={actual}, expected={expected}, images={len(images)})"
                )

        def _get_vllm_engine_stats() -> dict[str, int | None]:
            stats: dict[str, int | None] = {}
            llm = getattr(trainer, "llm", None)
            if llm is None:
                return stats
            engine = getattr(llm, "llm_engine", None)
            if engine is None:
                return stats
            vllm_config = getattr(engine, "vllm_config", None)
            model_config = getattr(vllm_config, "model_config", None) if vllm_config is not None else None
            scheduler_config = getattr(vllm_config, "scheduler_config", None) if vllm_config is not None else None
            cache_config = getattr(vllm_config, "cache_config", None) if vllm_config is not None else None
            if model_config is None:
                model_config = getattr(engine, "model_config", None)
            if scheduler_config is None:
                scheduler_config = getattr(engine, "scheduler_config", None)
            if cache_config is None:
                cache_config = getattr(engine, "cache_config", None)
            if model_config is not None:
                stats["engine_max_model_len"] = getattr(model_config, "max_model_len", None) or getattr(
                    model_config, "max_seq_len", None
                )
            if scheduler_config is not None:
                stats["max_num_batched_tokens"] = getattr(scheduler_config, "max_num_batched_tokens", None)
                stats["max_num_seqs"] = getattr(scheduler_config, "max_num_seqs", None)
            if cache_config is not None:
                stats["block_size"] = getattr(cache_config, "block_size", None)
                stats["num_gpu_blocks"] = getattr(cache_config, "num_gpu_blocks", None) or getattr(
                    cache_config, "num_gpu_blocks_override", None
                )
            return stats

        def _plane_size(renderer: RealFluidRenderer, plane: str) -> int:
            plane = plane.lower() if plane else "xy"
            if plane == "xy":
                return renderer.num_z
            if plane == "yz":
                return renderer.num_x
            return renderer.num_y

        def _coord_to_index(coord: float, size: int, mode: str) -> int:
            if mode == "slice":
                idx = int(round(coord * size + 0.5))
            else:
                idx = int(round(coord * size - 0.5))
            return max(0, min(size - 1, idx))

        def _as_float(value) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _as_int(value) -> int | None:
            val = _as_float(value)
            if val is None:
                return None
            return int(round(val))

        def _is_option_letter(text: str | None) -> bool:
            if not text:
                return False
            return text.strip().upper() in {"A", "B", "C", "D", "E"}

        def _map_text_to_option(pred_text: str, options: dict | None) -> tuple[str | None, float]:
            if not pred_text or not options:
                return None, 0.0
            best_letter = None
            best_score = 0.0
            for letter, option_text in options.items():
                score = compute_answer_similarity(pred_text, str(option_text))
                if score > best_score:
                    best_score = score
                    best_letter = str(letter).strip().upper()
            return best_letter, best_score

        def _resolve_choice_letters(options: dict | None) -> list[str]:
            if not isinstance(options, dict) or not options:
                return ["A", "B", "C", "D"]
            letters = [str(k).strip().upper() for k in options.keys()]
            letters = [k for k in letters if len(k) == 1 and "A" <= k <= "Z"]
            letters = sorted(set(letters))
            return letters if letters else ["A", "B", "C", "D"]

        def _build_choice_token_candidates(choice_letters: list[str]) -> dict[str, list[int]]:
            candidates: dict[str, list[int]] = {}
            if tokenizer is None:
                return candidates
            for letter in choice_letters:
                ids = []
                for variant in (letter, f" {letter}", f"\n{letter}"):
                    token_ids = _encode_text(variant)
                    if len(token_ids) == 1:
                        token_id = int(token_ids[0])
                        if token_id not in ids:
                            ids.append(token_id)
                if ids:
                    candidates[letter] = ids
            return candidates

        def _probe_choice_distribution(
            history_ids: list[int],
            images: list[Image.Image],
            choice_letters: list[str],
            token_candidates: dict[str, list[int]],
            step_seed: int | None,
        ) -> dict[str, float] | None:
            if not qclr_enabled:
                return None
            if not choice_letters or not token_candidates:
                return None

            probe_suffix = "Based on all the information gathered above, the answer is"
            probe_prompt_ids = history_ids + _encode_text(probe_suffix)
            if not probe_prompt_ids:
                return None

            allowed_ids = []
            for letter in choice_letters:
                for token_id in token_candidates.get(letter, []):
                    if token_id not in allowed_ids:
                        allowed_ids.append(token_id)
            if not allowed_ids:
                return None

            def _run_probe(probe_images: list[Image.Image]) -> dict:
                return generate_with_vllm(
                    trainer,
                    probe_prompt_ids,
                    max_new_tokens=1,
                    multi_modal_data={"image": probe_images} if probe_images else None,
                    mm_processor_kwargs=mm_processor_kwargs or None,
                    seed=step_seed,
                    allowed_token_ids=allowed_ids,
                )

            try:
                step_output = _run_probe(images)
            except Exception as exc:
                msg = str(exc).lower()
                is_oom = ("out of memory" in msg) or ("cuda error" in msg)
                if is_oom and images:
                    try:
                        step_output = _run_probe([])
                        if not probe_fail_once["reported"]:
                            print("    [QCLR] probe显存不足，已降级为无图probe")
                            probe_fail_once["reported"] = True
                    except Exception as exc_retry:
                        if not probe_fail_once["reported"]:
                            print(f"    [QCLR] probe失败，已禁用当前样本QCLR: {exc_retry}")
                            probe_fail_once["reported"] = True
                        return None
                else:
                    if not probe_fail_once["reported"]:
                        print(f"    [QCLR] probe失败，已禁用当前样本QCLR: {exc}")
                        probe_fail_once["reported"] = True
                    return None

            completion_ids = list(step_output.get("completion_ids") or [])
            logprobs = list(step_output.get("logprobs") or [])
            first_token_logprobs = step_output.get("first_token_logprobs") or {}
            if not completion_ids:
                return None

            token_id = int(completion_ids[0])
            picked_logp = float(logprobs[0]) if logprobs else 0.0
            logits: dict[str, float] = {}
            for letter in choice_letters:
                cands = token_candidates.get(letter, [])
                if not cands:
                    continue
                best = None
                for tid in cands:
                    if tid in first_token_logprobs:
                        val = float(first_token_logprobs[tid])
                        best = val if best is None else max(best, val)
                if best is None:
                    if token_id in cands:
                        best = picked_logp
                    else:
                        best = -20.0
                logits[letter] = best

            if not logits:
                return None
            max_logit = max(logits.values())
            exps = {k: math.exp(v - max_logit) for k, v in logits.items()}
            denom = sum(exps.values())
            if denom <= 0:
                return None
            probs = {k: exps[k] / denom for k in logits}
            return probs

        def _safe_probe_logprob(
            history_ids: list[int],
            images: list[Image.Image],
            choice_letters: list[str],
            token_candidates: dict[str, list[int]],
            correct_letter: str | None,
            step_seed: int | None,
        ) -> float | None:
            if not qclr_enabled:
                return None
            if not correct_letter or correct_letter not in choice_letters:
                return None
            probs = _probe_choice_distribution(
                history_ids=history_ids,
                images=images,
                choice_letters=choice_letters,
                token_candidates=token_candidates,
                step_seed=step_seed,
            )
            if not probs:
                return None
            p = float(probs.get(correct_letter, 0.0))
            p = max(1e-8, min(1.0, p))
            return math.log(p)

        def _normalize_plane_name(plane: str | None) -> str:
            if not plane:
                return "xy"
            plane = str(plane).lower()
            if plane in {"xy", "yz", "xz"}:
                return plane
            if plane in {"xyz", "xyplane"}:
                return "xy"
            if plane in {"yzplane"}:
                return "yz"
            if plane in {"xzplane"}:
                return "xz"
            return "xy"

        def _normalize_quantity_name(quantity: str | None) -> str:
            if not quantity:
                return "velocity"
            q = str(quantity).strip().lower()
            if q in {"vorticity", "omega", "curl", "vort", "vorticity_magnitude", "omega_mag", "vort_mag"}:
                return "vorticity"
            return "velocity"

        def _parse_bound_pair(value) -> tuple[float | None, float | None]:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return _as_float(value[0]), _as_float(value[1])
            if isinstance(value, dict):
                low = value.get("low") if "low" in value else value.get("min")
                high = value.get("high") if "high" in value else value.get("max")
                return _as_float(low), _as_float(high)
            return None, None

        def _bound_to_index(value: float | None, size: int) -> int | None:
            if value is None:
                return None
            if size <= 1:
                return 0
            if 0.0 <= value <= 1.0:
                return max(0, min(size - 1, int(round(value * size - 0.5))))
            return max(0, min(size - 1, int(round(value))))

        def _normalize_roi_indices_2d(plane: str, shape: tuple[int, int], roi) -> tuple[int, int, int, int] | None:
            if roi is None:
                return None
            if isinstance(roi, (list, tuple)) and len(roi) >= 2:
                b0 = _parse_bound_pair(roi[0])
                b1 = _parse_bound_pair(roi[1])
            elif isinstance(roi, dict):
                axis_map = {
                    "xy": ("x", "y"),
                    "yz": ("y", "z"),
                    "xz": ("x", "z"),
                }
                a0, a1 = axis_map.get(plane, ("x", "y"))
                b0 = _parse_bound_pair(roi.get(a0))
                b1 = _parse_bound_pair(roi.get(a1))
            else:
                return None

            l0, h0 = b0
            l1, h1 = b1
            n0, n1 = int(shape[0]), int(shape[1])
            lo0 = _bound_to_index(l0, n0)
            hi0 = _bound_to_index(h0, n0)
            lo1 = _bound_to_index(l1, n1)
            hi1 = _bound_to_index(h1, n1)
            if lo0 is None and hi0 is None and lo1 is None and hi1 is None:
                return None
            if lo0 is None:
                lo0 = 0
            if hi0 is None:
                hi0 = n0 - 1
            if lo1 is None:
                lo1 = 0
            if hi1 is None:
                hi1 = n1 - 1
            if lo0 > hi0:
                lo0, hi0 = hi0, lo0
            if lo1 > hi1:
                lo1, hi1 = hi1, lo1
            return lo0, hi0, lo1, hi1

        def _resolve_slice_index_from_args(renderer: RealFluidRenderer, args: dict, plane: str) -> int:
            size = _plane_size(renderer, plane)
            idx = _as_int(args.get("slice_index"))
            if idx is not None:
                idx0 = idx - 1 if idx >= 1 else idx
                return max(0, min(size - 1, int(idx0)))
            coord = _as_float(args.get("slice_coord"))
            if coord is not None:
                return _coord_to_index(coord, size, mode="slice")
            return size // 2

        def _compute_gradient_weight(renderer: RealFluidRenderer, tool_name: str, args: dict) -> float:
            if not qclr_enabled or qclr_lambda_p <= 0:
                return 0.0
            if tool_name not in {"slice_stats", "slice_compare", "slice_view_colorbar"}:
                return 0.0
            try:
                plane = _normalize_plane_name(args.get("plane"))
                quantity = _normalize_quantity_name(args.get("quantity"))
                if tool_name == "slice_compare":
                    t0 = _as_int(args.get("time_a"))
                    t1 = _as_int(args.get("time_b"))
                    time_raw = t0 if t0 is not None else t1
                else:
                    time_raw = _as_int(args.get("time"))
                if time_raw is None:
                    time_raw = 1
                time_idx = max(0, min(renderer.num_timesteps - 1, int(time_raw) - 1))
                slice_index = _resolve_slice_index_from_args(renderer, args, plane)
                if quantity == "vorticity":
                    field = renderer._get_vorticity_field(time_idx, plane, slice_index)
                else:
                    field = renderer._get_velocity_field(time_idx, plane, slice_index)
                field = np.asarray(field, dtype=np.float32)
                if field.ndim != 2 or field.size == 0:
                    return 0.0
                grad0, grad1 = np.gradient(field)
                grad_mag = np.sqrt(grad0 * grad0 + grad1 * grad1)

                roi = args.get("roi")
                bounds = _normalize_roi_indices_2d(plane, grad_mag.shape, roi)
                if bounds:
                    lo0, hi0, lo1, hi1 = bounds
                    region = grad_mag[lo0:hi0 + 1, lo1:hi1 + 1]
                else:
                    region = grad_mag
                if region.size == 0:
                    return 0.0
                g_mean = float(np.mean(region))
                g_max = float(np.max(grad_mag))
                if g_max <= 1e-8:
                    return 0.0
                return max(0.0, min(1.0, g_mean / (g_max + 1e-8)))
            except Exception:
                return 0.0


        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_tool_masks = []
        all_images = []


        answer_rewards = []
        final_answers = []
        qa_options_list = []
        pred_label_list = []
        true_label_list = []
        qclr_prog_rewards = []
        qclr_soft_rewards = []
        qclr_tool_counts = []
        qclr_s0_values = []
        qclr_s_last_values = []


        num_repeats = trainer.num_generations if trainer.model.training else trainer.num_generations_eval
        repeat_counts: dict[str, int] = {}

        print(f"\n{'─'*60}")
        print(f"Rollout: {len(prompts)} prompts")
        print(f"{'─'*60}")

        for idx, question in enumerate(prompts):
            do_timing = _should_time()
            t_prompt_start = time.perf_counter() if do_timing else 0.0
            t_gen_total = 0.0
            t_tool_total = 0.0
            t_reward_total = 0.0

            if question not in prompt_to_paths:
                raise ValueError(f"问题未找到匹配数据: {question}")

            gaussians_path, qa_path = prompt_to_paths[question]

            renderer_cached = gaussians_path in renderer_cache
            t_renderer_start = time.perf_counter() if do_timing else 0.0
            renderer = get_renderer(gaussians_path, qa_path)
            if do_timing and not renderer_cached:
                print(f"    [Timing] renderer_load={time.perf_counter() - t_renderer_start:.3f}s")


            qa_item = prompt_to_qa.get(question, {}) if isinstance(prompt_to_qa, dict) else {}
            options_text = ""
            options = qa_item.get("options") if isinstance(qa_item, dict) else None
            qid = str(qa_item.get("id", "")).strip().upper() if isinstance(qa_item, dict) else ""
            correct_letter = str(qa_item.get("answer", "")).strip().upper() if isinstance(qa_item, dict) else ""
            if len(correct_letter) != 1:
                correct_letter = ""
            choice_letters = _resolve_choice_letters(options)
            choice_token_candidates = _build_choice_token_candidates(choice_letters)
            if isinstance(options, dict) and options:
                option_lines = [f"{key}: {options[key]}" for key in sorted(options.keys())]
                options_text = "Options:\n" + "\n".join(option_lines) + "\n\n"

            system_prompt = f"""You are a fluid-field analysis assistant. You need to analyze 3D phsical field data with time steps and answer questions.

[Data Info]
- Time step range: [1, {renderer.num_timesteps}], 1 = initial step, {renderer.num_timesteps} = final step
- Spatial grid: Z x X x Y = {renderer.num_z} x {renderer.num_x} x {renderer.num_y}
- Slice planes:
  - xy: fixed z, shows x-y plane
  - yz: fixed x, shows y-z plane
  - xz: fixed y, shows x-z plane

[Tools (args & returns)]
1) slice_stats(time, plane, slice_coord, quantity, point_index | point_coord, roi?)
   - Purpose: statistics for a single slice (max/min location, mean/std, gradients, point value).
   - Args (types/range):
     - time: int, 1-based step index
     - plane: str in {"xy","yz","xz"}
     - slice_coord: float in [0,1]
     - quantity: str in {"velocity","vorticity"}
     - point_coord: [float,float] in [0,1] or point_index: [int,int] (only when a point is specified)
     - roi: 2D region dict (each axis [low,high], coord or index)
   - Returns (subset):
     - mean/std/cv: float
     - max_value/min_value: float
     - max_coord/min_coord: dict with axis keys, e.g. xy={{"x":0.25,"y":0.75}}, yz={{"y":0.25,"z":0.75}}, xz={{"x":0.25,"z":0.75}}
     - max_index/min_index: [int,int] with axis order xy->[x,y], yz->[y,z], xz->[x,z]
     - point_coord/point_index/point_value: if point_* provided (point_coord is a dict with axis keys)
     - grad_x/grad_y/grad_median/grad_std: if point_* provided
     - energy_range_ratio: only for quantity=velocity and plane=xy
     - triangle_means/triangle_diff: for quantity=velocity on any plane
     - omega_z_stats(mean/std): only for quantity=vorticity and plane=xy

2) slice_compare(time_a, time_b, plane, slice_coord, quantity, time_indices?, roi?)
   - Purpose: compare two time steps on the same slice (correlation, mean change, energy shift).
   - Args:
     - time_a/time_b: int (1-based)
     - time_indices: list[int], if len>=3 returns max_values
     - other args same as slice_stats
   - Returns (subset):
     - correlation, mean_a, mean_b: float
     - mean_change: float (velocity+xy only)
     - energy_center_shift {{"dx":float, "dy":float}}: float (velocity on any plane, normalized axis0/axis1 coords)
     - energy_cv_change: float (velocity+xy only)
     - max_values: [float,float,float] (if time_indices provided)
     - vort_max_distance: float (vorticity+xy only)

3) cube_components(time, center_coord, radius, time_indices?, roi?)
   - Purpose: average velocity component magnitudes in a 3D cube.
   - Args:
     - time: int (1-based)
     - center_coord: [float,float,float] in [0,1]
     - radius: int>=0 (index radius) or float in (0,1] (normalized radius)
     - time_indices: list[int] (for 3-step point comparison)
     - roi: 3D region dict (x/y/z each [low,high])
   - Returns (subset):
     - mean_abs_components: [x,y,z]
     - mean_speed/local_spread/global_spread: float
     - center_coord: [float,float,float]
     - point_values/point_series_std: if time_indices provided

4) plane_uniformity(time, slice_indices | slice_coords, quantity, roi?, plane_rois?)
   - Purpose: compare CV (uniformity) across three planes.
   - Args:
     - time: int (1-based)
     - slice_indices: {{"xy":int, "yz":int, "xz":int}} or slice_coords: {{"xy":float, "yz":float, "xz":float}}
     - quantity: str in {"velocity","vorticity"}
     - roi: global 2D region; plane_rois: per-plane ROI
   - Returns:
     - cv: {{"xy":float, "yz":float, "xz":float}}

5) vorticity_orientation(time, xy_slice_coord, yz_slice_coord, xz_slice_coord, roi?)
  - Purpose: principal-axis / anisotropy for high-vorticity regions (optional ROI).
  - Args:
    - time: int (1-based)
    - xy/yz/xz slice_coord (defaults to center if omitted)
    - roi: 3D region dict (x/y/z each [low,high])
  - Notes:
    - If slice_coord is provided, stats are computed on the slice (and ROI if provided).
  - Returns (subset):
    - variances: [var_x,var_y,var_z]
    - anisotropy_ratio, baseline_ratio: float
    - displacement/disp_mag/disp_threshold: if displacement is available (XY+XZ or XY+YZ组合)

6) slice_view_colorbar(time, plane, slice_coord, quantity)
   - Purpose: render a slice image with colorbar for velocity/vorticity magnitude.
   - Args:
     - time: int
     - plane/slice_coord/quantity: same as slice_stats
   - Returns:
     - image (RGB) with colorbar appended on the right
     - Image coordinate system:
       - Origin at top-left pixel.
       - Axis-0 (H / rows) increases downward.
       - Axis-1 (W / cols) increases rightward.
     - Plane ↔ image axis mapping (H, W):
       - xy slice (fixed z): H ↔ x, W ↔ y
       - yz slice (fixed x): H ↔ y, W ↔ z
       - xz slice (fixed y): H ↔ x, W ↔ z
     - Note: interpret spatial locations with this mapping (not Cartesian lower-left origin).

[Parameter Mapping (must follow)]
- time/time_a/time_b: step from the question (1-based)
- plane: "xy" / "yz" / "xz"
- slice_coord: float in [0,1]
- point_coord: [float,float] in [0,1]
  - xy plane -> [x, y]
  - yz plane -> [y, z]
  - xz plane -> [x, z]
- point_index: [int,int] (same order as point_coord; ranges match the two axes)
- roi: 2D/3D range dict, do NOT use point-list form
  - 2D: xy uses {{"x":[low,high], "y":[low,high]}}; yz uses {{"y":[low,high], "z":[low,high]}}; xz uses {{"x":[low,high], "z":[low,high]}}
  - 3D: {{"x":[low,high], "y":[low,high], "z":[low,high]}}
- Coordinate keys for max_coord/min_coord/point_coord: xy -> {{"x","y"}}, yz -> {{"y","z"}}, xz -> {{"x","z"}}

[Workflow]
1. First round: output a "reasoning steps" (short text, must not include query_plan) describing which tools to call and where parameters come from (do not give the final answer).
2. Second round: output query_plan JSON based on the reasoning steps (must include tool calls).
3. The system executes the plan and returns tool results.
4. Third round: output the final answer JSON (do not call tools; must include steps).

[Strict Stage Output Policy]
- Round 1 only: output plain reasoning text (no JSON object, no code block, no query_plan keyword).
- Round 2 only: output query_plan JSON.
- Round 3 only: output final_answer JSON.

[query_plan JSON]
{{
  "tools": [
    {{"name":"tool_name","args":{{...}}}},
    {{"name":"tool_name_2","args":{{...}}}}
  ]
}}
Each tool call must be on its own line inside tools, and each item can only contain name and args.
Second round only: Output ONLY the JSON object, no comments or extra text.

[Final Answer JSON]
{{"type": "final_answer", "steps": ["Step1: ...", "Step2: ..."], "answer": "Answer"}}
Third round only: Output ONLY this JSON object format.

[Answer Requirements]
- Single-choice questions: output only the option letter, e.g., "A" / "B"
- Must include a steps field to record your chain of thoughts (2-4 items), short structured sentences that compare tool results
- Do not output full chain-of-thought or long explanations
- DO NOT rely on images to make a judgment; always call other tools to support your reasoning.
- The answer field must not include coordinates; steps may include necessary numeric comparisons"""

            user_prompt = (
                f"Question: {question}\n\n{options_text}"
                "Please first output a reasoning steps (short text, not JSON, no final answer, at most 2 slice_view_colorbar calls). "
            )
            base_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            repeat_counts[question] = repeat_counts.get(question, 0) + 1
            generation_index = repeat_counts[question]

            gen_tag = f" G{generation_index}/{num_repeats}" if num_repeats > 1 else ""
            print(f"\n  [{idx + 1}/{len(prompts)}]{gen_tag} Q: {question[:50]}{'...' if len(question) > 50 else ''}")

            messages = list(base_messages)
            tool_calls_made = []  
            plan_ready = False
            plan_summary_ready = False
            plan_summary_text = ""
            tool_results_for_tokens: list[tuple[str, object]] = []
            answer_text = ""
            prompt_ids_this_gen = []
            completion_ids_this_gen = []
            logprobs_this_gen = []
            tool_mask_this_gen = []
            step_completion_lengths = []
            history_ids_cache: list[int] = []
            mm_images_cache: list[Image.Image] = []
            qclr_step_scores: list[float] = []
            qclr_s_values: list[float] = []
            qclr_probe_ready = bool(qclr_enabled and correct_letter and choice_token_candidates)


            trajectory = {
                "question": question,
                "gaussians": Path(gaussians_path).name,
                "generation": generation_index,
                "generation_total": num_repeats,
                "steps": []
            }


            trajectory["steps"].append({"role": "system", "content": "(system prompt)"})
            trajectory["steps"].append({"role": "user", "content": question})


            if not sys_prompt_printed:
                print(f"    [Sys] (流体分析助手, T={renderer.num_timesteps}, 6 tools)")
                sys_prompt_printed = True
            print(f"    [User] {question[:60]}{'...' if len(question) > 60 else ''}")

            history_ids_cache += _encode_role("system", system_prompt)
            history_ids_cache += _encode_role("user", user_prompt)
            history_ids_cache += assistant_prefix_ids
            if not history_ids_cache:
                raise ValueError("初始化 prompt_token_ids 失败")

            if qclr_probe_ready:
                s0 = _safe_probe_logprob(
                    history_ids=history_ids_cache,
                    images=mm_images_cache,
                    choice_letters=choice_letters,
                    token_candidates=choice_token_candidates,
                    correct_letter=correct_letter,
                    step_seed=_sampling_seed(idx, generation_index, -1),
                )
                if s0 is None:
                    qclr_probe_ready = False
                else:
                    qclr_s_values.append(float(s0))

            prompt_ids_this_gen = list(history_ids_cache)

            base_max_steps = 6
            max_query_plan_corrections = 3
            query_plan_corrections = 0
            max_steps = base_max_steps
            step = 0
            preparsed_plan_obj: dict | None = None


            while step < max_steps:
                vllm_stats = _get_vllm_engine_stats()
                max_model_len = vllm_stats.get("engine_max_model_len") or getattr(
                    getattr(trainer, "args", None), "vllm_max_model_length", None
                )
                max_num_seqs = vllm_stats.get("max_num_seqs")
                max_num_batched_tokens = vllm_stats.get("max_num_batched_tokens")
                block_size = vllm_stats.get("block_size")
                num_gpu_blocks = vllm_stats.get("num_gpu_blocks")
                cache_max_len_est = None
                if block_size and num_gpu_blocks and max_num_seqs:
                    cache_max_len_est = (num_gpu_blocks // max_num_seqs) * block_size
                est_img_tokens, img_details = _estimate_image_tokens(mm_images_cache)
                print(
                    "    [VLLM] prompt_tokens="
                    f"{len(history_ids_cache)}, images={len(mm_images_cache)}, "
                    f"est_image_tokens={est_img_tokens}, max_model_len={max_model_len}, "
                    f"cache_max_len_est={cache_max_len_est}, max_num_seqs={max_num_seqs}, "
                    f"max_num_batched_tokens={max_num_batched_tokens}"
                )
                if img_details:
                    for w, h, t in img_details:
                        print(f"    [VLLM] image_size={w}x{h}, est_tokens={t}")

                _dump_context_tokens(history_ids_cache, f"sample_{idx + 1}_gen_{generation_index}", step)
                t_gen_start = time.perf_counter() if do_timing else 0.0
                step_output = generate_with_vllm(
                    trainer,
                    history_ids_cache,
                    max_new_tokens=512,
                    multi_modal_data={"image": mm_images_cache} if mm_images_cache else None,
                    mm_processor_kwargs=mm_processor_kwargs or None,
                    seed=_sampling_seed(idx, generation_index, step),
                )
                if do_timing:
                    t_gen_total += time.perf_counter() - t_gen_start

                step_prompt_ids = list(step_output.get("prompt_ids") or history_ids_cache)
                step_completion_ids = list(step_output.get("completion_ids") or [])
                step_logprobs = list(step_output.get("logprobs") or [])

                _assert_history_alignment(
                    history_ids_cache,
                    prompt_ids_this_gen,
                    completion_ids_this_gen,
                    label="对齐前",
                )

                if step_prompt_ids != history_ids_cache:
                    local_history_ids = list(history_ids_cache)
                    if step_prompt_ids[: len(prompt_ids_this_gen)] != prompt_ids_this_gen:
                        raise ValueError("vLLM prompt 前缀发生变化，无法保证对齐")
                    if not _is_subsequence(local_history_ids, step_prompt_ids):
                        raise ValueError("vLLM prompt 发生非插入式变化，无法对齐")
                    local_len = len(history_ids_cache)
                    vllm_len = len(step_prompt_ids)
                    delta = local_len - vllm_len
                    if delta > 0:
                        missing_tokens = delta
                        missing_blocks = None
                        overflow_tokens = None
                        if block_size:
                            need_blocks = math.ceil(local_len / block_size)
                            have_blocks = math.ceil(vllm_len / block_size)
                            missing_blocks = max(0, need_blocks - have_blocks)
                        if cache_max_len_est is not None:
                            overflow_tokens = max(0, local_len - cache_max_len_est)
                        print(
                            "    [Align] 截断疑似: vLLM prompt 变短，已同步 "
                            f"(local={local_len}, vllm={vllm_len}, missing_tokens={missing_tokens}, "
                            f"missing_blocks={missing_blocks}, overflow_tokens={overflow_tokens}, block_size={block_size}, "
                            f"max_model_len={max_model_len}, cache_max_len_est={cache_max_len_est})"
                        )
                        drop_prefix = local_len - vllm_len
                        if local_history_ids[drop_prefix:] != step_prompt_ids:
                            raise ValueError("vLLM prompt 截断无法对齐本地历史")
                        if drop_prefix > 0:
                            if drop_prefix < len(prompt_ids_this_gen):
                                prompt_ids_this_gen = prompt_ids_this_gen[drop_prefix:]
                            else:
                                drop_prefix -= len(prompt_ids_this_gen)
                                prompt_ids_this_gen = []
                                if drop_prefix > 0:
                                    completion_ids_this_gen = completion_ids_this_gen[drop_prefix:]
                                    tool_mask_this_gen = tool_mask_this_gen[drop_prefix:]
                                    logprobs_this_gen = logprobs_this_gen[drop_prefix:]
                    else:
                        print(
                            "    [Align] prompt 扩展: vLLM 展开多模态 token，已同步 "
                            f"(local={local_len}, vllm={vllm_len}, delta={-delta})"
                        )
                    history_ids_cache = list(step_prompt_ids)
                    completion_ids_this_gen, tool_mask_this_gen, logprobs_this_gen = (
                        _sync_completion_with_prompt_ids(
                            prompt_ids_this_gen,
                            local_history_ids,
                            history_ids_cache,
                            completion_ids_this_gen,
                            tool_mask_this_gen,
                            logprobs_this_gen,
                        )
                    )
                    _assert_history_alignment(
                        history_ids_cache,
                        prompt_ids_this_gen,
                        completion_ids_this_gen,
                        label="对齐后",
                    )

                _assert_image_token_alignment(
                    history_ids_cache,
                    mm_images_cache,
                    label="对齐检查",
                )


                if step_logprobs and len(step_logprobs) != len(step_completion_ids):
                    step_logprobs = (step_logprobs + [0.0] * len(step_completion_ids))[: len(step_completion_ids)]
                if not step_logprobs:
                    step_logprobs = [0.0] * len(step_completion_ids)
                completion_ids_this_gen += step_completion_ids
                tool_mask_this_gen += [1] * len(step_completion_ids)
                logprobs_this_gen += step_logprobs
                history_ids_cache = history_ids_cache + step_completion_ids
                _assert_history_alignment(
                    history_ids_cache,
                    prompt_ids_this_gen,
                    completion_ids_this_gen,
                    label="step完成后",
                )
                has_suffix = (
                    assistant_suffix_ids
                    and len(step_completion_ids) >= len(assistant_suffix_ids)
                    and step_completion_ids[-len(assistant_suffix_ids):] == assistant_suffix_ids
                )
                if not has_suffix:
                    _extend_history(
                        history_ids_cache,
                        assistant_suffix_ids,
                        completion_ids_this_gen,
                        tool_mask_this_gen,
                        logprobs_this_gen,
                    )
                step_completion_lengths.append(len(step_completion_ids))
                step += 1
                raw_text = step_output.get("raw_text", step_output["text"])
                output_text = step_output["text"].strip()
                messages.append({"role": "assistant", "content": raw_text})

                if not plan_summary_ready:
                    plan_obj = parse_query_plan(raw_text, allow_incomplete=True) or parse_query_plan(
                        step_output["text"],
                        allow_incomplete=True,
                    )
                    if plan_obj:
                        plan_summary_text = _extract_reasoning_text_before_json(raw_text)
                        if not plan_summary_text:
                            plan_summary_text = _extract_reasoning_text_before_json(step_output["text"])
                        if not plan_summary_text:
                            plan_summary_text = "Use the question requirements to produce tool calls."
                        plan_summary_ready = True
                        trajectory["steps"].append({"role": "assistant", "content": f"[PlanSummaryFromMixed] {plan_summary_text}"})
                    else:
                        plan_summary_text = output_text
                        plan_summary_ready = True
                        trajectory["steps"].append({"role": "assistant", "content": f"[PlanSummary] {plan_summary_text}"})

                    plan_prompt = (
                        "This is your reasoning steps: "
                        f"{plan_summary_text}\n"
                        "Please output the query_plan JSON based on it (containing all necessary tool calls):\n"
                        "{\"tools\":[\n"
                        "  {\"name\":\"tool_name\",\"args\":{...}},\n"
                        "  {\"name\":\"tool_name_2\",\"args\":{...}}\n"
                        "]}. "
                        "Each tool call must be on its own line. "
                        "Do not output the final answer, at most 2 slice_view_colorbar calls."
                    )
                    messages.append({"role": "user", "content": plan_prompt})
                    plan_prompt_ids = _encode_role("user", plan_prompt)
                    _extend_history(
                        history_ids_cache,
                        plan_prompt_ids,
                        completion_ids_this_gen,
                        tool_mask_this_gen,
                        logprobs_this_gen,
                    )
                    _extend_history(
                        history_ids_cache,
                        assistant_prefix_ids,
                        completion_ids_this_gen,
                        tool_mask_this_gen,
                        logprobs_this_gen,
                    )
                    trajectory["steps"].append({"role": "user", "content": plan_prompt})
                    continue

                if not plan_ready:
                    plan_obj = preparsed_plan_obj or (
                        parse_query_plan(raw_text, allow_incomplete=True)
                        or parse_query_plan(step_output["text"], allow_incomplete=True)
                    )
                    preparsed_plan_obj = None
                    if not plan_obj:
                        correction_prompt = (
                            "Invalid query_plan JSON format. Please output ONLY a JSON object with "
                            "a non-empty tools array. Example:\n"
                            "{\"tools\":[\n"
                            "  {\"name\":\"tool_name\",\"args\":{...}},\n"
                            "  {\"name\":\"tool_name_2\",\"args\":{...}}\n"
                            "]}. "
                            "Each tool call must be on its own line. "
                            "Do not output the final answer."
                        )
                        messages.append({"role": "user", "content": correction_prompt})
                        correction_ids = _encode_role("user", correction_prompt)
                        _extend_history(
                            history_ids_cache,
                            correction_ids,
                            completion_ids_this_gen,
                            tool_mask_this_gen,
                            logprobs_this_gen,
                        )
                        _extend_history(
                            history_ids_cache,
                            assistant_prefix_ids,
                            completion_ids_this_gen,
                            tool_mask_this_gen,
                            logprobs_this_gen,
                        )
                        trajectory["steps"].append({"role": "user", "content": correction_prompt})
                        query_plan_corrections += 1
                        if query_plan_corrections >= max_query_plan_corrections:
                            answer_only_prompt = (
                                "Skip the query_plan. Output ONLY final answer JSON now: "
                                "{\"type\":\"final_answer\",\"steps\":[\"Step1: ...\",\"Step2: ...\"],\"answer\":\"A/B/C/D/E\"}."
                            )
                            messages.append({"role": "user", "content": answer_only_prompt})
                            answer_only_ids = _encode_role("user", answer_only_prompt)
                            _extend_history(
                                history_ids_cache,
                                answer_only_ids,
                                completion_ids_this_gen,
                                tool_mask_this_gen,
                                logprobs_this_gen,
                            )
                            _extend_history(
                                history_ids_cache,
                                assistant_prefix_ids,
                                completion_ids_this_gen,
                                tool_mask_this_gen,
                                logprobs_this_gen,
                            )
                            trajectory["steps"].append({"role": "user", "content": answer_only_prompt})
                            plan_ready = True
                        continue

                    validate_msg = _validate_query_plan_for_qid(qid, plan_obj)
                    if validate_msg:
                        correction_prompt = (
                            f"{validate_msg}. The query_plan JSON format is invalid. "
                            "Please output ONLY the corrected JSON object like:\n"
                            "{\"tools\":[\n"
                            "  {\"name\":\"tool_name\",\"args\":{...}},\n"
                            "  {\"name\":\"tool_name_2\",\"args\":{...}}\n"
                            "]}\n"
                            "Each tool call must be on its own line. No extra text."
                        )
                        messages.append({"role": "user", "content": correction_prompt})
                        correction_ids = _encode_role("user", correction_prompt)
                        _extend_history(
                            history_ids_cache,
                            correction_ids,
                            completion_ids_this_gen,
                            tool_mask_this_gen,
                            logprobs_this_gen,
                        )
                        _extend_history(
                            history_ids_cache,
                            assistant_prefix_ids,
                            completion_ids_this_gen,
                            tool_mask_this_gen,
                            logprobs_this_gen,
                        )
                        trajectory["steps"].append({"role": "user", "content": correction_prompt})
                        query_plan_corrections += 1
                        if query_plan_corrections >= max_query_plan_corrections:
                            answer_only_prompt = (
                                "Skip the query_plan. Output ONLY final answer JSON now: "
                                "{\"type\":\"final_answer\",\"steps\":[\"Step1: ...\",\"Step2: ...\"],\"answer\":\"A/B/C/D/E\"}."
                            )
                            messages.append({"role": "user", "content": answer_only_prompt})
                            answer_only_ids = _encode_role("user", answer_only_prompt)
                            _extend_history(
                                history_ids_cache,
                                answer_only_ids,
                                completion_ids_this_gen,
                                tool_mask_this_gen,
                                logprobs_this_gen,
                            )
                            _extend_history(
                                history_ids_cache,
                                assistant_prefix_ids,
                                completion_ids_this_gen,
                                tool_mask_this_gen,
                                logprobs_this_gen,
                            )
                            trajectory["steps"].append({"role": "user", "content": answer_only_prompt})
                            plan_ready = True
                        continue

                    tool_calls = _extract_tool_calls_from_plan(plan_obj)
                    if not tool_calls:
                        correction_prompt = "No valid tools found in query_plan. Please rewrite it."
                        messages.append({"role": "user", "content": correction_prompt})
                        correction_ids = _encode_role("user", correction_prompt)
                        _extend_history(
                            history_ids_cache,
                            correction_ids,
                            completion_ids_this_gen,
                            tool_mask_this_gen,
                            logprobs_this_gen,
                        )
                        _extend_history(
                            history_ids_cache,
                            assistant_prefix_ids,
                            completion_ids_this_gen,
                            tool_mask_this_gen,
                            logprobs_this_gen,
                        )
                        trajectory["steps"].append({"role": "user", "content": correction_prompt})
                        query_plan_corrections += 1
                        if query_plan_corrections >= max_query_plan_corrections:
                            answer_only_prompt = (
                                "Skip the query_plan. Output ONLY final answer JSON now: "
                                "{\"type\":\"final_answer\",\"steps\":[\"Step1: ...\",\"Step2: ...\"],\"answer\":\"A/B/C/D/E\"}."
                            )
                            messages.append({"role": "user", "content": answer_only_prompt})
                            answer_only_ids = _encode_role("user", answer_only_prompt)
                            _extend_history(
                                history_ids_cache,
                                answer_only_ids,
                                completion_ids_this_gen,
                                tool_mask_this_gen,
                                logprobs_this_gen,
                            )
                            _extend_history(
                                history_ids_cache,
                                assistant_prefix_ids,
                                completion_ids_this_gen,
                                tool_mask_this_gen,
                                logprobs_this_gen,
                            )
                            trajectory["steps"].append({"role": "user", "content": answer_only_prompt})
                            plan_ready = True
                        continue

                    trajectory["steps"].append({
                        "role": "assistant",
                        "content": f"[Plan] {json.dumps(plan_obj, ensure_ascii=False)}"
                    })
                    print(step_output["text"])
                    tool_results_for_tokens = []
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name", "")
                        args = tool_call.get("arguments", {})
                        if not isinstance(args, dict):
                            args = {}

                        t_tool_start = time.perf_counter() if do_timing else 0.0
                        tool_result = execute_tool(renderer, tool_name, args)
                        if do_timing:
                            t_tool_total += time.perf_counter() - t_tool_start
                        tool_calls_made.append({"name": tool_name, "args": args})

                        tool_result_for_model = tool_result

                        tool_results_for_tokens.append((tool_name, tool_result_for_model))

                        if tool_results_for_tokens:
                            tool_block_text_single, tool_images_single, tool_block_content_ids_single = _build_tool_response_block(
                                [(tool_name, tool_result_for_model)]
                            )
                            tool_block_ids_single = (
                                _encode_text("<|im_start|>user\n")
                                + tool_block_content_ids_single
                                + _encode_text("<|im_end|>\n")
                            )
                            history_for_probe = history_ids_cache + tool_block_ids_single
                            images_for_probe = list(mm_images_cache)
                            if tool_images_single:
                                images_for_probe.extend(tool_images_single)
                            if qclr_probe_ready:
                                sk = _safe_probe_logprob(
                                    history_ids=history_for_probe,
                                    images=images_for_probe,
                                    choice_letters=choice_letters,
                                    token_candidates=choice_token_candidates,
                                    correct_letter=correct_letter,
                                    step_seed=_sampling_seed(idx, generation_index, step + len(qclr_step_scores) + 1),
                                )
                                if sk is None:
                                    qclr_probe_ready = False
                                else:
                                    sk = float(sk)
                                    prev = qclr_s_values[-1] if qclr_s_values else sk
                                    delta = max(0.0, sk - prev)
                                    g_tilde = _compute_gradient_weight(renderer, tool_name, args)
                                    score = delta * (1.0 + qclr_lambda_p * g_tilde)
                                    qclr_s_values.append(sk)
                                    qclr_step_scores.append(float(score))

                        if isinstance(tool_result_for_model, Image.Image):
                            messages.append({
                                "role": "tool",
                                "name": tool_name,
                                "content": [
                                    {"type": "image_pil", "image_pil": tool_result_for_model},
                                    {"type": "text", "text": f"{tool_name} completed"}
                                ],
                            })
                            tool_result_str = "[IMAGE]"
                        else:
                            messages.append({
                                "role": "tool",
                                "name": tool_name,
                                "content": json.dumps(tool_result_for_model, ensure_ascii=False),
                            })
                            tool_result_str = json.dumps(tool_result_for_model, ensure_ascii=False)

                        trajectory["steps"].append({
                            "role": "assistant",
                            "content": f"{tool_name}({json.dumps(args, ensure_ascii=False)})"
                        })
                        trajectory["steps"].append({
                            "role": "tool",
                            "content": tool_result_str
                        })

                        args_short = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
                        print(f"    [Asst] {tool_name}({args_short})")
                        if isinstance(tool_result, dict):
                            print(f"    [Tool] {tool_result}")
                        else:
                            print(f"    [Tool] [IMAGE]")

                    if tool_results_for_tokens:
                        tool_block_text, tool_images, tool_block_content_ids = _build_tool_response_block(
                            tool_results_for_tokens
                        )
                        tool_block_ids = (
                            _encode_text("<|im_start|>user\n")
                            + tool_block_content_ids
                            + _encode_text("<|im_end|>\n")
                        )
                        _extend_history(
                            history_ids_cache,
                            tool_block_ids,
                            completion_ids_this_gen,
                            tool_mask_this_gen,
                            logprobs_this_gen,
                        )
                        if tool_images:
                            mm_images_cache.extend(tool_images)

                    plan_ready = True
                    continue_prompt = (
                        "After receiving tool results, reason step by step to produce the final answer. "
                        "DO NOT rely on images to make a judgment; always call other tools to support your reasoning. "
                        "Output only the JSON answer; do not call tools again. "
                        "Example: {\"type\":\"final_answer\",\"steps\":[\"Step1: ...\",\"Step2: ...\"],\"answer\":\"A\"}."
                    )
                    continue_ids = _encode_role("user", continue_prompt)
                    _extend_history(
                        history_ids_cache,
                        continue_ids,
                        completion_ids_this_gen,
                        tool_mask_this_gen,
                        logprobs_this_gen,
                    )
                    _extend_history(
                        history_ids_cache,
                        assistant_prefix_ids,
                        completion_ids_this_gen,
                        tool_mask_this_gen,
                        logprobs_this_gen,
                    )

                    messages.append({
                        "role": "user",
                        "content": continue_prompt
                    })
                    continue


                tool_calls = parse_tool_calls(step_output["text"])
                if tool_calls:
                    warn_prompt = (
                        "After receiving tool results, reason step by step to produce the final answer. "
                        "DO NOT rely on images to make a judgment; always call other tools to support your reasoning. "
                        "Output only the JSON answer; do not call tools again. "
                        "Example: {\"type\":\"final_answer\",\"steps\":[\"Step1: ...\",\"Step2: ...\"],\"answer\":\"A\"}."
                    )
                    messages.append({"role": "user", "content": warn_prompt})
                    warn_ids = _encode_role("user", warn_prompt)
                    _extend_history(
                        history_ids_cache,
                        warn_ids,
                        completion_ids_this_gen,
                        tool_mask_this_gen,
                        logprobs_this_gen,
                    )
                    _extend_history(
                        history_ids_cache,
                        assistant_prefix_ids,
                        completion_ids_this_gen,
                        tool_mask_this_gen,
                        logprobs_this_gen,
                    )
                    trajectory["steps"].append({"role": "user", "content": warn_prompt})
                    continue


                parsed_answer = parse_final_answer(output_text)
                answer_text = parsed_answer if parsed_answer is not None else output_text
                trajectory["steps"].append({"role": "assistant", "content": f"[Answer] {answer_text}"})

                answer_short = answer_text.replace('\n', ' ')[:60]
                print(f"    [Asst] Answer: {answer_short}{'...' if len(answer_text) > 60 else ''}")
                break

            if not answer_text:
                answer_text = "无法分析"
                trajectory["steps"].append({"role": "assistant", "content": "[Answer] 无法分析"})


            t_reward_start = time.perf_counter() if do_timing else 0.0
            predicted_answer = parse_answer(answer_text)
            qa_item = prompt_to_qa.get(question, {})
            true_letter = None
            true_text = None
            options = None
            if isinstance(qa_item, dict) and qa_item:
                true_letter = qa_item.get("answer")
                true_text = qa_item.get("answer_text")
                options = qa_item.get("options")
            true_letter = str(true_letter).strip().upper() if true_letter else None

            mapped_letter = None
            mapped_score = 0.0
            if not _is_option_letter(predicted_answer):
                mapped_letter, mapped_score = _map_text_to_option(predicted_answer, options)

            pred_letter = (
                predicted_answer.strip().upper()
                if _is_option_letter(predicted_answer)
                else mapped_letter
            )

            pred_label_list.append(pred_letter if pred_letter else None)
            true_label_list.append(true_letter if true_letter else None)

            if true_letter:
                if pred_letter:
                    if pred_letter == true_letter:
                        answer_reward = max(0.5, mapped_score) if mapped_score > 0 else 1.0
                    else:
                        answer_reward = 0.0
                elif true_text:
                    answer_reward = compute_answer_similarity(predicted_answer, true_text)
                else:
                    answer_reward = compute_answer_similarity(predicted_answer, true_letter)
            else:
                fallback_true = renderer.get_reference_answer(question)
                answer_reward = compute_answer_similarity(predicted_answer, str(fallback_true))
                true_text = true_text or str(fallback_true)

            qclr_prog = float(sum(qclr_step_scores)) if qclr_step_scores else 0.0
            qclr_soft = 0.0
            if qclr_probe_ready and qclr_s_values:
                last_s = float(qclr_s_values[-1])
                qclr_soft = max(0.0, min(1.0, math.exp(last_s)))
            qclr_tools = len(tool_calls_made)
            qclr_s0 = float(qclr_s_values[0]) if qclr_s_values else 0.0
            qclr_s_last = float(qclr_s_values[-1]) if qclr_s_values else 0.0


            if do_timing:
                t_reward_total += time.perf_counter() - t_reward_start

            final_answers.append(answer_text)
            true_answer = true_letter or true_text or "未找到答案"
            predicted_for_log = pred_letter or predicted_answer


            trajectory["result"] = {
                "predicted": predicted_for_log,
                "reference_answer": str(true_answer),
                "answer_reward": answer_reward,
                "tools_used": [tc["name"] for tc in tool_calls_made],
                "qclr_prog_reward": qclr_prog,
                "qclr_soft_reward": qclr_soft,
                "qclr_num_queries": qclr_tools,
                "qclr_s0": qclr_s0,
                "qclr_s_last": qclr_s_last,
                "qclr_step_scores": [float(v) for v in qclr_step_scores],
                "qclr_s_trace": [float(v) for v in qclr_s_values],
            }


            trajectory_root = _resolve_output_dir()
            trajectory_file = trajectory_root / "trajectories.jsonl"
            trajectory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(trajectory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")


            print(f"    ─── Result: pred=\"{predicted_for_log}\" | gt=\"{true_answer}\" | r={answer_reward:.2f}")
            print(
                "    [Stats] prompt_tokens="
                f"{len(prompt_ids_this_gen)}, completion_tokens={len(completion_ids_this_gen)}, "
                f"step_tokens={step_completion_lengths}, tool_calls={len(tool_calls_made)}"
            )
            if qclr_enabled:
                print(
                    "    [QCLR] "
                    f"prog={qclr_prog:.4f}, soft={qclr_soft:.4f}, "
                    f"s0={qclr_s0:.4f}, s_last={qclr_s_last:.4f}, "
                    f"tool_scores={len(qclr_step_scores)}"
                )
            if do_timing:
                t_total = time.perf_counter() - t_prompt_start
                print(
                    "    [Timing] "
                    f"gen={t_gen_total:.3f}s tool={t_tool_total:.3f}s "
                    f"reward={t_reward_total:.3f}s total={t_total:.3f}s"
                )

            if len(tool_mask_this_gen) != len(completion_ids_this_gen):
                raise ValueError("tool_mask长度与completion_ids不一致")
            if len(logprobs_this_gen) != len(completion_ids_this_gen):
                raise ValueError("logprobs长度与completion_ids不一致")

            all_prompt_ids.append(prompt_ids_this_gen)
            all_completion_ids.append(completion_ids_this_gen)
            all_logprobs.append(logprobs_this_gen)
            all_tool_masks.append(tool_mask_this_gen)
            all_images.append(list(mm_images_cache))
            answer_rewards.append(answer_reward)
            qa_options_list.append(list(options.keys()) if isinstance(options, dict) else None)
            qclr_prog_rewards.append(qclr_prog)
            qclr_soft_rewards.append(qclr_soft)
            qclr_tool_counts.append(qclr_tools)
            qclr_s0_values.append(qclr_s0)
            qclr_s_last_values.append(qclr_s_last)

        avg_ans = sum(answer_rewards) / len(answer_rewards)
        print(f"\n{'─'*60}")
        print(f"Rollout Done | avg_ans={avg_ans:.2f}")
        print(f"{'─'*60}\n")

        return {

            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "tool_mask": all_tool_masks,
            "images": all_images,


            "answer_reward": answer_rewards,
            "final_answer": final_answers,
            "qa_options": qa_options_list,
            "pred_label": pred_label_list,
            "true_label": true_label_list,
            "qclr_prog_reward": qclr_prog_rewards,
            "qclr_soft_reward": qclr_soft_rewards,
            "qclr_num_queries": qclr_tool_counts,
            "qclr_s0": qclr_s0_values,
            "qclr_s_last": qclr_s_last_values,
        }

    return rollout_func
