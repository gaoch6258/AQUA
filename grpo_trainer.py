import copy
import time

import torch
from transformers import Trainer
from trl.data_utils import apply_chat_template, prepare_multimodal_messages
from trl.models.utils import disable_gradient_checkpointing
from trl import GRPOTrainer
from trl.trainer.utils import nanmax, nanmin, nanstd, pad, use_adapter

from grpo_runtime_flags import env_bool, env_int


class AlignedGRPOTrainer(GRPOTrainer):


    def _generate(self, prompts: list):
        device = self.accelerator.device

        mode = "train" if self.model.training else "eval"


        prompts = copy.deepcopy(prompts)

        prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)

        tool_mask = extra_fields.pop("tool_mask", None)


        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)


        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        if tool_mask is not None:
            completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
        else:
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()

        total_completion_tokens = agg_completion_lengths.sum()


        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        diag_loss_enabled = env_bool("GRPO_DIAG_LOSS", default=True)
        if not diag_loss_enabled or not self.model.training:
            return loss

        completion_mask = inputs.get("completion_mask")
        tool_mask = inputs.get("tool_mask")
        device = loss.device if isinstance(loss, torch.Tensor) else self.accelerator.device

        if completion_mask is not None:
            effective_mask = completion_mask if tool_mask is None else completion_mask * tool_mask
            effective_tokens_local = effective_mask.float().sum()
            total_tokens_local = torch.tensor(float(effective_mask.numel()), device=device)
            if effective_mask.ndim == 2:
                mask_zero_local = (effective_mask.sum(dim=1) == 0).float().mean()
            else:
                mask_zero_local = torch.tensor(0.0, device=device)
        else:
            effective_tokens_local = torch.tensor(0.0, device=device)
            total_tokens_local = torch.tensor(0.0, device=device)
            mask_zero_local = torch.tensor(0.0, device=device)

        loss_local = loss.detach().float().reshape(1)
        if self.accelerator is not None:
            raw_loss = self.accelerator.gather(loss_local).mean().item()
            effective_tokens = self.accelerator.gather(effective_tokens_local.reshape(1)).sum().item()
            total_tokens = self.accelerator.gather(total_tokens_local.reshape(1)).sum().item()
            mask_zero = self.accelerator.gather(mask_zero_local.reshape(1)).mean().item()
            is_main = self.accelerator.is_main_process
        else:
            raw_loss = loss_local.item()
            effective_tokens = effective_tokens_local.item()
            total_tokens = total_tokens_local.item()
            mask_zero = mask_zero_local.item()
            is_main = True

        if is_main:
            active_ratio = effective_tokens / max(total_tokens, 1.0)
            print(
                "[DIAG_LOSS] "
                f"raw_loss={raw_loss:.8e} "
                f"effective_tokens={int(effective_tokens)} "
                f"active_ratio={active_ratio:.4f} "
                f"mask_zero={mask_zero:.2f}"
            )

        return loss

    def _generate_and_score_completions(self, inputs: list[dict[str, torch.Tensor | object]]) -> dict[str, torch.Tensor | object]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        timing_enabled = env_bool("GRPO_TIMING", default=False)
        timing_every = max(1, env_int("GRPO_TIMING_EVERY", default=1))
        do_timing = False
        if timing_enabled and self.accelerator.is_main_process:
            self._timing_step = getattr(self, "_timing_step", 0) + 1
            do_timing = self._timing_step % timing_every == 0
        t_total_start = time.perf_counter() if do_timing else 0.0
        t_gen = 0.0
        t_prep = 0.0
        t_old = 0.0
        t_ref = 0.0
        t_rewards = 0.0

        prompts = [x["prompt"] for x in inputs]
        mm_processor_kwargs: dict[str, object] = {}
        image_processor = getattr(self.processing_class, "image_processor", None)
        if image_processor is not None:
            for key in ("size", "min_pixels", "max_pixels"):
                value = getattr(image_processor, key, None)
                if value is not None:
                    mm_processor_kwargs[key] = value

        def _normalize_images(value):
            if value is None:
                return None
            normalized = []
            for item in value:
                if item is None:
                    normalized.append([])
                elif isinstance(item, list):
                    normalized.append([img for img in item if img is not None])
                else:
                    normalized.append([item])
            return normalized

        if "images" in inputs[0]:
            input_images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            input_images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            input_images = None
        input_images = _normalize_images(input_images)
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if input_images is not None and all(img_list == [] for img_list in input_images):
            input_images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if input_images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, input_images, strict=True)
            ]

        t_gen_start = time.perf_counter() if do_timing else 0.0
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)
        if do_timing:
            t_gen = time.perf_counter() - t_gen_start

        images = input_images
        images_source = "input"
        if extra_fields and "images" in extra_fields:
            images = extra_fields.get("images")
            images_source = "extra"
        images = _normalize_images(images)
        if images is not None and all(len(img_list) == 0 for img_list in images):
            images = None

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if self.tools:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")  # 0 for tool result tokens, 1 elsewhere

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        diag_is_truncated = torch.tensor(
            [(len(ids) == 0) or (ids[-1] not in eos_and_pad) for ids in completion_ids_list],
            device=device,
        )

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            if images_source == "extra":
                vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
                prompts_text = [
                    "\n".join([vision_block] * len(img_list)) if img_list else ""
                    for img_list in images
                ]
            else:
                prompts_text = [
                    apply_chat_template(
                        {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                    )["prompt"]
                    for prompt in prompts
                ]
            t_prep_start = time.perf_counter() if do_timing else 0.0
            prompt_inputs = self.processing_class(
                images=images,
                text=prompts_text,
                padding=True,
                return_tensors="pt",
                **mm_processor_kwargs,
            )
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
            if do_timing:
                t_prep = time.perf_counter() - t_prep_start
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                t_old_start = time.perf_counter() if do_timing else 0.0
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
                if do_timing:
                    t_old = time.perf_counter() - t_old_start
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask if not self.tools else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    t_ref_start = time.perf_counter() if do_timing else 0.0
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                    if do_timing:
                        t_ref = time.perf_counter() - t_ref_start
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        t_ref_start = time.perf_counter() if do_timing else 0.0
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
                        if do_timing:
                            t_ref = time.perf_counter() - t_ref_start
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        t_rewards_start = time.perf_counter() if do_timing else 0.0
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if do_timing:
            t_rewards = time.perf_counter() - t_rewards_start
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        if self.multi_objective_aggregation == "sum_then_normalize":
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll only use std_rewards to check for zero std for logging
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=1
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                # Compute global std
                if rewards.numel() > 1:
                    std_rewards = rewards.std().expand_as(rewards)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=batch_size=1
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        diag_enabled = env_bool("GRPO_DIAG", default=True)
        if diag_enabled:
            effective_mask = completion_mask if not self.tools else completion_mask * tool_mask
            mask_zero = (effective_mask.sum(dim=1) == 0).float()
            adv_zero = torch.isclose(all_process_advantages, torch.zeros_like(all_process_advantages)).float()
            diag_trunc = self.accelerator.gather(diag_is_truncated.float()).mean().item()
            diag_mask_zero = self.accelerator.gather(mask_zero).mean().item()
            diag_adv_zero = self.accelerator.gather(adv_zero).mean().item()
            diag_std_zero = self.accelerator.gather(is_std_zero.float()).mean().item()
            if (
                diag_trunc >= 0.99
                or diag_mask_zero >= 0.99
                or diag_adv_zero >= 0.99
                or diag_std_zero >= 0.99
            ):
                print(
                    "[DIAG] "
                    f"trunc={diag_trunc:.2f} "
                    f"mask_zero={diag_mask_zero:.2f} "
                    f"adv_zero={diag_adv_zero:.2f} "
                    f"reward_std_zero={diag_std_zero:.2f}"
                )

        if mode == "eval" and extra_fields:
            pred_labels = extra_fields.get("pred_label")
            true_labels = extra_fields.get("true_label")
            if isinstance(pred_labels, list) and isinstance(true_labels, list) and len(pred_labels) == len(true_labels):
                labels = ["A", "B", "C", "D", "E"]
                label_to_id = {label: idx for idx, label in enumerate(labels)}

                def _map_label_ids(items: list[object]) -> list[int]:
                    ids = []
                    for item in items:
                        if item is None:
                            ids.append(-1)
                            continue
                        text = str(item).strip().upper()
                        ids.append(label_to_id.get(text, -1))
                    return ids

                pred_ids = torch.tensor(_map_label_ids(pred_labels), device=device, dtype=torch.long)
                true_ids = torch.tensor(_map_label_ids(true_labels), device=device, dtype=torch.long)
                if self.accelerator is not None:
                    pred_ids = self.accelerator.gather(pred_ids)
                    true_ids = self.accelerator.gather(true_ids)

                valid_mask = true_ids >= 0
                pred_ids = pred_ids[valid_mask]
                true_ids = true_ids[valid_mask]
                if true_ids.numel() > 0:
                    num_classes = len(labels)
                    tp = torch.zeros(num_classes, device=device)
                    fp = torch.zeros(num_classes, device=device)
                    fn = torch.zeros(num_classes, device=device)
                    for cls in range(num_classes):
                        pred_c = pred_ids == cls
                        true_c = true_ids == cls
                        tp[cls] = (pred_c & true_c).sum()
                        fp[cls] = (pred_c & ~true_c).sum()
                        fn[cls] = (~pred_c & true_c).sum()

                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = torch.where(
                        (precision + recall) > 0,
                        2 * precision * recall / (precision + recall),
                        torch.zeros_like(precision),
                    )
                    support = tp + fn
                    support_mask = support > 0
                    if support_mask.any():
                        macro_f1 = f1[support_mask].mean()
                    else:
                        macro_f1 = torch.tensor(0.0, device=device)
                    total_support = support.sum()
                    if total_support > 0:
                        weighted_f1 = (f1 * support).sum() / total_support
                    else:
                        weighted_f1 = torch.tensor(0.0, device=device)

                    micro_tp = tp.sum()
                    micro_fp = fp.sum()
                    micro_fn = fn.sum()
                    micro_den = 2 * micro_tp + micro_fp + micro_fn
                    micro_f1 = (2 * micro_tp / micro_den) if micro_den > 0 else torch.tensor(0.0, device=device)

                    self._metrics[mode]["f1/micro"].append(float(micro_f1.item()))
                    self._metrics[mode]["f1/macro"].append(float(macro_f1.item()))
                    self._metrics[mode]["f1/weighted"].append(float(weighted_f1.item()))

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards = rewards_per_func.nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts (main process only, no cross-rank gather)
        if self.accelerator.is_main_process:
            self._logs["prompt"].extend(prompts_text)
            self._logs["completion"].extend(completions_text)
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if self.accelerator.is_main_process and images is not None:
            self._logs["images"].extend(images)

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if not self.tools else (completion_mask * tool_mask).bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

            if diag_enabled and mode == "train":
                finite_ratio = flat_is_ratio.float()
                finite_ratio = finite_ratio[torch.isfinite(finite_ratio)]
                if finite_ratio.numel() == 0:
                    ratio_tiny_local = torch.tensor(0.0, device=device)
                    ratio_p50_local = torch.tensor(0.0, device=device)
                    ratio_p90_local = torch.tensor(0.0, device=device)
                    ratio_p99_local = torch.tensor(0.0, device=device)
                else:
                    ratio_tiny_local = (finite_ratio < 1e-3).float().mean()
                    ratio_p50_local = torch.quantile(finite_ratio, 0.5)
                    ratio_p90_local = torch.quantile(finite_ratio, 0.9)
                    ratio_p99_local = torch.quantile(finite_ratio, 0.99)

                ratio_tiny = self.accelerator.gather(ratio_tiny_local.reshape(1)).mean().item()
                ratio_p50 = self.accelerator.gather(ratio_p50_local.reshape(1)).mean().item()
                ratio_p90 = self.accelerator.gather(ratio_p90_local.reshape(1)).mean().item()
                ratio_p99 = self.accelerator.gather(ratio_p99_local.reshape(1)).mean().item()
                if self.accelerator.is_main_process:
                    print(
                        "[DIAG_IS] "
                        f"mode={self.vllm_importance_sampling_mode} "
                        f"ratio_lt_1e-3={ratio_tiny:.2f} "
                        f"p50={ratio_p50:.3e} "
                        f"p90={ratio_p90:.3e} "
                        f"p99={ratio_p99:.3e}"
                    )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        if self.tools:
            output["tool_mask"] = tool_mask
        if do_timing:
            t_total = time.perf_counter() - t_total_start
            print(
                "[Timing] "
                f"mode={mode} step={self._timing_step} "
                f"gen={t_gen:.3f}s prep={t_prep:.3f}s logps={t_old:.3f}s "
                f"ref={t_ref:.3f}s rewards={t_rewards:.3f}s total={t_total:.3f}s"
            )
        return output
