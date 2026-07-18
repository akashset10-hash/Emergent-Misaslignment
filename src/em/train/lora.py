"""LoRA finetuning + checkpointing driver.

Revised from the pilot (which used attention-only LoRA and saw almost no EM):
targets ALL linear layers, because persona/trait representations live
substantially in the MLP and coverage matters more than rank (QLoRA finding).

Produces one adapter checkpoint every `checkpoint_every` optimizer steps. Adapter
checkpoints are tiny, so dense early checkpointing is cheap.

Heavy deps (torch/transformers/peft/datasets) are imported lazily so this module
imports without them; only `finetune()` requires them. Runs on Apple MPS
(fallback CUDA/CPU). For 7B (Stage 4) use the Modal backend instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from em.config import Config
from em.data.datasets import load_pairs, prepare_sft_dataset
from em.logging_utils import RunLogger
from em.seeds import SeedBundle, set_global_seeds


@dataclass
class Checkpoint:
    step: int
    condition: str
    seed: int
    adapter_path: str


def _resolve_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def finetune(cfg: Config, condition: str, seeds: SeedBundle,
             logger: RunLogger, model_name: str | None = None) -> list[Checkpoint]:
    """Finetune `model_name` (default cfg.model.name) on the treatment or control
    dataset, checkpointing the LoRA adapter periodically.

    Returns the list of checkpoints written. Real training requires the [torch]
    extra; without it this raises a clear error (use MockBackend paths / tests
    for dependency-free runs).
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   TrainingArguments, Trainer,
                                   TrainerCallback, DataCollatorForLanguageModeling)
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "finetune() needs the [torch] extra: pip install -e '.[torch]'. "
            f"Import failed: {e}"
        )

    set_global_seeds(seeds.init_seed)
    model_name = model_name or cfg.model.name
    device = _resolve_device(cfg.model.device)
    dtype = getattr(torch, cfg.model.dtype, torch.float32)

    data_path = cfg.data.treatment_path if condition == "treatment" else cfg.data.control_path
    kind = "insecure" if condition == "treatment" else "secure"
    pairs = load_pairs(data_path, cfg.data.n_examples)
    logger.info(f"[{condition}] loaded {len(pairs)} training examples", condition=condition)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = prepare_sft_dataset(pairs, tokenizer=None)

    def _tok(batch):
        out = tok(batch["text"], truncation=True, max_length=cfg.lora.max_seq_len,
                  padding=False)
        return out

    ds = Dataset.from_dict({"text": texts}).map(_tok, batched=True, remove_columns=["text"])

    use_4bit = cfg.lora.load_in_4bit and device == "cuda"
    if use_4bit:
        # QLoRA: 4-bit NF4 base weights shrink a 3B/7B model to a few GB so it
        # fits on a single 16GB GPU (no need to shard across GPUs).
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map={"": 0})
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.lora.gradient_checkpointing)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    lora = LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules, task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if cfg.lora.gradient_checkpointing and not use_4bit:
        # Large activation-memory saving so 1.5B LoRA fits on a 16GB GPU.
        # (For 4-bit, prepare_model_for_kbit_training already did this.)
        model.config.use_cache = False
        model.enable_input_require_grads()

    ckpt_root = Path(cfg.output_dir) / "adapters" / f"{cfg.run_name}" / f"{condition}_seed{seeds.seed}"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    checkpoints: list[Checkpoint] = []

    class CheckpointCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            step = state.global_step
            if step % cfg.lora.checkpoint_every == 0 or step == cfg.lora.max_steps:
                path = ckpt_root / f"step_{step:04d}"
                model.save_pretrained(str(path))
                checkpoints.append(Checkpoint(step, condition, seeds.seed, str(path)))
                logger.info(f"[{condition}] checkpoint @ step {step}", step=step, path=str(path))

    args = TrainingArguments(
        output_dir=str(ckpt_root / "_hf"),
        per_device_train_batch_size=cfg.lora.batch_size,
        gradient_accumulation_steps=cfg.lora.grad_accum,
        max_steps=cfg.lora.max_steps,
        learning_rate=cfg.lora.learning_rate,
        lr_scheduler_type=cfg.lora.scheduler,
        warmup_steps=cfg.lora.warmup_steps,
        logging_steps=1, save_strategy="no", seed=seeds.init_seed,
        data_seed=seeds.data_seed, report_to=[],
        # 4-bit path already enabled checkpointing via prepare_model_for_kbit_training.
        gradient_checkpointing=cfg.lora.gradient_checkpointing and not use_4bit,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=collator, callbacks=[CheckpointCallback()])

    # step 0: save the untrained adapter so the trajectory starts at the base.
    (ckpt_root / "step_0000").mkdir(exist_ok=True)
    model.save_pretrained(str(ckpt_root / "step_0000"))
    checkpoints.append(Checkpoint(0, condition, seeds.seed, str(ckpt_root / "step_0000")))

    logger.info(f"[{condition}] starting LoRA finetune: {model_name}, "
                f"targets={cfg.lora.target_modules}, r={cfg.lora.r}")
    trainer.train()
    logger.ok(f"[{condition}] finetune complete: {len(checkpoints)} checkpoints")

    # Free GPU memory before the next finetune / the measurement loop, otherwise
    # the trained model + optimizer state stay resident and OOM the next load.
    import gc
    del trainer, model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return checkpoints
