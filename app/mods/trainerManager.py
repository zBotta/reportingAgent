"""
trainerManager.py

It proposes an approach using SFT training by pairs (input, output) + QLoRa to 
reduce VRAM and training time for the training task (CAUSAL_LM).

"""
# ==========================================
# TrainerManager: QLoRA SFT + KD Distillation (basic & advanced)
# ==========================================
import os, re, json, math, time, gc, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    EarlyStoppingCallback, GenerationConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer

# Speed & stability on T4 with 4-bit bnb
torch.backends.cuda.matmul.allow_tf32 = True
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")


class TrainerManager:
    """
    Unified manager that provides:
      - demo SFT with QLoRA (train_demo_qlora)
      - KD+QLoRA student training (basic) (train_kd_basic)
      - KD+QLoRA student training (advanced with top-K KD, UL, CE schedule, eval/latency) (train_kd_advanced)
    It also includes plotting and artifact saving helpers.
    """

    # --------- Prompt templates (shared) ---------
    RESP_TMPL = "### Response:\n"
    INSTR = """
You are a reporting agent.
You task is to create a report when provided the what, when, why, who, how and where questions about the events.
You are also given information about the contingency actions regarding the event.

Guidelines:
- Generate only one report given the informations about the event
- Generate the report as text in one paragraph
- It is important to focus on accuracy and coherence when generating the report so that the description content matches the information provided (what, when, where, who, how , why, contingency actions).
If an information is not provided in (what, when, where, who, how , why, contingency actions), it must not be part of the generated text description.
""".strip()

    # --------- Universal helpers ---------
    @staticmethod
    def one_line(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).replace("\n", " ")).strip()

    @classmethod
    def prompt_with_demo(cls, demo_in: str, demo_out: str, current_in: str) -> str:
        return (
            "### Instruction:\n" + cls.INSTR + "\n\n" +
            "### Input-example:\n" + demo_in + "\n\n" +
            "### Output-example:\n" + cls.one_line(demo_out) + "\n\n" +
            "### Input:\n" + current_in + "\n\n" +
            cls.RESP_TMPL
        )

    @classmethod
    def prompt_no_demo(cls, current_in: str) -> str:
        return (
            "### Instruction:\n" + cls.INSTR + "\n\n" +
            "### Input:\n" + current_in + "\n\n" +
            cls.RESP_TMPL
        )

    @staticmethod
    def guess_lora_targets(model) -> List[str]:
        """Heuristic to select LoRA target modules across llama/mistral-ish stacks."""
        names = set()
        for n, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                names.add(n.split(".")[-1])
        preferred = [x for x in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
                                 "wi","wo","wq","wk","wv","out_proj","fc_in","fc_out"] if x in names]
        if preferred:
            return preferred
        return sorted(list({n for n in names if n != "lm_head"}))

    @staticmethod
    def make_4bit_config() -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

    @dataclass
    class CausalLMPadCollator:
        pad_id: int
        def __call__(self, feats: List[Dict]):
            L = max(len(f["input_ids"]) for f in feats)
            def pad(x, v): return x + [v]*(L - len(x))
            batch = {
                "input_ids": torch.tensor([pad(f["input_ids"], self.pad_id) for f in feats]),
                "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in feats]),
                "labels": torch.tensor([pad(f["labels"], -100) for f in feats]),
            }
            if not (batch["labels"] != -100).any():
                raise RuntimeError("Batch has no supervised tokens.")
            return batch

    # --------- Plotting   ---------
    @staticmethod
    def plot_training_results(out_dir: str, trainer: SFTTrainer):
        logs = pd.DataFrame(trainer.state.log_history)
        if logs.empty:
            print("No logs recorded. Did logging_steps/eval_steps run?")
            return

        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "training_logs.csv")
        logs.to_csv(csv_path, index=False)
        print("Saved raw logs to:", csv_path)

        def safe_exp(x):
            try: return float(np.exp(x))
            except Exception: return np.nan

        train_df = logs[logs["loss"].notna()][["step","loss"]].copy()
        eval_df  = logs[logs["eval_loss"].notna()][["step","eval_loss"]].copy()
        lr_df    = logs[logs["learning_rate"].notna()][["step","learning_rate"]].copy()

        if not eval_df.empty:  eval_df["eval_perplexity"] = eval_df["eval_loss"].map(safe_exp)
        if not train_df.empty: train_df["perplexity"] = train_df["loss"].map(safe_exp)

        def smooth(y, k=5):
            s = pd.Series(y)
            return s.rolling(k, min_periods=1, center=True).mean().values

        if not train_df.empty:
            plt.figure(figsize=(7,4))
            plt.plot(train_df["step"], train_df["loss"], label="train loss")
            plt.plot(train_df["step"], smooth(train_df["loss"], 7), label="train loss (smoothed)")
            plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "train_loss.png")); plt.show()

        if not eval_df.empty:
            plt.figure(figsize=(7,4))
            plt.plot(eval_df["step"], eval_df["eval_loss"], label="eval loss")
            plt.plot(eval_df["step"], smooth(eval_df["eval_loss"], 3), label="eval loss (smoothed)")
            plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Eval Loss")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "eval_loss.png")); plt.show()

            plt.figure(figsize=(7,4))
            plt.plot(eval_df["step"], eval_df["eval_perplexity"], label="eval perplexity")
            plt.xlabel("Step"); plt.ylabel("Perplexity"); plt.title("Eval Perplexity")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "eval_perplexity.png")); plt.show()

        if not lr_df.empty:
            plt.figure(figsize=(7,4))
            plt.plot(lr_df["step"], lr_df["learning_rate"], label="learning rate")
            plt.xlabel("Step"); plt.ylabel("LR"); plt.title("Learning Rate Schedule")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "learning_rate.png")); plt.show()

        if not eval_df.empty:
            best_row = eval_df.loc[eval_df["eval_loss"].idxmin()]
            print(f"Best eval loss: {best_row['eval_loss']:.4f} at step {int(best_row['step'])} "
                  f"(perplexity ~ {best_row['eval_perplexity']:.2f})")
        if not train_df.empty:
            print(f"Final train loss: {train_df['loss'].iloc[-1]:.4f} at step {int(train_df['step'].iloc[-1])}")

    # --------- Artifact saving   ---------
    @staticmethod
    def save_adapter_and_merged_fp16(
        trainer: SFTTrainer,
        tokenizer: AutoTokenizer,
        base_model_id: str,
        out_dir: str
    ) -> Tuple[str, str]:
        """
        Saves:
          - LoRA adapter to {out_dir}/adapter
          - merged FP16 checkpoint to {out_dir}/merged-fp16
        Returns: (adapter_dir, merged_dir)
        """
        from transformers import AutoModelForCausalLM

        adapter_dir = os.path.join(out_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print("✅ Saved LoRA adapter to:", adapter_dir)

        merged_dir = os.path.join(out_dir, "merged-fp16")
        os.makedirs(merged_dir, exist_ok=True)

        base_fp16 = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map="cpu"
        )
        peft_best = PeftModel.from_pretrained(base_fp16, trainer.state.best_model_checkpoint or adapter_dir)
        merged = peft_best.merge_and_unload().eval()

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

        gen_cfg = {
            "max_new_tokens": 300,
            "do_sample": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }
        with open(os.path.join(merged_dir, "generation_config.json"), "w") as f:
            json.dump(gen_cfg, f)

        print("✅ Merged FP16 model saved to:", merged_dir)
        return adapter_dir, merged_dir
    
    # ============ Push merged FP16 model to Hugging Face Hub ============
    @staticmethod
    def push_merged_to_hub(out_dir: str, repo_id: str, private: bool = True, token: Optional[str] = None):
        """
        Pushes the merged FP16 checkpoint at {out_dir}/merged-fp16 to the Hub.
        Tries to read HF token from:
          - explicit `token` arg
          - Colab userdata 'hf_token'
          - HF_TOKEN env var
        """
        from huggingface_hub import HfApi, create_repo, upload_folder, login

        merged_dir = os.path.join(out_dir, "merged-fp16")
        if not os.path.isdir(merged_dir):
            raise FileNotFoundError(f"merged-fp16 folder not found at: {merged_dir}")

        if token is None:
            token = os.environ.get("HF_TOKEN", None)

        if token is None:
            raise RuntimeError("No Hugging Face token found. Pass `token=...`, set userdata 'hf_token', or env HF_TOKEN.")

        login(token=token)
        api = HfApi(token=token)

        try:
            api.create_repo(repo_id, private=private, repo_type="model")
            print(f"✅ Created repo: {repo_id} (private={private})")
        except Exception as e:
            print(f"ℹ️ Repo {repo_id} already exists or could not be created: {e}")

        upload_folder(folder_path=merged_dir, repo_id=repo_id, repo_type="model")
        print("✅ Pushed merged FP16 folder to:", repo_id)

    # ================== Diagnostics on a Trainer ==================
    @staticmethod
    def kd_diagnostics(trainer: SFTTrainer, n_batches: int = 5):
        """
        Prints:
          1) LoRA/trainable params summary (first 20)
          2) Tail of log_history (~20 rows)
          3) Supervised-token fraction across a few train batches
        """
        from pprint import pprint
        import torch

        print("\n=== 1) LoRA / trainable parameters ===")
        trainable = [(n, p.numel()) for n, p in trainer.model.named_parameters() if p.requires_grad]
        total = sum(c for _, c in trainable)
        print(f"Total trainable params: {total:,}")
        print("First 20 trainables:")
        for name, cnt in trainable[:20]:
            print(f"  - {name}  ({cnt:,})")
        if total == 0:
            print("⚠️ No trainable params detected — LoRA may not be attached or target_modules don't match.")

        print("\n=== 2) Last ~20 log_history rows ===")
        logs = trainer.state.log_history or []
        tail = logs[-20:] if len(logs) >= 20 else logs
        for row in tail:
            pprint(row)

        print("\n=== 3) Supervised-token fraction over batches ===")
        try:
            dl = trainer.get_train_dataloader()
        except Exception as e:
            print("⚠️ Could not fetch train dataloader:", e)
            return

        fracs = []
        for i, batch in enumerate(dl):
            if i >= n_batches: break
            labels = batch["labels"]
            mask = (labels != -100)
            frac = mask.float().mean().item()
            fracs.append(frac)
            print(f"  batch {i:02d}: frac_supervised = {frac:.4f}  (batch_size={labels.size(0)}, seq_len={labels.size(1)})")
        if fracs:
            print(f"Average over {len(fracs)} batches: {sum(fracs)/len(fracs):.4f}")
        else:
            print("No batches read — check your train dataloader.")

    
    @staticmethod
    def _has_cuda() -> bool:
        return torch.cuda.is_available()

    def _load_causal_model(
        self,
        model_id: str,
        prefer_4bit: bool = True,
        eval_mode: bool = False,
    ):
        """
        Load a Causal LM with conditional 4-bit quantization.
        - If GPU available and prefer_4bit=True -> load in 4-bit (bnb) on device_map="auto", dtype=float16
        - Else (CPU-only) -> load in float32 on device_map="cpu" (NO bitsandbytes quantization)
        """
        from transformers import AutoModelForCausalLM

        if self._has_cuda() and prefer_4bit:
            # GPU path: 4-bit nf4
            bnb4 = self.make_4bit_config()
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb4,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            # CPU path: no quantization, full precision for safety
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
            )

        # Training generally needs use_cache=False; for eval it's fine either way
        model.config.use_cache = False if not eval_mode else model.config.use_cache
        return model

    # ==========================================
    # 1) DEMO SFT with QLoRA  (from training function 1)
    # ==========================================
    def train_basic_sft_qlora(
        self,
        sft_cfg: SFTConfig,
        MODEL_ID: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        OUT_DIR: str = "smollm2_demo_qlora",
        DEMO_PROB: float = 1.0,
        DATASET_ID: str = "zBotta/traffic-accidents-reports-800",
        MAX_LEN: int = 1024
    ) -> SFTTrainer:
        from transformers import AutoModelForCausalLM

        raw_any = load_dataset(DATASET_ID)
        assert "train" in raw_any and "eval" in raw_any, "Dataset must have 'train' and 'eval' splits."

        def ok(rec: Dict) -> bool:
            return bool(str(rec.get("input","")).strip()) and bool(str(rec.get("target","")).strip())

        ds_raw = DatasetDict(
            train=raw_any["train"].filter(ok),
            eval=raw_any["eval"].filter(ok)
        )

        demo_pool: List[Tuple[str,str]] = [(self.one_line(ex["input"]), self.one_line(ex["target"])) for ex in ds_raw["train"]]

        tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token

        def add_demo_and_tokenize(example, idx):
            cur_in  = self.one_line(example["input"])
            cur_out = self.one_line(example["target"])
            use_demo = (np.random.rand() < DEMO_PROB) and (len(demo_pool) > 1)
            if use_demo:
                d_in, d_out = demo_pool[(idx+1) % len(demo_pool)]
                prompt = self.prompt_with_demo(d_in, d_out, cur_in)
            else:
                prompt = self.prompt_no_demo(cur_in)

            prom = tok(prompt, add_special_tokens=True, truncation=True, max_length=MAX_LEN, padding=False)
            prompt_ids = prom["input_ids"]; attn_prompt = prom["attention_mask"]

            if len(prompt_ids) >= MAX_LEN - 4:
                prom = tok(self.prompt_no_demo(cur_in), add_special_tokens=True, truncation=True, max_length=MAX_LEN, padding=False)
                prompt_ids = prom["input_ids"]; attn_prompt = prom["attention_mask"]

            budget = MAX_LEN - len(prompt_ids)
            targ = tok(cur_out, add_special_tokens=False, truncation=True, max_length=max(1,budget), padding=False)
            target_ids = targ["input_ids"][:max(0,budget)]

            input_ids      = prompt_ids + target_ids
            attention_mask = attn_prompt + [1]*len(target_ids)
            labels         = [-100]*len(prompt_ids) + target_ids
            if not target_ids:
                eos = tok.eos_token_id
                input_ids += [eos]; attention_mask += [1]; labels += [eos]
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        ds_tok = DatasetDict(
            train=ds_raw["train"].map(add_demo_and_tokenize, with_indices=True, remove_columns=ds_raw["train"].column_names, desc="Build+tokenize train"),
            eval =ds_raw["eval"].map(add_demo_and_tokenize,  with_indices=True, remove_columns=ds_raw["eval"].column_names,  desc="Build+tokenize eval")
        ).remove_columns([c for c in ds_raw["train"].column_names if c not in ("input_ids","attention_mask","labels")])
        ds_tok.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

        model = self._load_causal_model(MODEL_ID, prefer_4bit=True, eval_mode=False)

        # prefer explicit targets; fallback to heuristic guesser if needed
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        if not any(any(t in n for n,_ in model.named_modules()) for t in target_modules):
            target_modules = self.guess_lora_targets(model)

        lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=target_modules, task_type="CAUSAL_LM")
        collator = self.CausalLMPadCollator(tok.pad_token_id)

        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            peft_config=lora_cfg,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["eval"],
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-3)],
        )
        trainer.train()

        # Save artifacts (adapter + merged FP16)
        self.save_adapter_and_merged_fp16(trainer, tok, MODEL_ID, OUT_DIR)
        return trainer

    # ==========================================
    # 2) KD + QLoRA student (BASIC)  (from training function 2)
    #    Renamed trainer to avoid collision: DistillSFTTrainerBasic
    # ==========================================
    class DistillSFTTrainerBasic(SFTTrainer):
        def __init__(self, *args, teacher_model=None, kd_alpha=0.5, kd_temp=2.0, **kwargs):
            super().__init__(*args, **kwargs)
            assert teacher_model is not None
            self.teacher = teacher_model.eval()
            for p in self.teacher.parameters(): p.requires_grad_(False)
            self.kd_alpha = float(kd_alpha)
            self.kd_temp = float(kd_temp)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            out_s = model(**inputs, use_cache=False)
            logits_s = out_s.logits
            with torch.inference_mode():
                out_t = self.teacher(**inputs, use_cache=False)
                logits_t = out_t.logits

            # next-token alignment
            ls = logits_s[:, :-1, :].contiguous()
            lt = logits_t[:, :-1, :].contiguous()
            y  = labels[:, 1:].contiguous()
            m  = (y != -100)

            if not m.any():
                loss = torch.zeros((), device=logits_s.device, dtype=logits_s.dtype)
                return (loss, out_s) if return_outputs else loss

            logp_s = torch.log_softmax(ls, dim=-1)
            ce = torch.nn.functional.nll_loss(logp_s[m], y[m], reduction="mean")

            T = self.kd_temp
            logp_s_T = torch.log_softmax(ls / T, dim=-1)
            p_t_T    = torch.softmax(lt / T, dim=-1)
            kl_tok   = torch.nn.functional.kl_div(logp_s_T, p_t_T, reduction="none").sum(dim=-1)
            kd = (kl_tok[m]).mean() * (T * T)

            loss = self.kd_alpha * ce + (1.0 - self.kd_alpha) * kd
            return (loss, out_s) if return_outputs else loss

    def train_kd_basic(
        self,
        STUDENT_ID: str,
        TEACHER_ID: str,
        sft_cfg: SFTConfig,
        OUT_DIR: str = "student_kd_output",
        KD_DATASET_REPO: str = "zBotta/traffic-accidents-reports-kd-smollm2-360M-7k",
        DEMO_PROB: float = 0.8,
        KD_ALPHA: float = 0.5,
        KD_T: float = 2.0,
        RANK: int = 8,
        MAX_LEN: int = 1024
    ) -> SFTTrainer:
        """
        Trains a student model via KD + QLoRA (basic version). 
        KD expression is based on the basic T^2 weight (Hinton et al. 2020)
        """
        raw_kd = load_dataset(KD_DATASET_REPO)
        assert "train" in raw_kd and "eval" in raw_kd

        # tokenizers MUST match for logit KD
        tok_student = AutoTokenizer.from_pretrained(STUDENT_ID, use_fast=True)
        tok_teacher = AutoTokenizer.from_pretrained(TEACHER_ID, use_fast=True)
        if tok_student.pad_token_id is None: tok_student.pad_token = tok_student.eos_token
        if tok_teacher.pad_token_id is None: tok_teacher.pad_token = tok_teacher.eos_token
        assert tok_student.get_vocab() == tok_teacher.get_vocab(), \
            "Tokenizer mismatch: choose student & teacher with identical tokenizers for logit KD."

        # Build a small demo pool from train (use KD dataset inputs/targets)
        def ok(r): return bool(str(r.get("input","")).strip()) and bool(str(r.get("target","")).strip())
        ds_raw = DatasetDict(train=raw_kd["train"].filter(ok), eval=raw_kd["eval"].filter(ok))
        demo_pool = [(self.one_line(r["input"]), self.one_line(r["target"])) for r in ds_raw["train"]]

        def add_demo_and_tokenize(example, idx):
            cur_in, cur_out = self.one_line(example["input"]), self.one_line(example["target"])
            use_demo = (np.random.rand() < DEMO_PROB) and (len(demo_pool) > 1)
            if use_demo:
                d_in, d_out = demo_pool[(idx+1) % len(demo_pool)]
                prompt = self.prompt_with_demo(d_in, d_out, cur_in)
            else:
                prompt = self.prompt_no_demo(cur_in)

            prom = tok_student(prompt, add_special_tokens=True, truncation=True, max_length=MAX_LEN)
            prompt_ids = prom["input_ids"]; attn_prompt = prom["attention_mask"]

            if len(prompt_ids) >= MAX_LEN - 4:
                prom = tok_student(self.prompt_no_demo(cur_in), add_special_tokens=True, truncation=True, max_length=MAX_LEN)
                prompt_ids = prom["input_ids"]; attn_prompt = prom["attention_mask"]

            eos = tok_student.eos_token_id
            if eos is None:
                tok_student.pad_token = tok_student.eos_token
                eos = tok_student.eos_token_id

            budget = MAX_LEN - len(prompt_ids)
            if budget <= 0:
                prompt_ids = prompt_ids[:-1]; attn_prompt = attn_prompt[:-1]
                budget = 1

            allow = max(0, budget - 1)
            targ_ids = tok_student(cur_out, add_special_tokens=False, truncation=True, max_length=max(1, allow)).input_ids
            targ_ids = targ_ids[:allow]
            target_with_eos = targ_ids + [eos]

            input_ids      = prompt_ids + target_with_eos
            attention_mask = attn_prompt + [1]*len(target_with_eos)
            labels         = [-100]*len(prompt_ids) + target_with_eos
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        ds_tok = DatasetDict(
            train = ds_raw["train"].map(add_demo_and_tokenize, with_indices=True, remove_columns=ds_raw["train"].column_names, desc="Tokenize train"),
            eval  = ds_raw["eval"].map(add_demo_and_tokenize,  with_indices=True, remove_columns=ds_raw["eval"].column_names,  desc="Tokenize eval"),
        )

        student = self._load_causal_model(STUDENT_ID, prefer_4bit=True, eval_mode=False)
        teacher = self._load_causal_model(TEACHER_ID, prefer_4bit=True, eval_mode=True).eval()
        for p in teacher.parameters(): p.requires_grad_(False)

        lora_cfg = LoraConfig(
            r=RANK, lora_alpha=2*RANK, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM"
        )
        collator = self.CausalLMPadCollator(tok_student.pad_token_id)

        trainer = self.DistillSFTTrainerBasic(
            model=student,
            args=sft_cfg,
            peft_config=lora_cfg,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["eval"],
            data_collator=collator,
            teacher_model=teacher,
            kd_alpha=KD_ALPHA,
            kd_temp=KD_T,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-3)],
        )
        print("Accelerate mixed precision:", trainer.accelerator.mixed_precision)
        print("Trainable params:", sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))

        trainer.train()
        self.save_adapter_and_merged_fp16(trainer, tok_student, STUDENT_ID, OUT_DIR)
        return trainer

    # ==========================================
    # 3) KD + QLoRA student (ADVANCED)  (from training function 3)
    #    Renamed trainer to avoid collision: DistillSFTTrainerAdvanced
    # ==========================================
    class DistillSFTTrainerAdvanced(SFTTrainer):
        def __init__(self, *args,
                     teacher_model=None,
                     kd_alpha_ce_start=0.85,
                     kd_alpha_ce_end=0.65,
                     kd_warmup_epochs=2,
                     kd_ramp_epochs=2,
                     kd_temp=3.0,
                     kd_top_K=128,
                     beta_ul=0.1,
                     ce_smoothing=0.05,
                     soft_kd_gamma=2.0,
                     soft_kd_wmin=0.05,
                     **kwargs):
            super().__init__(*args, **kwargs)
            assert teacher_model is not None
            self.teacher = teacher_model.eval()
            for p in self.teacher.parameters(): p.requires_grad_(False)
            self.kd_alpha_ce_start = float(kd_alpha_ce_start)
            self.kd_alpha_ce_end   = float(kd_alpha_ce_end)
            self.kd_warmup_epochs  = int(kd_warmup_epochs)
            self.kd_ramp_epochs    = int(kd_ramp_epochs)
            self.kd_temp           = float(kd_temp)
            self.kd_top_K          = int(kd_top_K)
            self.beta_ul           = float(beta_ul)
            self.ce_smoothing      = float(ce_smoothing)
            self.soft_kd_gamma     = float(soft_kd_gamma)
            self.soft_kd_wmin      = float(soft_kd_wmin)

        def _ce_weight(self):
            e = float(self.state.epoch or 0.0)
            if e < self.kd_warmup_epochs: return 1.0
            if e < self.kd_warmup_epochs + self.kd_ramp_epochs:
                t = (e - self.kd_warmup_epochs) / max(1.0, self.kd_ramp_epochs)
                return self.kd_alpha_ce_start + (self.kd_alpha_ce_end - self.kd_alpha_ce_start) * t
            return self.kd_alpha_ce_end

        def _smoothed_ce(self, logp, labels, mask, eps=0.05):
            gather = torch.gather(logp, -1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            nll = -(gather[mask]).mean()
            H   = -(logp[mask]).mean()
            return (1 - eps) * nll + eps * H

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out_s = model(**inputs, use_cache=False)
            logits_s = out_s.logits

            logits_s_shift = logits_s[:, :-1, :]
            labels_shift   = labels[:, 1:]
            mask_shift     = (labels_shift != -100)

            logp_s_shift = torch.log_softmax(logits_s_shift, dim=-1)
            ce = self._smoothed_ce(logp_s_shift, labels_shift, mask_shift, eps=self.ce_smoothing)

            ce_w = self._ce_weight()
            if ce_w >= 0.999:
                return (ce, out_s) if return_outputs else ce

            with torch.inference_mode():
                out_t = self.teacher(**inputs, use_cache=False)
                logits_t_shift = out_t.logits[:, :-1, :]

            T = self.kd_temp
            V = logits_s_shift.shape[-1]
            K = min(self.kd_top_K, V)
            topk_val, topk_idx = torch.topk(logits_t_shift / T, k=K, dim=-1)
            p_t_topk = torch.softmax(topk_val, dim=-1)
            conf = p_t_topk[..., 0]
            w = torch.clamp(conf ** self.soft_kd_gamma, min=self.soft_kd_wmin) * mask_shift.float()
            has_kd = (w.sum() > 0)

            if has_kd:
                logp_s_T_shift = torch.log_softmax(logits_s_shift / T, dim=-1)
                gather_logp_s  = torch.gather(logp_s_T_shift, -1, topk_idx)
                kl_pos = torch.sum(p_t_topk * (torch.log(p_t_topk + 1e-9) - gather_logp_s), dim=-1)
                kd = (w * kl_pos).sum() / (w.sum() + 1e-6)
            else:
                kd = torch.zeros((), device=logits_s.device)

            beta_ul = self.beta_ul
            if beta_ul > 0:
                prev_tok = inputs["input_ids"][:, :-1]
                p_next   = torch.softmax(logits_s_shift, dim=-1)
                p_repeat = torch.gather(p_next, -1, prev_tok.unsqueeze(-1)).squeeze(-1)
                ul = -torch.log(1.0 - p_repeat + 1e-6)
                ul = (ul[mask_shift]).mean() if mask_shift.any() else torch.zeros((), device=logits_s.device)
            else:
                ul = torch.zeros((), device=logits_s.device)

            kd_w = 1.0 - ce_w
            loss = ce_w * ce + kd_w * kd + beta_ul * ul

            if self.state.global_step % 50 == 0:
                self.log({
                    "train/ce": float(ce.detach()),
                    "train/kd": float(kd.detach()),
                    "train/ul": float(ul.detach()),
                    "train/ce_weight": float(ce_w),
                    "train/kd_weight": float(kd_w),
                })
            return (loss, out_s) if return_outputs else loss

    class SampleGenerateCallback(TrainerCallback):
        def __init__(self, tok, make_prompt_fn, eval_ds, max_new=180, resp_marker="### Response:\n"):
            self.tok = tok
            self.make_prompt_fn = make_prompt_fn
            self.eval_ds = eval_ds
            self.max_new = max_new
            self.resp_marker = resp_marker
            self.rng = np.random.default_rng(123)

        @staticmethod
        def _one_line(s: str) -> str:
            return re.sub(r"\s+", " ", str(s).replace("\n", " ")).strip()

        def _extract_report(self, decoded_full: str, decoded_new: str) -> str:
            marker = self.resp_marker.strip()
            idx = decoded_full.find(marker)
            out = decoded_full[idx + len(marker):] if idx != -1 else decoded_new
            out = out.split("###", 1)[0]
            out = re.sub(r'^\s*(#+\s*Response:)?\s*', '', out)
            out = re.sub(r'^\s*(Factual\s+(paragraph|report)\s*:)\s*', '', out, flags=re.I)
            return self._one_line(out)

        def _gen(self, model, prompt: str, MAX_LEN=1024) -> str:
            device = next(model.parameters()).device
            enc = self.tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=self.max_new,
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.05,
                    renormalize_logits=True,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id
                )
            input_len = int(enc["attention_mask"].sum().item())
            decoded_full = self.tok.decode(out[0], skip_special_tokens=True)
            decoded_new  = self.tok.decode(out[0, input_len:], skip_special_tokens=True)
            return self._extract_report(decoded_full, decoded_new)

        def on_epoch_end(self, args, state, control, **kwargs):
            model = kwargs["model"]
            was_training = model.training
            model.eval()
            try:
                idx = int(self.rng.integers(0, len(self.eval_ds)))
                cur_in = self._one_line(self.eval_ds[idx]["input"])
                prompt = self.make_prompt_fn(cur_in)
                text = self._gen(model, prompt)
                print(f"\n[epoch {int(state.epoch)}] sample report (post-RESP_TMPL)")
                print("Report:", (text[:300] + ("..." if len(text) > 300 else "")))
            finally:
                if was_training: model.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def train_kd_advanced(
        self,
        STUDENT_ID: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        TEACHER_ID: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        sft_cfg: Optional[SFTConfig] = None,
        OUT_DIR: str = "student_kd_output",
        GOLD_DATASET_ID: str = "zBotta/traffic-accidents-reports-5k",
        USE_KD_DATASET: bool = False,
        KD_DATASET_REPO: Optional[str] = None,
        DEMO_PROB: float = 0.8,
        KD_ALPHA: List[float] = [0.85, 0.65],
        KD_T: float = 2.0,
        KD_TOP_K: int = 128,
        SOFT_KD_WMIN: float = 0.05,
        SOFT_KD_GAMMA: float = 2.0,
        BETA_UL: float = 0.1,
        RANK: int = 8,
        LORA_DROPOUT: float = 0.05,
        MAX_LEN: int = 1024,
        EVAL_N: int = 5,
        GEN_MAX_NEW: int = 256,
    ) -> SFTTrainer:
        # Load GOLD dataset
        raw_gold = load_dataset(GOLD_DATASET_ID)
        assert "train" in raw_gold and "eval" in raw_gold
        def ok(r): return bool(str(r.get("input","")).strip()) and bool(str(r.get("target","")).strip())
        ds_gold = DatasetDict(train=raw_gold["train"].filter(ok), eval=raw_gold["eval"].filter(ok))
        demo_pool = [(self.one_line(r["input"]), self.one_line(r["target"])) for r in ds_gold["train"]]

        tok_student = AutoTokenizer.from_pretrained(STUDENT_ID, use_fast=True)
        tok_teacher = AutoTokenizer.from_pretrained(TEACHER_ID, use_fast=True)
        if tok_student.pad_token_id is None: tok_student.pad_token = tok_student.eos_token
        if tok_teacher.pad_token_id is None: tok_teacher.pad_token = tok_teacher.eos_token
        if tok_student.get_vocab() != tok_teacher.get_vocab():
            raise RuntimeError("Tokenizer mismatch: student and teacher must share IDENTICAL vocab for logit KD.")

        teacher_for_logits = self._load_causal_model(TEACHER_ID, prefer_4bit=True, eval_mode=True).eval()
        for p in teacher_for_logits.parameters(): p.requires_grad_(False)
        # Student model + LoRA
        student = self._load_causal_model(STUDENT_ID, prefer_4bit=True, eval_mode=False)

        # Optional KD dataset source
        def try_load_kd_repo(repo_id: Optional[str]) -> Optional[DatasetDict]:
            if not repo_id: return None
            try:
                ds = load_dataset(repo_id)
                if "train" in ds and "eval" in ds and all(k in ds["train"].column_names for k in ("input","target")):
                    print(f"✅ KD dataset loaded from: {repo_id}")
                    return DatasetDict(train=ds["train"], eval=ds["eval"])
                else:
                    print(f"⚠️ KD repo {repo_id} missing expected splits/columns; will generate from GOLD.")
                    return None
            except Exception as e:
                print(f"⚠️ Could not load KD repo {repo_id}: {e}\nWill generate teacher outputs from GOLD.")
                return None

        ds_base = None
        need_kd_from_gold = False
        if USE_KD_DATASET:
            ds_base = try_load_kd_repo(KD_DATASET_REPO)
            need_kd_from_gold = ds_base is None
        else:
            ds_base = ds_gold

        # Generate teacher texts if needed (no-demo prompts)
        def gen_teacher_texts(ds: Dataset, batch_size: int = 4, max_new: int = 256) -> List[str]:
            outs = []
            device = next(teacher_for_logits.parameters()).device
            for i in range(0, len(ds), batch_size):
                batch = ds[i:i+batch_size]
                prompts = [self.prompt_no_demo(self.one_line(x)) for x in batch["input"]]
                enc = tok_teacher(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.inference_mode():
                    gen = teacher_for_logits.generate(
                        **enc, do_sample=False, temperature=1.0, max_new_tokens=max_new,
                        eos_token_id=tok_teacher.eos_token_id, pad_token_id=tok_teacher.pad_token_id
                    )
                input_lens = enc["attention_mask"].sum(dim=1).tolist()
                for j in range(len(prompts)):
                    new_ids = gen[j, input_lens[j]:]
                    text = tok_teacher.decode(new_ids, skip_special_tokens=True)
                    outs.append(self.one_line(text))
            return outs

        if need_kd_from_gold:
            print("🛠 Generating teacher outputs from GOLD for KD...")
            kd_train_out = gen_teacher_texts(ds_gold["train"])
            kd_eval_out  = gen_teacher_texts(ds_gold["eval"])
            ds_base = DatasetDict(
                train = ds_gold["train"].add_column("target", kd_train_out),
                eval  = ds_gold["eval"].add_column("target",  kd_eval_out),
            )

        # Tokenization (response-only labels, always EOS-labeled)
        def add_demo_and_tokenize(example, idx):
            cur_in, cur_out = self.one_line(example["input"]), self.one_line(example["target"])
            use_demo = (np.random.rand() < DEMO_PROB) and (len(demo_pool) > 1)
            if use_demo:
                d_in, d_out = demo_pool[(idx+1) % len(demo_pool)]
                prompt = self.prompt_with_demo(d_in, d_out, cur_in)
            else:
                prompt = self.prompt_no_demo(cur_in)

            prom = tok_student(prompt, add_special_tokens=True, truncation=True, max_length=MAX_LEN)
            prompt_ids = prom["input_ids"]; attn_prompt = prom["attention_mask"]

            if len(prompt_ids) >= MAX_LEN - 4:
                prom = tok_student(self.prompt_no_demo(cur_in), add_special_tokens=True, truncation=True, max_length=MAX_LEN)
                prompt_ids = prom["input_ids"]; attn_prompt = prom["attention_mask"]

            eos = tok_student.eos_token_id
            if eos is None:
                tok_student.pad_token = tok_student.eos_token
                eos = tok_student.eos_token_id

            budget = MAX_LEN - len(prompt_ids)
            if budget <= 0:
                prompt_ids = prompt_ids[:-1]; attn_prompt = attn_prompt[:-1]
                budget = 1

            allow = max(0, budget - 1)
            targ_ids = tok_student(cur_out, add_special_tokens=False, truncation=True, max_length=max(1, allow)).input_ids
            targ_ids = targ_ids[:allow]
            target_with_eos = targ_ids + [eos]

            input_ids      = prompt_ids + target_with_eos
            attention_mask = attn_prompt + [1]*len(target_with_eos)
            labels         = [-100]*len(prompt_ids) + target_with_eos
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        ds_tok = DatasetDict(
            train = ds_base["train"].map(add_demo_and_tokenize, with_indices=True, remove_columns=ds_base["train"].column_names, desc="Tokenize train"),
            eval  = ds_base["eval"].map(add_demo_and_tokenize,  with_indices=True, remove_columns=ds_base["eval"].column_names,  desc="Tokenize eval"),
        )

        lora_cfg = LoraConfig(
            r=RANK, lora_alpha=2*RANK, lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM"
        )
        collator = self.CausalLMPadCollator(tok_student.pad_token_id)

        # Sample-gen callback (epoch-end)
        def make_eval_prompt(x): return self.prompt_no_demo(self.one_line(x))
        sample_cb = self.SampleGenerateCallback(
            tok_student, make_eval_prompt, ds_gold["eval"], max_new=300, resp_marker=self.RESP_TMPL
        )

        trainer = self.DistillSFTTrainerAdvanced(
            model=student,
            args=sft_cfg,
            peft_config=lora_cfg,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["eval"],
            data_collator=collator,
            teacher_model=teacher_for_logits,
            kd_alpha_ce_start=KD_ALPHA[0],
            kd_alpha_ce_end=KD_ALPHA[1],
            kd_warmup_epochs=2,
            kd_ramp_epochs=2,
            kd_temp=KD_T,
            kd_top_K=KD_TOP_K,
            beta_ul=BETA_UL,
            ce_smoothing=0.05,
            soft_kd_gamma=SOFT_KD_GAMMA,
            soft_kd_wmin=SOFT_KD_WMIN,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-3), sample_cb],
        )

        print("Accelerate mixed precision:", trainer.accelerator.mixed_precision)
        print("Trainable params:", sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))

        trainer.train()
        _, merged_dir = self.save_adapter_and_merged_fp16(trainer, tok_student, STUDENT_ID, OUT_DIR)

        # ---- Quick eval + latency (optional) ----
        try:
            import evaluate
            rouge = evaluate.load("rouge")
            bleu  = evaluate.load("sacrebleu")
        except Exception as e:
            print("Skipping metrics (evaluate not available):", e)
            return trainer

        rng = np.random.default_rng(42)
        idxs = rng.choice(len(ds_gold["eval"]), size=min(EVAL_N, len(ds_gold["eval"])), replace=False)
        eval_subset = ds_gold["eval"].select(list(idxs))
        prompts = [make_eval_prompt(x["input"]) for x in eval_subset]
        refs    = [[self.one_line(x["target"])] for x in eval_subset]

        def gen_text(model, tok, prompts, max_new=GEN_MAX_NEW, do_sample=False, temperature=1.0):
            device = next(model.parameters()).device
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    do_sample=do_sample, temperature=temperature,
                    max_new_tokens=max_new,
                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
                )
            input_lens = enc["attention_mask"].sum(dim=1).tolist()
            texts = []
            for i in range(len(prompts)):
                out_ids = gen[i, input_lens[i]:]
                texts.append(self.one_line(tok.decode(out_ids, skip_special_tokens=True)))
            return texts, gen, enc

        student_eval = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.float16, device_map="auto").eval()
        teacher_eval = teacher_for_logits

        pred_teacher, _, _ = gen_text(teacher_eval, tok_teacher, prompts, do_sample=False, temperature=1.0)
        rouge_teacher = rouge.compute(predictions=pred_teacher, references=[r[0] for r in refs], use_stemmer=True)
        bleu_teacher  = bleu.compute(predictions=pred_teacher, references=refs)
        print("\n=== Accuracy on", len(prompts), "samples (Teacher) ===")
        print("ROUGE-Lsum:", round(rouge_teacher["rougeLsum"], 4), " | SacreBLEU:", round(bleu_teacher["score"], 2))

        pred_student, _, _ = gen_text(student_eval, tok_student, prompts, do_sample=False, temperature=1.0)
        rouge_student = rouge.compute(predictions=pred_student, references=[r[0] for r in refs], use_stemmer=True)
        bleu_student  = bleu.compute(predictions=pred_student, references=refs)
        print("\n=== Accuracy on", len(prompts), "samples (Student) ===")
        print("ROUGE-Lsum:", round(rouge_student["rougeLsum"], 4), " | SacreBLEU:", round(bleu_student["score"], 2))

        for i,(ref,ts,ss) in enumerate(zip([r[0] for r in refs], pred_teacher, pred_student)):
            print(f"\n--- Sample {i+1} ---")
            print("Ref  :", ref[:200] + ("..." if len(ref)>200 else ""))
            print("Teach:", ts[:200] + ("..." if len(ts)>200 else ""))
            print("Stud :", ss[:200] + ("..." if len(ss)>200 else ""))

        def latency_test(model, tok, prompts, max_new=GEN_MAX_NEW, label="model"):
            device = next(model.parameters()).device
            # warmup
            _ = gen_text(model, tok, prompts[:1], max_new=32, do_sample=False, temperature=1.0)
            if device.type == "cuda": torch.cuda.synchronize()
            times, toks = [], []
            for prompt in prompts:
                enc = tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
                enc = {k: v.to(device) for k, v in enc.items()}
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    gen = model.generate(
                        **enc, do_sample=False, temperature=1.0, max_new_tokens=max_new,
                        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
                    )
                if device.type == "cuda": torch.cuda.synchronize()
                input_len = int(enc["attention_mask"].sum().item())
                out_len = int(gen.shape[1] - input_len)
                times.append(time.perf_counter() - t0); toks.append(out_len)
            avg_lat = sum(times)/len(times); avg_tokens = sum(toks)/len(toks)
            tps = (sum(toks)/sum(times)) if sum(times)>0 else 0.0
            print(f"\n=== Latency ({label}) ===")
            for i,(dt,tk) in enumerate(zip(times,toks)):
                print(f"ex{i+1}: {dt:.3f}s, gen_tokens={tk}, tok/s={tk/dt if dt>0 else 0.0:.1f}")
            print(f"AVG: {avg_lat:.3f}s per sample | {tps:.1f} tok/s | avg_gen_tokens={avg_tokens:.1f}")

        latency_test(teacher_eval, tok_teacher, prompts, max_new=GEN_MAX_NEW, label="Teacher")
        latency_test(student_eval, tok_student, prompts, max_new=GEN_MAX_NEW, label="Student")

        print("\nDone.")
        return trainer

if __name__ == "__main__":
    # ==== Example training an advanced KD  ====
    TRAINING_DIR = "app/datasets/training"
    out_dir  = TRAINING_DIR + "/student_kd_smol_360m_adv"

    # -------- Trainer config --------
    sft_cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=1, # 20-30
        per_device_train_batch_size=4, #4
        gradient_accumulation_steps=16, #16    # eff batch ~64
        learning_rate=2e-5, # 2e-5
        lr_scheduler_type="cosine",
        warmup_ratio=0.1, # 0.08
        weight_decay=0.05,
        label_smoothing_factor=0.05,
        max_grad_norm=0.5, # 1.0
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # eval_loss
        greater_is_better=False,
        fp16=False, bf16=False,
        optim="adamw_bnb_8bit",
        packing=False,
        max_length=1024,
        gradient_checkpointing=True,       # ON
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=4,   # new
        dataloader_pin_memory=True,  # new
        report_to="none",
        seed=42,
    )

    try:
        mgr = TrainerManager()
        trainer = mgr.train_kd_advanced(
            TEACHER_ID="HuggingFaceTB/SmolLM2-360M-Instruct",      # Teacher
            STUDENT_ID="HuggingFaceTB/SmolLM2-135M-Instruct",      # Student
            OUT_DIR=out_dir,
            GOLD_DATASET_ID="zBotta/traffic-accidents-reports-5k", 
            USE_KD_DATASET = False,            # Keep it false as we do not need to regenerate targets with teacher
            DEMO_PROB = 1,
            KD_ALPHA= [0.9, 0.75],     # [start, end]  # CE weight (KL gets 1-KD_ALPHA)
            KD_T= 3,                 # temperature for KD
            KD_TOP_K= 64,              # Top-K distillation
            SOFT_KD_GAMMA=1.5,
            SOFT_KD_WMIN=0.20,
            BETA_UL=0.1,
            RANK= 8,
            LORA_DROPOUT = 0.05,
            sft_cfg = sft_cfg,
            MAX_LEN = 1024,
            EVAL_N = 5,                   # number of evaluation examples for metrics & latency
            GEN_MAX_NEW = 256,            # generation budget for eval/latency
        )
        # Plot training results
        mgr.plot_training_results(trainer, out_dir)
    except Exception as e:
        print(f"Error while training KD trainer: {e}")