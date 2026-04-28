#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SALAD_ROOT = WORKSPACE_ROOT / "humanmodels" / "salad"
BENCHMARK_RESULTS_DIR = WORKSPACE_ROOT / "experiments" / "benchmark" / "results"
BENCHMARK_PARSE_SCRIPT = WORKSPACE_ROOT / "experiments" / "benchmark" / "scripts" / "parse_results.py"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lam"))
sys.path.insert(0, str(REPO_ROOT / "worldmodel"))
if str(SALAD_ROOT) not in sys.path:
    sys.path.append(str(SALAD_ROOT))

from common.humanml_representation import humanml_vector_to_sal_rep
from lam.model import load_lam_from_checkpoint
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from data.t2m_dataset import Text2MotionDatasetEval, collate_fn as salad_collate_fn
from utils.get_opt import get_opt
from utils.word_vectorizer import WordVectorizer
from utils.metrics import (
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
    euclidean_distance_matrix,
)
from worldmodel.train import (
    _build_noise_scheduler,
    _lengths_to_latent_mask,
    _pool_action_sequence,
    load_action_prior_from_checkpoint,
    load_official_action_adapter_from_checkpoint,
)
from vwm.models.salad_official import load_official_salad_action_denoiser, load_official_salad_vae


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mean_conf(values):
    arr = np.array(values, dtype=np.float64)
    return arr.mean(axis=0), 1.96 * arr.std(axis=0) / np.sqrt(max(1, len(arr)))


def _write_benchmark_result(model_key: str, group_name: str, group_metrics: dict, output_path: Path, results_dir: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("benchmark_parse_results", BENCHMARK_PARSE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark parser from {BENCHMARK_PARSE_SCRIPT}")
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "model_key": model_key,
        "run_id": run_id,
        "gpu_id": "",
        "source_log": str(output_path.resolve()),
        "source_log_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "status": "parsed",
        "metrics": {group_name: group_metrics},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{model_key}.{run_id}.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    parser_module.write_summary_csv(results_dir)
    parser_module.write_manifest(results_dir)
    return result_path


def _load_models(args, device: torch.device):
    adapter_cfg = load_config(args.adapter_config)
    if args.adapter_checkpoint and Path(args.adapter_checkpoint).exists():
        denoiser, vae, lam, world_cfg = load_official_action_adapter_from_checkpoint(args.adapter_checkpoint)
    else:
        lam = load_lam_from_checkpoint(adapter_cfg["model"]["lam_checkpoint"])
        vae = load_official_salad_vae(
            adapter_cfg["model"]["official_salad_vae_opt"],
            adapter_cfg["model"]["official_salad_vae_checkpoint"],
            device,
        )
        denoiser = load_official_salad_action_denoiser(
            adapter_cfg["model"]["official_salad_denoiser_opt"],
            adapter_cfg["model"]["official_salad_denoiser_checkpoint"],
            vae_dim=vae.latent_dim,
            action_dim=lam.latent_dim,
            device=device,
            train_base=False,
        )
        world_cfg = adapter_cfg
    denoiser = denoiser.to(device).eval()
    vae = vae.to(device).eval()
    vae.freeze()
    lam = lam.to(device).eval()
    for param in lam.parameters():
        param.requires_grad = False
    prior = None
    if args.mode == "prior_action":
        if not args.prior_checkpoint:
            raise ValueError("--prior-checkpoint is required for --mode prior_action")
        prior, _ = load_action_prior_from_checkpoint(args.prior_checkpoint)
        prior = prior.to(device).eval()
    return denoiser, vae, lam, prior, world_cfg


@torch.no_grad()
def _sample_motion(
    denoiser,
    vae,
    lam,
    prior,
    scheduler: DDIMScheduler,
    mode: str,
    texts: list[str],
    motion: torch.Tensor,
    lengths: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
    cond_scale: float,
    num_inference_timesteps: int,
    unit_length: int,
) -> torch.Tensor:
    device = motion.device
    z_ref = vae.encode_deterministic(motion)[0]
    len_mask = _lengths_to_latent_mask(lengths, z_ref.shape[1], unit_length=unit_length)
    latents = torch.randn_like(z_ref) * scheduler.init_noise_sigma
    latents = latents * len_mask[..., None, None].float()

    if mode == "oracle_action":
        raw_motion = motion.detach().cpu().numpy() * std + mean
        action_source = torch.from_numpy(humanml_vector_to_sal_rep(raw_motion)).to(device=device, dtype=motion.dtype)
        action_seq = lam.encode_action_sequence(action_source, texts=texts)
        action_seq = _pool_action_sequence(action_seq, z_ref.shape[1])
        use_action = True
    elif mode == "prior_action":
        action_seq = prior(texts, lengths)
        if action_seq.shape[1] != z_ref.shape[1]:
            action_seq = _pool_action_sequence(action_seq, z_ref.shape[1])
        use_action = True
    elif mode == "salad_no_action":
        action_seq = torch.zeros(motion.shape[0], 0, lam.latent_dim, device=device, dtype=motion.dtype)
        use_action = False
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if action_seq.shape[1] > 0:
        action_seq = action_seq * len_mask[..., None].float()

    scheduler.set_timesteps(num_inference_timesteps)
    timesteps = scheduler.timesteps.to(device)
    for timestep in timesteps:
        if cond_scale > 1.0:
            latent_input = torch.cat([latents, latents], dim=0)
            mask_input = torch.cat([len_mask, len_mask], dim=0)
            if use_action:
                action_input = torch.cat([torch.zeros_like(action_seq), action_seq], dim=0)
            else:
                action_input = torch.zeros(latent_input.shape[0], 0, lam.latent_dim, device=device, dtype=motion.dtype)
            pred = denoiser(
                noisy_motion_latent=latent_input,
                timesteps=timestep.expand(latent_input.shape[0]),
                texts=[""] * len(texts) + texts,
                action_latent_seq=action_input,
                len_mask=mask_input,
                use_cached_clip=True,
                use_action_condition=use_action,
            )
            pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
            pred = pred_uncond + cond_scale * (pred_cond - pred_uncond)
        else:
            pred = denoiser(
                noisy_motion_latent=latents,
                timesteps=timestep.expand(latents.shape[0]),
                texts=texts,
                action_latent_seq=action_seq,
                len_mask=len_mask,
                use_cached_clip=True,
                use_action_condition=use_action,
            )
        latents = scheduler.step(pred, timestep, latents).prev_sample
        latents = latents * len_mask[..., None, None].float()
    denoiser.remove_clip_cache()
    pred_motion = vae.decode(latents)
    mask = torch.arange(pred_motion.shape[1], device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    return pred_motion.masked_fill(mask.unsqueeze(-1), 0.0)


@torch.no_grad()
def evaluate_once(args, device, eval_wrapper, val_loader, dataset, denoiser, vae, lam, prior, scheduler, cfg):
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    r_precision = 0
    matching_score = 0.0
    nb_sample = 0
    num_mm_batch = args.mm_batches
    mean = dataset.mean
    std = dataset.std
    unit_length = cfg["data"]["dataset"].get("unit_length", 4)

    for batch_idx, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, _ = batch
        motion = motion.to(device, dtype=torch.float32)
        m_length = m_length.to(device, dtype=torch.long)
        et, _ = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        if batch_idx < num_mm_batch:
            mm_batch = []
            for _ in range(args.mm_repeats):
                pred_motion = _sample_motion(
                    denoiser,
                    vae,
                    lam,
                    prior,
                    scheduler,
                    args.mode,
                    list(caption),
                    motion,
                    m_length,
                    mean,
                    std,
                    args.cond_scale,
                    args.num_inference_timesteps,
                    unit_length,
                )
                _, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)
                mm_batch.append(em_pred.unsqueeze(1))
            motion_multimodality.append(torch.cat(mm_batch, dim=1))
        else:
            pred_motion = _sample_motion(
                denoiser,
                vae,
                lam,
                prior,
                scheduler,
                args.mode,
                list(caption),
                motion,
                m_length,
                mean,
                std,
                args.cond_scale,
                args.num_inference_timesteps,
                unit_length,
            )
            _, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)

        _, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        temp_r = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        r_precision += temp_r
        matching_score += temp_match
        nb_sample += motion.shape[0]
        if args.num_batches is not None and batch_idx + 1 >= args.num_batches:
            break

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    mm = 0.0
    if motion_multimodality:
        mm_np = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        mm = calculate_multimodality(mm_np, args.mm_times)
    return {
        "R_precision": r_precision / nb_sample,
        "FID": calculate_frechet_distance(gt_mu, gt_cov, mu, cov),
        "MM_Dist": matching_score / nb_sample,
        "Diversity": calculate_diversity(motion_pred_np, min(args.diversity_times, len(motion_pred_np))),
        "MultiModality": mm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["salad_no_action", "oracle_action", "prior_action"], required=True)
    parser.add_argument("--adapter-config", default="/workspace/AdaMotion/configs/salad_adapter_mom_full.yaml")
    parser.add_argument("--adapter-checkpoint", default="/workspace/AdaMotion/experiments/salad_adapter_mom_full/world_best.pt")
    parser.add_argument("--prior-checkpoint", default="/workspace/AdaMotion/experiments/salad_prior_mom_full/action_prior_best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replication-times", type=int, default=20)
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--mm-batches", type=int, default=3)
    parser.add_argument("--mm-repeats", type=int, default=30)
    parser.add_argument("--mm-times", type=int, default=10)
    parser.add_argument("--diversity-times", type=int, default=300)
    parser.add_argument("--num-inference-timesteps", type=int, default=50)
    parser.add_argument("--cond-scale", type=float, default=7.5)
    parser.add_argument("--output", default="/workspace/AdaMotion/experiments/evals/official_salad_action_benchmark.json")
    parser.add_argument("--benchmark-results-dir", default=str(BENCHMARK_RESULTS_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    cfg = load_config(args.adapter_config)
    original_cwd = Path.cwd()
    os.chdir(SALAD_ROOT)
    try:
        dataset_opt_path = "checkpoints/t2m/Comp_v6_KLD005/opt.txt"
        wrapper_opt = get_opt(dataset_opt_path, device)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        mean = np.load(Path(wrapper_opt.meta_dir) / "mean.npy")
        std = np.load(Path(wrapper_opt.meta_dir) / "std.npy")
        w_vectorizer = WordVectorizer("./glove", "our_vab")
        split_file = Path(wrapper_opt.data_root) / f"{args.split}.txt"
        dataset = Text2MotionDatasetEval(wrapper_opt, mean, std, str(split_file), w_vectorizer)
        val_loader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=0,
            drop_last=True,
            shuffle=True,
            collate_fn=salad_collate_fn,
        )
    finally:
        os.chdir(original_cwd)

    denoiser, vae, lam, prior, world_cfg = _load_models(args, device)
    scheduler = _build_noise_scheduler(world_cfg)
    raw_metrics = OrderedDict(
        {
            "R_precision_top1": [],
            "R_precision_top2": [],
            "R_precision_top3": [],
            "FID": [],
            "MM_Dist": [],
            "Diversity": [],
            "MultiModality": [],
        }
    )
    for _ in range(args.replication_times):
        metrics = evaluate_once(args, device, eval_wrapper, val_loader, dataset, denoiser, vae, lam, prior, scheduler, cfg)
        raw_metrics["R_precision_top1"].append(float(metrics["R_precision"][0]))
        raw_metrics["R_precision_top2"].append(float(metrics["R_precision"][1]))
        raw_metrics["R_precision_top3"].append(float(metrics["R_precision"][2]))
        for key in ["FID", "MM_Dist", "Diversity", "MultiModality"]:
            raw_metrics[key].append(float(metrics[key]))

    summary = {
        "mode": args.mode,
        "replication_times": args.replication_times,
        "adapter_config": str(Path(args.adapter_config).resolve()),
        "adapter_checkpoint": str(Path(args.adapter_checkpoint).resolve()) if args.adapter_checkpoint else None,
        "prior_checkpoint": str(Path(args.prior_checkpoint).resolve()) if args.prior_checkpoint else None,
        "metrics": {},
    }
    for key, values in raw_metrics.items():
        mean, conf = _mean_conf(values)
        summary["metrics"][key] = {"mean": float(mean), "conf": float(conf), "values": values}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    model_key = {
        "salad_no_action": "adamotion_salad_no_action",
        "oracle_action": "adamotion_oracle_action",
        "prior_action": "adamotion_prior_action",
    }[args.mode]
    _write_benchmark_result(model_key, args.mode, summary["metrics"], output_path, Path(args.benchmark_results_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
