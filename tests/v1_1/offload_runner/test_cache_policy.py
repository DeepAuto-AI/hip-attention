import gc
import os
import random
import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from hip_research.utils.load_checkouts import load_checkouts

from hip_attn.v1_1.attention2_draft_prefetch import (
    HiPAttentionArgs as HiPAttentionArgs11,
)
from hip_attn.v1_1.attention2_draft_prefetch import hip_attention as hip_attention_11
from hip_attn.v1_1.offload_runner.cache_policy import (
    access_log_to_dense,
    access_score_log_to_dense,
    img_reduce,
    perform_gd_score,
    perform_lfu,
    perform_lfu_timestep_aware,
    perform_lru,
    perform_lru_hot,
    perform_lru_hot_prefetch,
    perform_lru_hot_prefetch_unified,
    perform_lru_hot_score,
    perform_lru_k,
    perform_lru_score,
    perform_lru_tie_break_lfu,
    perform_lru_tie_break_lre,
)


class TestCachePolicy(unittest.TestCase):

    def setUp(self):
        os.environ["HIP_DISABLE_AUTOTUNE"] = "1"
        self.exp_result_root = "./saves/cache_policy"
        self.visualization_root = "./saves/cache_policy/visualizations"
        os.makedirs(self.exp_result_root, exist_ok=True)
        os.makedirs(self.visualization_root, exist_ok=True)

    def test_main_exp(self):
        seq_len = 1024 * 64
        q, k, v, out, cos, sin = load_checkouts(
            idx=0, window=40, seq_len=seq_len, return_cos_sin=True, dtype=torch.bfloat16
        )
        print(q.shape, k.shape, v.shape)

        HEAD = q.shape[0]
        HEAD_KV = k.shape[0]

        def reshape(x, HEAD):
            N, T, H = x.shape
            x = (
                x.contiguous()
                .view(N // HEAD, HEAD, T, H)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            assert x.shape == (N // HEAD, T, HEAD, H)
            assert x.is_contiguous()
            return x

        q = reshape(q, HEAD)
        k = reshape(k, HEAD_KV)
        v = reshape(v, HEAD_KV)
        out = reshape(out, HEAD)

        args = HiPAttentionArgs11(
            mask_k=512,
            block_size_q=32,
            block_stride_q=2,
            block_size_k=2,
            block_stride_k=1,
            sliding_window_size=512,
            sink_token_size=16,
            output_key_access_log=False,
            output_block_access_log=True,
        )
        refresh_interval = 8

        BSZ, TDST, HEAD, HID = q.shape
        B = BSZ * HEAD

        block_access_logs = []
        block_access_scores = []
        block_access_counts = []
        for itdst in tqdm.tqdm(
            range(0, TDST, refresh_interval),
            desc="sample",
            dynamic_ncols=True,
            leave=False,
        ):
            _, metadata = hip_attention_11(
                q[:, itdst : itdst + 1, :, :],
                k[:, : itdst + 1],
                v[:, : itdst + 1],
                args=args,
            )
            log = metadata.block_access_log.cpu()
            score = metadata.block_access_score.float().cpu()
            count = metadata.block_access_count.cpu()
            block_access_logs.append(log)
            block_access_scores.append(score)
            block_access_counts.append(count)

        block_access_log = torch.full(
            (B, len(block_access_logs), max(t.shape[-1] for t in block_access_logs)),
            dtype=block_access_logs[0].dtype,
            fill_value=-1,
        )
        block_access_score = torch.full(
            (
                B,
                len(block_access_scores),
                max(t.shape[-1] for t in block_access_scores),
            ),
            dtype=block_access_scores[0].dtype,
            fill_value=0,
        )
        block_access_count = torch.full(
            (
                B,
                len(block_access_counts),
            ),
            dtype=block_access_counts[0].dtype,
            fill_value=0,
        )

        for i in tqdm.tqdm(
            range(len(block_access_logs)), desc="copy", dynamic_ncols=True, leave=False
        ):
            log = block_access_logs[i]
            score = block_access_scores[i]
            count = block_access_counts[i]
            block_access_log[:, i : i + 1, : log.shape[-1]] = log
            block_access_score[:, i : i + 1, : score.shape[-1]] = score
            block_access_count[:, i : i + 1] = count

        args.block_size_q = refresh_interval
        args.block_size_q = args.block_size_q // args.block_size_k

        TDST = q.shape[1] // args.block_size_k
        TSRC = k.shape[1] // args.block_size_k

        KV_HEAD_REPEAT = HEAD // HEAD_KV
        # KV_HEAD_REPEAT = 4
        # KV_HEAD_REPEAT = H

        del metadata, q, k, v, out, cos, sin, B
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        block_access_map = access_log_to_dense(
            block_access_log.cpu().numpy(),
            block_access_count.cpu().numpy(),
            TSRC,
            KV_HEAD_REPEAT,
        )
        block_access_score_map = access_score_log_to_dense(
            block_access_log.cpu().numpy(),
            block_access_count.cpu().numpy(),
            TSRC,
            KV_HEAD_REPEAT,
            block_access_score.cpu().numpy(),
        )
        block_access_mask = np.clip(block_access_map, 0, 1)

        def render_recalls():
            recalls = {}
            B, BDST, TSRC = block_access_mask.shape
            mask = block_access_mask.reshape(
                B // KV_HEAD_REPEAT, KV_HEAD_REPEAT, BDST, TSRC
            )
            mask = torch.tensor(
                np.clip(np.sum(mask, axis=1), 0, 1), device=0, dtype=torch.int32
            )
            for i in tqdm.tqdm(
                range(args.mask_k // args.block_size_q + 1, BDST),
                dynamic_ncols=True,
                desc="recalls",
                leave=False,
            ):
                for j in range(i + 1, BDST):
                    if (random.random() > (20 / (BDST - i))) and (j - i) > 16:
                        continue
                    pred = mask[:, i, : j * args.block_size_q]
                    target = mask[:, j, : j * args.block_size_q]
                    match = ((pred == target).to(torch.int32) * target).to(torch.int32)
                    num_match = torch.sum(match)
                    num_target = torch.sum(target)
                    points = recalls.get(j - i, [])
                    points.append(
                        (num_match / (num_target + 1e-20) * 100).to(
                            "cpu", non_blocking=True
                        )
                    )
                    recalls[j - i] = points
            del mask
            data = list(
                map(
                    lambda x: list(map(lambda y: y.item(), x[1])),
                    sorted(recalls.items(), key=lambda z: z[0]),
                )
            )
            means = np.array([np.mean(d) for d in data])
            stds = np.array([np.std(d) for d in data])
            xs = np.array(list(recalls.keys()))
            xs.sort()
            # print(xs, means, stds, xs.shape, means.shape, stds.shape)
            plt.clf()
            plt.fill_between(
                xs, means - stds, means + stds, alpha=0.3, facecolor="green"
            )
            plt.plot(xs, means, color="green")
            plt.xlabel(f"Decode Step Distance")
            plt.ylabel("Key Access Pattern Recall (%)")
            plt.xlim(1, 128)
            plt.ylim(50, 100)
            plt.xscale("log", base=2)
            plt.grid()
            path = os.path.join(self.exp_result_root, "access_recalls")
            plt.savefig(path + ".png", dpi=300, bbox_inches="tight")
            plt.savefig(path + ".pdf", dpi=300, bbox_inches="tight")
            print(f"saved {path}.png")

        # render_recalls()

        def render_access_map_fullres():
            root = os.path.join(self.visualization_root, "access_map")
            os.makedirs(root, exist_ok=True)
            for i in range(block_access_map.shape[0]):
                path = os.path.join(root, f"access_map_fullres_{i}.png")
                cv2.imwrite(path, (block_access_map[i] * 255).astype(np.uint8))
                print(f"saved {path}")

        # render_access_map_fullres()

        def render_access_score_map_fullres():
            t_min = -10
            t_max = 10
            t = (block_access_score_map - t_min) / (t_max - t_min)
            t = np.clip(t, 0, 1)
            root = os.path.join(self.visualization_root, "access_score_map")
            os.makedirs(root, exist_ok=True)
            for i in range(t.shape[0]):
                path = os.path.join(root, f"access_score_map_fullres_{i}.png")
                cv2.imwrite(path, (t[i] * 255).astype(np.uint8))
                print(f"saved {path}")

        # render_access_score_map_fullres()

        # plot key access map
        def render_access_map():
            root = os.path.join(self.visualization_root, "access_map_reduced")
            os.makedirs(root, exist_ok=True)

            img = block_access_map[0]
            img = img_reduce(img, 1, args.block_size_q)
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.colorbar()
            plt.title(
                f"Average Access Count\\(T={TSRC}, bq={args.block_size_q}, bk={args.block_size_k})"
            )
            plt.tight_layout()
            path = os.path.join(root, "access.png")
            plt.savefig(path, dpi=96, bbox_inches="tight")
            print(f"saved {path}")

        # render_access_map()

        def plot_stats(
            name,
            loaded_key_mask,
        ):
            root = os.path.join(self.visualization_root, "stats_" + name)
            os.makedirs(root, exist_ok=True)

            print("-" * 20, name, "-" * 20)
            # calc pre-fetchs
            prefetched_key_mask = loaded_key_mask[:, 1:, :] - loaded_key_mask[:, :-1, :]
            prefetched_key_mask = prefetched_key_mask - block_access_mask[:, :-1, :]
            prefetched_key_mask = np.clip(prefetched_key_mask, 0, 1)

            # calc misses
            missed_key_mask = np.clip(
                block_access_mask[:, 1:, :] - loaded_key_mask[:, 1:, :], 0, 1
            )

            cv2.imwrite(
                os.path.join(root, "loaded.png"), loaded_key_mask[0, 1:, :] * 255
            )
            cv2.imwrite(
                os.path.join(root, "prefetched.png"), prefetched_key_mask[0] * 255
            )
            cv2.imwrite(os.path.join(root, "missed.png"), missed_key_mask[0] * 255)
            cv2.imwrite(
                os.path.join(root, "accessed.png"), block_access_mask[0, 1:, :] * 255
            )

            # 0 (black): not loaded
            # 1 (white): loaded but not used
            # 2 (green): cache hit
            # 3 (red): missed
            load_map = cache_map = loaded_key_mask[0, 1:, :]
            access_map = block_access_map[0, 1:, :]
            cache_map = np.where(load_map * access_map, 2, cache_map)
            cache_map = np.where((1 - load_map) * access_map, 3, cache_map)
            colormap = np.array(
                [
                    [0, 0, 0],
                    [255, 255, 255],
                    [0, 255, 0],
                    [0, 0, 255],
                ],
                dtype=np.int64,
            )
            H, W = cache_map.shape
            cache_image = np.take(
                colormap.reshape(1, 1, 4, 3), cache_map.reshape(H, W, 1, 1), axis=-2
            )
            cache_image = np.reshape(cache_image, (H, W, 3))
            path = os.path.join(root, f"combined.png")
            cv2.imwrite(path, cache_image)
            print("saved", path, cache_image.shape)

            accessed_key_counts = block_access_mask[:, 1:, :].sum(axis=-1)
            loaded_key_counts = loaded_key_mask[:, 1:, :].sum(axis=-1)
            fetched_key_counts = prefetched_key_mask.sum(axis=-1)
            missed_key_counts = missed_key_mask.sum(axis=-1)
            xs = np.arange(args.block_size_q, TDST, args.block_size_q)

            plt.figure(figsize=(8, 12))
            plt.plot(xs, loaded_key_counts.T.mean(axis=-1), color="gray")
            plt.plot(xs, fetched_key_counts.T.mean(axis=-1), color="green")
            plt.plot(xs, missed_key_counts.T.mean(axis=-1), color="red")
            plt.plot(
                xs,
                fetched_key_counts.T.mean(axis=-1) + missed_key_counts.T.mean(axis=-1),
                color="orange",
            )
            plt.plot(xs, accessed_key_counts.T.mean(axis=-1), color="blue")
            plt.axhline(TSRC / args.block_stride_k, color="darkgray")
            plt.grid()
            filename = os.path.join(root, "stats")
            path = f"{filename}.png"
            plt.savefig(path, dpi=96, bbox_inches="tight")
            plt.savefig(f"{filename}.pdf", dpi=96, bbox_inches="tight")
            print(f"saved {path}")

            accessed_count = accessed_key_counts.T[-1].mean()
            missed_count = missed_key_counts.T[-1].mean()
            cache_hit_ratio = 1 - missed_count / accessed_count
            print(f"cache hit ratio: {cache_hit_ratio * 100:.4f}")

            n_layer = 32
            n_kv_hid = HID
            n_kv_head = HEAD / KV_HEAD_REPEAT
            PCIE_GB_PER_SEC = 64

            est_cache_budget = loaded_key_counts.T[-1].mean() * args.block_size_k
            oracle_cache_budget = accessed_key_counts.T[-1].mean() * args.block_size_k
            print(
                f"estimated cache size: {est_cache_budget} "
                f"({est_cache_budget * n_kv_head * n_layer * 256 / (1024 * 1024 * 1024):.3f} GB), "
                f"oracle cache size: {oracle_cache_budget}, "
                f"relative size: {est_cache_budget / oracle_cache_budget:.2f}, "
                f"sparsity: {(1 - est_cache_budget / (block_access_map.shape[-1] * args.block_size_k)) * 100:.2f} %"
            )

            fetched_count = fetched_key_counts.T[-1].mean() * args.block_size_k
            fetched_mb = fetched_count * n_layer * n_kv_head * n_kv_hid / (1024 * 1024)
            print(
                f"fetched tokens: {fetched_count:.1f}, {fetched_mb:.4f} MB, took {fetched_mb / PCIE_GB_PER_SEC:.2f} ms (bsz=1) / {fetched_mb / PCIE_GB_PER_SEC * 32:.2f} ms (bsz=32) in PCIe 4.0"
            )

            missed_count = missed_key_counts.T[-1].mean() * args.block_size_k
            missed_mb = missed_count * n_layer * n_kv_head * n_kv_hid / (1024 * 1024)
            print(
                f"missed tokens: {missed_count:.1f}, {missed_mb:.4f} MB, took {missed_mb / PCIE_GB_PER_SEC:.2f} ms (bsz=1) / {missed_mb / PCIE_GB_PER_SEC * 32:.2f} ms (bsz=32) in PCIe 4.0"
            )

            return {
                "hit ratio": cache_hit_ratio,
                "relative size": est_cache_budget / oracle_cache_budget,
                "sparsity": 1
                - est_cache_budget / (block_access_map.shape[-1] * args.block_size_k),
                "prefetch (ms, bsz=32)": fetched_mb / PCIE_GB_PER_SEC * 32,
                "missed (ms, bsz=32)": missed_mb / PCIE_GB_PER_SEC * 32,
            }

        def render_lru_hot_prefetch_unified(lru_budget_log_scale=2):
            loaded_key_mask, cache_type_map, t, t2 = perform_lru_hot_prefetch_unified(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
                args.block_size_q,
            )

            print((t >= 0).astype(np.int32).sum(axis=-1), t2)

            name = f"lru_hot_prefetch_unified_{lru_budget_log_scale}"
            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            x = plot_stats(name, loaded_key_mask)

            root = os.path.join(self.visualization_root, "prefetch")
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, f"{name}_cache_type.png")
            cv2.imwrite(path, cache_type_map[0])
            print("saved", path)

            return x

        def render_lru_hot_prefetch(lru_budget_log_scale=2):
            loaded_key_mask, cache_type_map = perform_lru_hot_prefetch(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
                args.block_size_q,
            )

            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            name = f"lru_hot_prefetch_{lru_budget_log_scale}"
            x = plot_stats(name, loaded_key_mask)

            root = os.path.join(self.visualization_root, "prefetch")
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, f"{name}_cache_type.png")
            cv2.imwrite(path, cache_type_map[0])
            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            print("saved", path)

            return x

        def render_lru_hot(lru_budget_log_scale=2):
            loaded_key_mask = perform_lru_hot(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
            )
            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            return plot_stats(f"lru_hot_{lru_budget_log_scale}", loaded_key_mask)

        def render_lru(lru_budget_log_scale=2):
            loaded_key_mask = perform_lru(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
            )
            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            return plot_stats(f"lru_{lru_budget_log_scale}", loaded_key_mask)

        def render_gd_score(lru_budget_log_scale=2, temperature=10):
            scores = block_access_score.cpu().numpy()
            probs = (
                torch.softmax((block_access_score / temperature), dim=-1).cpu().numpy()
            )
            loaded_key_mask, loaded_key_probs_map = perform_gd_score(
                block_access_map,
                block_access_log.cpu().numpy(),
                scores,
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
                temperature,
            )

            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            return plot_stats(f"lru_gd_score_{lru_budget_log_scale}", loaded_key_mask)

        def render_lru_hot_score(lru_budget_log_scale=2):
            loaded_key_mask = perform_lru_hot_score(
                block_access_map,
                block_access_log.cpu().numpy(),
                torch.softmax(block_access_score, dim=-1).cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
            )
            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            return plot_stats(f"lru_hot_score_{lru_budget_log_scale}", loaded_key_mask)

        def render_lru_score(lru_budget_log_scale=2):
            loaded_key_mask = perform_lru_score(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_score.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
            )
            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            return plot_stats(f"lru_score_{lru_budget_log_scale}", loaded_key_mask)

        def render_lru_k(lru_budget_log_scale=2, k=4):
            loaded_key_mask = perform_lru_k(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
                k,
            )
            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            return plot_stats(f"lru_{k}_{lru_budget_log_scale}", loaded_key_mask)

        def render_lru_tie_break_lre(lru_budget_log_scale=2):
            loaded_key_mask = perform_lru_tie_break_lre(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
            )
            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            return plot_stats(
                f"lru_tie_break_lre_{lru_budget_log_scale}", loaded_key_mask
            )

        def render_lru_tie_break_lfu(lru_budget_log_scale=2):
            loaded_key_mask = perform_lru_tie_break_lfu(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
            )
            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            return plot_stats(
                f"lru_tie_break_lfu_{lru_budget_log_scale}", loaded_key_mask
            )

        def render_lfu_timestep_aware(lru_budget_log_scale=2):
            loaded_key_map = perform_lfu_timestep_aware(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
            )
            loaded_key_mask = np.clip(loaded_key_map, 0, 1)
            return plot_stats(f"lfu_timestep_{lru_budget_log_scale}", loaded_key_mask)

        def render_lfu_decay(lru_budget_log_scale=2):
            loaded_key_mask = perform_lfu(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
                True,
            )
            loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
            return plot_stats(f"lfu_decay_{lru_budget_log_scale}", loaded_key_mask)

        def render_lfu(lru_budget_log_scale=2):
            loaded_key_map = perform_lfu(
                block_access_map,
                block_access_log.cpu().numpy(),
                block_access_count.cpu().numpy(),
                args.block_size_q,
                args.mask_k // args.block_size_k * 2,
                lru_budget_log_scale,
                KV_HEAD_REPEAT,
            )
            loaded_key_mask = np.clip(loaded_key_map, 0, 1)
            return plot_stats(f"lfu_{lru_budget_log_scale}", loaded_key_mask)

        policies = [
            ("LRU-Temperature-Prefetch-Unified", render_lru_hot_prefetch_unified),
            ("LRU-Temperature-Prefetch", render_lru_hot_prefetch),
            ("LRU-Temperature", render_lru_hot),
            ("LRU", render_lru),
            ("GD-AttentionScore", render_gd_score),
            ("LRU-Temperature-TieBreakAttetentionScore", render_lru_hot_score),
            ("LRU-TieBreakAttentionScore", render_lru_score),
            ("LRU-1", lambda s: render_lru_k(s, 1)),
            ("LRU-2", lambda s: render_lru_k(s, 2)),
            ("LRU-3", lambda s: render_lru_k(s, 3)),
            ("LRU-4", lambda s: render_lru_k(s, 4)),
            ("LRU-TieBreakLRE", render_lru_tie_break_lre),
            ("LRU-TieBreakLFU", render_lru_tie_break_lfu),
            ("LFU-TieBreakLRU", render_lfu_timestep_aware),
            ("LFU-Decay", render_lfu_decay),
            ("LFU", render_lfu),
        ]

        # def render_lru_heuristic(lru_budget_log_scale=4):
        #     B, BDST, K = block_access_log.shape
        #     b = args.sliding_window_size
        #     s = lru_budget_log_scale
        #     print('performing heuristic', lru_budget_log_scale, flush=True)
        #     loaded_key_mask = perform_lru_heuristic(
        #         block_access_map,
        #         block_access_log.cpu().numpy(),
        #         block_access_count.cpu().numpy(),
        #         lru_budget_log_scale,
        #         round((math.log2((BDST * args.block_size_q + b) / b) * b - b) * s + b),
        #         KV_HEAD_REPEAT,
        #         args.block_size_q,
        #         args.block_size_k,
        #         args.sliding_window_size,
        #     )
        #     loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        #     print('plot stats', flush=True)
        #     plot_stats(f'lru_heuristic_{lru_budget_log_scale}', loaded_key_mask)

        scale_factor = 0.5
        scales = [
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            # 2.5,
            # 3.0,
        ]

        data_points = []
        for policy_name, policy_fn in policies:
            for scale in scales:
                scale = scale * scale_factor
                result = {}
                result["policy_name"] = policy_name
                result["scale"] = scale
                result.update(policy_fn(scale))
                data_points.append(result)
        df = pd.DataFrame(data_points)
        df.to_csv(os.path.join(self.exp_result_root, "results.csv"))

    def test_main_plot(self):
        df = pd.read_csv(os.path.join(self.exp_result_root, "results.csv"))
        # df.sort_values('hit ratio', ascending=False, inplace=True)

        methods = df["policy_name"].unique().tolist()
        hit_ratios = [
            df[df["policy_name"] == method]["hit ratio"].mean() for method in methods
        ]
        methods = list(
            map(
                lambda x: x[1],
                sorted(zip(hit_ratios, methods), key=lambda x: x[0], reverse=True),
            )
        )

        plt.figure(figsize=(6, 4.5))
        xss = [
            df[df["policy_name"] == method]["relative size"].to_numpy()
            for method in methods
        ]
        xss = np.array(xss)
        xs = np.max(xss, axis=0)
        for method in methods:
            # xs = df[df['policy_name'] == method]['relative size']
            ys = df[df["policy_name"] == method]["hit ratio"]
            ys = ys * 100
            if method.startswith("LRU"):
                marker = "+"
                linestyle = "-"
            elif method.startswith("LFU"):
                marker = "x"
                linestyle = "--"
            else:
                marker = "."
                linestyle = ":"
            plt.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                label=f"{method} ({ys.mean():.1f}%)",
            )

        print(df)

        plt.grid()
        plt.xlim(1, 2.5)
        plt.ylim(80, 100)
        plt.xlabel("Relative Cache Size Compared to Oracle")
        plt.ylabel("Hit Ratio")
        plt.legend(fontsize=10, bbox_to_anchor=(1.02, 0.5), loc="center left")

        path = os.path.join(self.exp_result_root, "plot")
        plt.savefig(path + ".png", bbox_inches="tight", dpi=300)
        plt.savefig(path + ".pdf", bbox_inches="tight")
        print("saved", path + ".png")
