-


import json
import gzip
import random
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'  # mixed precision only helps on GPU

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no GPU detected. In Colab: Runtime -> Change runtime type -> "
          "Hardware accelerator -> GPU (T4). All the optimizations below still "
          "work on CPU, but the biggest speedups (mixed precision, larger "
          "batches) only pay off on GPU.")


def parse_amazon_jsonl(review_jsonl_path,
                       min_item_freq=5,
                       min_user_interactions=5,
                       max_users=None,
                       max_items=None,
                       user_sample_seed=0):

    open_fn = gzip.open if review_jsonl_path.endswith('.gz') else open

    print("Pass 1: counting item and user frequencies (streaming)...")
    item_counts = Counter()
    user_counts = Counter()
    n_lines = 0

    with open_fn(review_jsonl_path, 'rt', encoding='utf-8') as f:
        for line in f:
            n_lines += 1
            if n_lines % 1_000_000 == 0:
                print(f"  ...{n_lines:,} lines scanned")
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = data.get('user_id') or data.get('reviewerID')
            item = data.get('parent_asin') or data.get('asin')
            ts = data.get('timestamp') or data.get('unixReviewTime')
            if user and item and ts is not None:
                item_counts[item] += 1
                user_counts[user] += 1

    print(f"  Total lines: {n_lines:,} | "
          f"Unique items: {len(item_counts):,} | "
          f"Unique users: {len(user_counts):,}")

    surviving_items = [item for item, c in item_counts.items() if c >= min_item_freq]
    if max_items is not None and len(surviving_items) > max_items:
        surviving_items = [item for item, _ in
                           sorted(((i, item_counts[i]) for i in surviving_items),
                                  key=lambda x: -x[1])[:max_items]]
    surviving_items_set = set(surviving_items)
    item2idx = {item: idx + 1 for idx, item in enumerate(surviving_items)}  # 0 = PAD
    idx2item = {idx: item for item, idx in item2idx.items()}
    del item_counts
    del surviving_items
    print(f"  Items surviving filters: {len(item2idx):,}")

    candidate_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    del user_counts
    print(f"  Candidate users after prefilter: {len(candidate_users):,}")

    print("Pass 2: building per-user interaction lists (streaming)...")
    user_interactions = defaultdict(list)
    n_kept = 0

    with open_fn(review_jsonl_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 1_000_000 == 0 and i > 0:
                print(f"  ...{i:,} lines processed | kept so far: {n_kept:,}")
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = data.get('user_id') or data.get('reviewerID')
            item = data.get('parent_asin') or data.get('asin')
            ts = data.get('timestamp') or data.get('unixReviewTime')
            if not (user and item and ts is not None):
                continue
            if user not in candidate_users:
                continue
            if item not in surviving_items_set:
                continue
            user_interactions[user].append((int(ts), item2idx[item]))
            n_kept += 1

    del candidate_users
    del surviving_items_set
    print(f"  Interactions kept: {n_kept:,}")

    print("Sorting per-user interactions...")
    user_sequences = []
    for user, events in user_interactions.items():
        if len(events) < max(3, min_user_interactions):
            continue
        events.sort(key=lambda x: x[0])
        seq = [item_idx for _, item_idx in events]
        user_sequences.append(seq)
    del user_interactions

    if max_users is not None and len(user_sequences) > max_users:
        rng = np.random.default_rng(user_sample_seed)
        idx = rng.choice(len(user_sequences), size=max_users, replace=False)
        user_sequences = [user_sequences[i] for i in idx]

    print(f"Data Loaded: {len(user_sequences):,} Users | {len(item2idx):,} Items")
    return user_sequences, item2idx, idx2item


def split_sequences(user_sequences):
    train_seqs, eval_inputs, eval_targets = [], [], []
    for seq in user_sequences:
        train_seqs.append(seq[:-1])
        eval_inputs.append(seq[:-1])
        eval_targets.append(seq[-1])
    return train_seqs, eval_inputs, eval_targets


class SASRecTrainDataset(Dataset):
    def __init__(self, sequences, max_len=50):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:-1]
        target = self.sequences[idx][1:]
        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            seq = [0] * pad_len + seq
            target = [0] * pad_len + target
        else:
            seq = seq[-self.max_len:]
            target = target[-self.max_len:]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class SASRecEvalDataset(Dataset):
    def __init__(self, eval_inputs, eval_targets, max_len=50):
        self.eval_inputs = eval_inputs
        self.eval_targets = eval_targets
        self.max_len = max_len

    def __len__(self):
        return len(self.eval_inputs)

    def __getitem__(self, idx):
        seq = self.eval_inputs[idx]
        target_item = self.eval_targets[idx]
        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            seq = [0] * pad_len + seq
        else:
            seq = seq[-self.max_len:]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target_item, dtype=torch.long)


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, patch_dict=None, layer_idx=0, need_head_outputs=True):
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:

            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, V)

        W_O = self.o_proj.weight.view(D, self.num_heads, self.head_dim)
        head_outputs = torch.einsum('bhtd, ehd -> bhte', context, W_O)

        if patch_dict is not None:
            for (p_layer, p_head), patch_value in patch_dict.items():
                if p_layer == layer_idx:
                    head_outputs[:, p_head, :, :] = patch_value

        attn_out = head_outputs.sum(dim=1)
        attn_out = self.out_dropout(attn_out)

        if not need_head_outputs:
            head_outputs = None

        return attn_out, head_outputs, attn_weights


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None, patch_dict=None, layer_idx=0, need_head_outputs=True):
        attn_out, head_outputs, attn_weights = self.attn(
            self.ln1(x), mask, patch_dict, layer_idx, need_head_outputs
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, head_outputs, attn_weights


class InterpretableSASRec(nn.Module):
    def __init__(self, item_count, max_len=50, hidden_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self.item_count = item_count
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.item_emb = nn.Embedding(item_count + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.blocks = nn.ModuleList([SASRecBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(hidden_dim)

    def forward(self, seqs, patch_dict=None, need_head_outputs=True, last_only=False):

        B, T = seqs.shape
        positions = torch.arange(T, device=seqs.device).unsqueeze(0).expand(B, -1)

        x = self.item_emb(seqs) + self.pos_emb(positions)

        causal_mask = torch.tril(torch.ones((T, T), device=seqs.device)).unsqueeze(0).unsqueeze(0)
        pad_mask = (seqs != 0).unsqueeze(1).unsqueeze(2)
        mask = causal_mask * pad_mask

        all_head_outputs = []
        all_attn_weights = []

        for l_idx, block in enumerate(self.blocks):
            x, head_outs, attn_weights = block(
                x, mask, patch_dict, layer_idx=l_idx, need_head_outputs=need_head_outputs
            )
            all_head_outputs.append(head_outs)
            all_attn_weights.append(attn_weights)

        final_state = self.ln_final(x)

        if last_only:
            final_state_for_logits = final_state[:, -1:, :]  # [B, 1, D]
        else:
            final_state_for_logits = final_state  # [B, T, D]

        logits = torch.matmul(final_state_for_logits, self.item_emb.weight.T)

        return logits, final_state, all_head_outputs, all_attn_weights


# ==========================================
# 5. DIRECT LOGIT ATTRIBUTION
# ==========================================
@torch.no_grad()
def run_direct_logit_attribution(model, sample_batch, target_item_ids):

    model.eval()
    seqs, _ = sample_batch
    seqs = seqs.to(DEVICE, non_blocking=True)
    target_item_ids = target_item_ids.to(DEVICE, non_blocking=True)

    with torch.autocast(device_type='cuda', enabled=USE_AMP):
        logits, final_state, all_head_outputs, _ = model(seqs, need_head_outputs=True, last_only=True)

    num_layers = model.num_layers
    num_heads = model.num_heads
    dla_matrix = np.zeros((num_layers, num_heads))

    target_unembed = model.item_emb.weight[target_item_ids].float()

    for l in range(num_layers):
        head_outs = all_head_outputs[l].float()
        last_pos_head_outs = head_outs[:, :, -1, :]
        for h in range(num_heads):
            head_vec = last_pos_head_outs[:, h, :]
            logit_contribution = (head_vec * target_unembed).sum(dim=-1)
            dla_matrix[l, h] = logit_contribution.mean().item()

    return dla_matrix



def mine_induction_patterns(train_seqs, item2idx_size, max_seq_len=50,
                             max_patterns=30, seed=0):
    rng = random.Random(seed)
    patterns = []

    for seq in train_seqs:
        if len(patterns) >= max_patterns:
            break
        n = len(seq)
        if n < 3:
            continue

        first_seen = {}
        for j, item in enumerate(seq):
            if item in first_seen:
                i = first_seen[item]
                if i + 1 < j:
                    b_item = seq[i + 1]
                    clean_seq = seq[:j + 1]
                    if len(clean_seq) < 2:
                        continue
                    clean_seq = clean_seq[-max_seq_len:]

                    corrupt_item = item
                    attempts = 0
                    while corrupt_item == item and attempts < 20:
                        corrupt_item = rng.randint(1, item2idx_size)
                        attempts += 1

                    corrupt_seq = clean_seq[:-1] + [corrupt_item]
                    patterns.append((clean_seq, corrupt_seq, b_item))
                    break
            else:
                first_seen[item] = j

    return patterns


def pad_seq(seq, max_len):
    pad_len = max_len - len(seq)
    if pad_len > 0:
        return [0] * pad_len + seq
    return seq[-max_len:]


def trimmed_mean(values, trim_frac=0.1):

    values = np.sort(np.asarray(values))
    n = len(values)
    if n == 0:
        return 0.0
    k = int(np.floor(n * trim_frac))
    trimmed = values[k: n - k] if n - 2 * k > 0 else values
    return float(np.mean(trimmed))


def bootstrap_median_ci(values, n_boot=2000, ci=95, seed=0):

    values = np.asarray(values)
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_medians = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_medians[i] = np.median(sample)
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    lo, hi = np.percentile(boot_medians, [lower_pct, upper_pct])
    return float(lo), float(hi)


def load_item_titles(meta_jsonl_path, item2idx, max_title_len=60, debug_sample=3):

    idx2title = {idx: asin for asin, idx in item2idx.items()}  # fallback: ASIN as title

    if meta_jsonl_path is None:
        print("No metadata path given -- example output will show raw ASINs instead of titles.")
        return idx2title

    open_fn = gzip.open if meta_jsonl_path.endswith('.gz') else open
    n_found = 0
    n_meta_lines = 0
    shown_debug = 0

    try:
        print(f"Loading item titles from {meta_jsonl_path} ...")
        with open_fn(meta_jsonl_path, 'rt', encoding='utf-8') as f:
            for line in f:
                n_meta_lines += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Show a handful of raw records early on so you can eyeball
                # whether the field names line up with what we expect.
                if shown_debug < debug_sample:
                    keys_preview = {k: data.get(k) for k in
                                     ('parent_asin', 'asin', 'title') if k in data}
                    print(f"  [debug] metadata record {n_meta_lines}: {keys_preview}")
                    shown_debug += 1

                asin = data.get('parent_asin') or data.get('asin')
                title = data.get('title')
                if asin in item2idx and title:
                    idx = item2idx[asin]
                    clean_title = title.strip()
                    if len(clean_title) > max_title_len:
                        clean_title = clean_title[:max_title_len].rstrip() + "..."
                    idx2title[idx] = clean_title
                    n_found += 1

        match_rate = n_found / max(1, len(item2idx))
        print(f"Loaded titles for {n_found}/{len(item2idx)} items "
              f"({match_rate:.1%} match rate) from {n_meta_lines:,} metadata records.")

        if match_rate < 0.5:
            print("  WARNING: match rate is low. Likely causes:")
            print("    1. META_PATH points at the wrong file (e.g. the reviews file "
                  "instead of the metadata file, or a different category).")
            print("    2. This metadata file uses a different id field than "
                  "'parent_asin'/'asin' -- check the [debug] lines printed above.")
            print("    3. DATA_PATH and META_PATH are for different product "
                  "categories, so ASINs simply don't overlap.")
    except FileNotFoundError:
        print(f"Metadata file not found at {meta_jsonl_path} -- "
              f"example output will show raw ASINs instead of titles.")

    return idx2title


def seq_to_titles(seq, idx2title, drop_padding=True):
    """Convert a list of item indices into a readable ' -> '-joined title string."""
    items = [i for i in seq if i != 0] if drop_padding else seq
    return " -> ".join(idx2title.get(i, f"<unknown item {i}>") for i in items)


@torch.no_grad()
def print_example_recommendations(model, eval_dataset, idx2title, num_examples=5, top_k=5, seed=0):

    model.eval()
    rng = np.random.default_rng(seed)
    n = len(eval_dataset)
    sample_indices = rng.choice(n, size=min(num_examples, n), replace=False)

    print("\n" + "=" * 70)
    print("EXAMPLE RECOMMENDATIONS (held-out users, real item titles)")
    print("=" * 70)

    for count, idx in enumerate(sample_indices):
        seq_tensor, target_tensor = eval_dataset[idx]
        seq_batch = seq_tensor.unsqueeze(0).to(DEVICE)

        with torch.autocast(device_type='cuda', enabled=USE_AMP):
            logits, _, _, _ = model(seq_batch, need_head_outputs=False, last_only=True)
        last_logits = logits[0, 0, :].float()

        topk_scores, topk_indices = torch.topk(last_logits, k=top_k)
        true_target_idx = target_tensor.item()
        true_target_score = last_logits[true_target_idx].item()
        hit = true_target_idx in topk_indices.tolist()

        history_titles = seq_to_titles(seq_tensor.tolist(), idx2title)
        true_title = idx2title.get(true_target_idx, f"<unknown item {true_target_idx}>")

        print(f"\n--- Example {count + 1} ---")
        print(f"History:      {history_titles}")
        print(f"Actual next:  {true_title}  (model's score: {true_target_score:.2f})")
        print(f"Top-{top_k} predicted:")
        for rank, (score, item_idx) in enumerate(zip(topk_scores.tolist(), topk_indices.tolist()), start=1):
            title = idx2title.get(item_idx, f"<unknown item {item_idx}>")
            marker = "  <-- correct!" if item_idx == true_target_idx else ""
            print(f"  {rank}. {title}  (score: {score:.2f}){marker}")
        print(f"In top-{top_k}: {'YES' if hit else 'no'}")

    print("=" * 70)



@torch.no_grad()
def run_causal_activation_patching(model, patterns, max_len=50, min_gap=1e-3):

    model.eval()
    num_layers = model.num_layers
    num_heads = model.num_heads

    N = len(patterns)
    clean_batch = torch.tensor(
        [pad_seq(c, max_len) for c, _, _ in patterns], dtype=torch.long
    ).to(DEVICE)
    corrupt_batch = torch.tensor(
        [pad_seq(c, max_len) for _, c, _ in patterns], dtype=torch.long
    ).to(DEVICE)
    target_items = torch.tensor([t for _, _, t in patterns], dtype=torch.long).to(DEVICE)

    with torch.autocast(device_type='cuda', enabled=USE_AMP):
        clean_logits, _, clean_head_outs, _ = model(clean_batch, need_head_outputs=True, last_only=True)
        corrupt_logits, _, _, _ = model(corrupt_batch, need_head_outputs=False, last_only=True)

    # logits shape [N, 1, vocab] -> gather each example's target logit
    clean_target_logit = clean_logits[:, 0, :].gather(1, target_items.unsqueeze(1)).squeeze(1).float()
    corrupt_target_logit = corrupt_logits[:, 0, :].gather(1, target_items.unsqueeze(1)).squeeze(1).float()
    gap = clean_target_logit - corrupt_target_logit  # [N]

    valid_mask = gap.abs() >= min_gap
    n_skipped = (~valid_mask).sum().item()

    trimmed_mean_matrix = np.zeros((num_layers, num_heads))
    median_matrix = np.zeros((num_layers, num_heads))
    iqr_matrix = np.zeros((num_layers, num_heads))
    ci_lo_matrix = np.zeros((num_layers, num_heads))
    ci_hi_matrix = np.zeros((num_layers, num_heads))

    for l in range(num_layers):
        for h in range(num_heads):
            clean_patch = clean_head_outs[l][:, h:h + 1, :, :].squeeze(1)  # [N, T, D], batched
            patch_dict = {(l, h): clean_patch}

            with torch.autocast(device_type='cuda', enabled=USE_AMP):
                patched_logits, _, _, _ = model(
                    corrupt_batch, patch_dict=patch_dict, need_head_outputs=False, last_only=True
                )
            patched_target_logit = patched_logits[:, 0, :].gather(
                1, target_items.unsqueeze(1)
            ).squeeze(1).float()

            recovery = (patched_target_logit - corrupt_target_logit) / (gap + 1e-8)
            recovery = recovery[valid_mask].cpu().numpy()

            if len(recovery) > 0:
                # Trimmed mean (drop top/bottom 10%) instead of the plain
                # mean, which was dominated by a handful of outlier patterns
                # and produced recovery ratios in the hundreds of percent.
                trimmed_mean_matrix[l, h] = trimmed_mean(recovery, trim_frac=0.1)
                median_matrix[l, h] = np.median(recovery)
                q75, q25 = np.percentile(recovery, [75, 25])
                iqr_matrix[l, h] = q75 - q25
                ci_lo, ci_hi = bootstrap_median_ci(recovery, n_boot=2000, ci=95, seed=l * num_heads + h)
                ci_lo_matrix[l, h] = ci_lo
                ci_hi_matrix[l, h] = ci_hi

    n_used = N - n_skipped
    print(f"Patching used {n_used}/{N} mined patterns "
          f"({n_skipped} skipped for near-zero clean/corrupt logit gap).")

    return trimmed_mean_matrix, median_matrix, iqr_matrix, ci_lo_matrix, ci_hi_matrix, n_used


@torch.no_grad()
def run_head_specialization_probe(model, eval_dataloader, train_seqs, sample_batches=5):


    model.eval()
    item_freq = Counter()
    for seq in train_seqs:
        item_freq.update(seq)
    max_freq = max(item_freq.values()) if item_freq else 1
    freq_lookup = np.zeros(max(item_freq.keys()) + 1 if item_freq else 1, dtype=np.float32)
    for item_id, c in item_freq.items():
        freq_lookup[item_id] = c / max_freq

    num_layers = model.num_layers
    num_heads = model.num_heads
    recency_scores = np.zeros((num_layers, num_heads))
    repeat_scores = np.zeros((num_layers, num_heads))
    popularity_scores = np.zeros((num_layers, num_heads))
    n_batches = 0

    for batch_idx, (seqs, _) in enumerate(eval_dataloader):
        if batch_idx >= sample_batches:
            break
        seqs = seqs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=USE_AMP):
            _, _, _, all_attn_weights = model(seqs, need_head_outputs=False, last_only=False)

        B, T = seqs.shape
        seqs_np = seqs.cpu().numpy()
        pad_mask = (seqs_np != 0).astype(np.float32)  # exclude padding positions

        for l in range(num_layers):
            attn = all_attn_weights[l][:, :, -1, :].float().cpu().numpy()  # [B, H, T]
            attn = attn * pad_mask[:, None, :]  # zero out attention to padding
            for h in range(num_heads):
                w = attn[:, h, :]  # [B, T]
                w_total = w.sum() + 1e-8

                recency_weight = (np.arange(T)[None, :] + 1) / T
                recency_scores[l, h] += (w * recency_weight).sum() / w_total

                last_items = seqs_np[:, -1:]
                is_repeat = (seqs_np == last_items).astype(np.float32)
                is_repeat[:, -1] = 0
                repeat_scores[l, h] += (w * is_repeat).sum() / w_total

                pop = freq_lookup[np.clip(seqs_np, 0, len(freq_lookup) - 1)]
                popularity_scores[l, h] += (w * pop).sum() / w_total
        n_batches += 1

    recency_scores /= max(1, n_batches)
    repeat_scores /= max(1, n_batches)
    popularity_scores /= max(1, n_batches)
    return recency_scores, repeat_scores, popularity_scores


def print_head_specialization(recency_scores, repeat_scores, popularity_scores):
    num_layers, num_heads = recency_scores.shape
    print("\n[Head Specialization Probe]")
    print("Per-head scores (0-1 scale, higher = more attention mass on that proxy):")
    print(f"{'Head':<14}{'Recency':>10}{'Repeat':>10}{'Popularity':>12}")
    for l in range(num_layers):
        for h in range(num_heads):
            print(f"L{l} H{h:<11}{recency_scores[l,h]:>10.3f}"
                  f"{repeat_scores[l,h]:>10.3f}{popularity_scores[l,h]:>12.3f}")

    flat_recency = recency_scores.flatten()
    flat_repeat = repeat_scores.flatten()
    flat_pop = popularity_scores.flatten()
    top_recency = np.unravel_index(np.argmax(flat_recency), recency_scores.shape)
    top_repeat = np.unravel_index(np.argmax(flat_repeat), repeat_scores.shape)
    top_pop = np.unravel_index(np.argmax(flat_pop), popularity_scores.shape)
    print(f"\nMost recency-driven head:    Layer {top_recency[0]}, Head {top_recency[1]} "
          f"({recency_scores[top_recency]:.3f})")
    print(f"Most repeat-driven head:     Layer {top_repeat[0]}, Head {top_repeat[1]} "
          f"({repeat_scores[top_repeat]:.3f}) -- closest thing to an induction head")
    print(f"Most popularity-driven head: Layer {top_pop[0]}, Head {top_pop[1]} "
          f"({popularity_scores[top_pop]:.3f}) -- may be a popularity shortcut, not real signal")



@torch.no_grad()
def run_zero_ablation_eval(model, eval_dataloader, target_layer, target_head):
    model.eval()
    recalls = []

    for seqs, target_items in eval_dataloader:
        seqs = seqs.to(DEVICE, non_blocking=True)
        target_items = target_items.to(DEVICE, non_blocking=True)

        B, T = seqs.shape
        if target_layer in range(model.num_layers) and target_head in range(model.num_heads):
            zero_patch = torch.zeros((B, T, model.hidden_dim), device=DEVICE)
            patch_dict = {(target_layer, target_head): zero_patch}
        else:
            patch_dict = None

        with torch.autocast(device_type='cuda', enabled=USE_AMP):
            logits, _, _, _ = model(
                seqs, patch_dict=patch_dict, need_head_outputs=(patch_dict is not None), last_only=True
            )
        last_logits = logits[:, 0, :].float()  # [B, vocab] (already last position only)

        top10_items = torch.topk(last_logits, k=10, dim=-1).indices
        target_expanded = target_items.unsqueeze(-1)
        hit = (top10_items == target_expanded).any(dim=-1).float()
        recalls.extend(hit.cpu().tolist())

    return np.mean(recalls)



if __name__ == "__main__":
    MAX_LEN = 50
    TRAIN_BATCH_SIZE = 128
    EVAL_BATCH_SIZE = 512
    NUM_WORKERS = 2 if DEVICE.type == 'cuda' else 0
    PIN_MEMORY = DEVICE.type == 'cuda'


    NUM_EPOCHS = 100
    WARMUP_EPOCHS = 10
    PEAK_LR = 0.001

    # 1. Load data
    DATA_PATH = '/Movies_and_TV.jsonl.gz'
    META_PATH = '/meta_Movies_and_TV.jsonl.gz'
    user_seqs, item2idx, idx2item = parse_amazon_jsonl(
        DATA_PATH, min_item_freq=5, min_user_interactions=5,
        max_users=20_000, max_items=None
    )
    idx2title = load_item_titles(META_PATH, item2idx)

    # 2. Leave-one-out split
    train_seqs, eval_inputs, eval_targets = split_sequences(user_seqs)

    train_dataset = SASRecTrainDataset(train_seqs, max_len=MAX_LEN)
    train_dataloader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    eval_dataset = SASRecEvalDataset(eval_inputs, eval_targets, max_len=MAX_LEN)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # 3. Model
    model = InterpretableSASRec(
        item_count=len(item2idx),
        max_len=MAX_LEN,
        hidden_dim=64,
        num_layers=2,
        num_heads=4
    ).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=PEAK_LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)


    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, (NUM_EPOCHS - WARMUP_EPOCHS))
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    EVAL_EVERY = 10
    PATIENCE = 3  # stop if no improvement for this many eval checks in a row
    best_recall = -1.0
    best_state = None
    epochs_without_improvement = 0

    print("\n--- Training SASRec ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for seqs, targets in train_dataloader:
            seqs = seqs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', enabled=USE_AMP):
                logits, _, _, _ = model(seqs, need_head_outputs=False, last_only=False)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = total_loss / len(train_dataloader)

        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:
            val_recall = run_zero_ablation_eval(model, eval_dataloader, target_layer=-1, target_head=-1)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | "
                  f"Held-out Recall@10: {val_recall:.4f}")
            if val_recall > best_recall:
                best_recall = val_recall
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping: no improvement in held-out Recall@10 "
                          f"for {PATIENCE} checks in a row. Best: {best_recall:.4f}")
                    break
        else:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    if best_state is not None:
        print(f"\nReloading best checkpoint (held-out Recall@10 = {best_recall:.4f}) "
              f"for interpretability experiments.")
        model.load_state_dict(best_state)

    # 5. Direct Logit Attribution (on held-out eval batch)
    print("\n--- Running Direct Logit Attribution (DLA) ---")
    sample_seqs, sample_targets = next(iter(eval_dataloader))
    dla_matrix = run_direct_logit_attribution(model, (sample_seqs, sample_targets), sample_targets)

    plt.figure(figsize=(6, 4))
    plt.imshow(dla_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Direct Logit Contribution (approx., pre-final-LN)')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.title('Direct Logit Attribution Heatmap (Rec-Induction Heads)')
    plt.savefig('movies_dla_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n[DLA Numerical Analysis]")
    print("Top 3 Heads directly driving the target prediction:")
    flat_dla = dla_matrix.flatten()
    top_indices = np.argsort(flat_dla)[::-1]
    for i in range(3):
        idx = top_indices[i]
        l, h = np.unravel_index(idx, dla_matrix.shape)
        score = dla_matrix[l, h]
        print(f"  Rank {i+1}: Layer {l}, Head {h} | Logit Contribution: {score:+.4f}")

    # 6. Causal Activation Patching, batched over mined patterns
    print("\n--- Mining real induction-style patterns from training data ---")
    patterns = mine_induction_patterns(train_seqs, item2idx_size=len(item2idx),
                                        max_seq_len=MAX_LEN, max_patterns=30)
    print(f"Found {len(patterns)} usable A...B...A patterns.")

    if len(patterns) > 0:
        print("\nExample mined patterns (real titles):")
        for clean_list, corrupt_list, target_idx in patterns[:3]:
            print(f"  Clean:   {seq_to_titles(clean_list, idx2title)}")
            print(f"  Corrupt: {seq_to_titles(corrupt_list, idx2title)}")
            print(f"  Target (item that should be predicted): "
                  f"{idx2title.get(target_idx, f'<unknown item {target_idx}>')}")
            print()

    print("\n--- Running Causal Activation Patching ---")
    if len(patterns) == 0:
        print("No induction-style repeat patterns found in this dataset's "
              "training sequences -- skipping patching.")
    else:
        patch_trimmed_mean, patch_median, patch_iqr, patch_ci_lo, patch_ci_hi, n_used = \
            run_causal_activation_patching(model, patterns, max_len=MAX_LEN)

        plt.figure(figsize=(6, 4))
        plt.imshow(patch_median, cmap='viridis', aspect='auto')
        plt.colorbar(label='Median Logit Recovery Ratio')
        plt.xlabel('Head Index')
        plt.ylabel('Layer Index')
        plt.title(f'Causal Activation Patching Recovery (median over {n_used} patterns)')
        plt.savefig('movies_patching_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\n[Causal Patching Numerical Analysis]")
        print(f"Top 3 Heads by median recovery, with 95% bootstrap CI on the "
              f"median and a 10%-trimmed mean (over {n_used} patterns):")
        flat_patch = patch_median.flatten()
        top_patch_indices = np.argsort(flat_patch)[::-1]
        for i in range(3):
            idx = top_patch_indices[i]
            l, h = np.unravel_index(idx, patch_median.shape)
            med = patch_median[l, h]
            iqr = patch_iqr[l, h]
            ci_lo = patch_ci_lo[l, h]
            ci_hi = patch_ci_hi[l, h]
            tmean = patch_trimmed_mean[l, h]
            print(f"  Rank {i+1}: Layer {l}, Head {h} | "
                  f"Median: {med:.2%} [95% CI: {ci_lo:.2%}, {ci_hi:.2%}] | "
                  f"IQR: {iqr:.2%} | Trimmed mean: {tmean:.2%}")

    # 7. Zero-Ablation Head Knockout Sweep (on held-out data)
    print("\n--- Running Head Zero-Ablation Sweep (held-out Recall@10) ---")
    baseline_recall = run_zero_ablation_eval(model, eval_dataloader, target_layer=-1, target_head=-1)
    print(f"Baseline Recall@10: {baseline_recall:.4f}")

    for l in range(model.num_layers):
        for h in range(model.num_heads):
            ablated_recall = run_zero_ablation_eval(model, eval_dataloader, target_layer=l, target_head=h)
            drop = baseline_recall - ablated_recall
            print(f"Ablating Head (Layer {l}, Head {h}) -> Recall@10: {ablated_recall:.4f} (Drop: {drop:+.4f})")


    print("\n--- Running Head Specialization Probe ---")
    recency_scores, repeat_scores, popularity_scores = run_head_specialization_probe(
        model, eval_dataloader, train_seqs, sample_batches=5
    )
    print_head_specialization(recency_scores, repeat_scores, popularity_scores)

    # 9. Illustrative example: real user history -> real predicted titles
    print_example_recommendations(model, eval_dataset, idx2title, num_examples=5, top_k=5)