


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
USE_AMP = DEVICE.type == 'cuda'

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


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
    item2idx = {item: idx + 1 for idx, item in enumerate(surviving_items)}
    idx2item = {idx: item for item, idx in item2idx.items()}
    del item_counts
    del surviving_items

    candidate_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    del user_counts
    print(f"  Items surviving filters: {len(item2idx):,}")
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


def load_item_titles(meta_jsonl_path, item2idx, max_title_len=60):
    idx2title = {idx: asin for asin, idx in item2idx.items()}
    if meta_jsonl_path is None:
        print("No metadata path given -- examples will show raw ASINs.")
        return idx2title
    open_fn = gzip.open if meta_jsonl_path.endswith('.gz') else open
    n_found = 0
    try:
        with open_fn(meta_jsonl_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = data.get('parent_asin') or data.get('asin')
                title = data.get('title')
                if asin in item2idx and title:
                    idx = item2idx[asin]
                    clean_title = title.strip()
                    if len(clean_title) > max_title_len:
                        clean_title = clean_title[:max_title_len].rstrip() + "..."
                    idx2title[idx] = clean_title
                    n_found += 1
        print(f"Loaded titles for {n_found}/{len(item2idx)} items.")
    except FileNotFoundError:
        print(f"Metadata file not found at {meta_jsonl_path}.")
    return idx2title


def build_graph_from_split(user_sequences, num_items):

    num_users = len(user_sequences)

    train_seqs = []
    eval_targets = []
    edges_src = []
    edges_dst = []

    for user_idx, seq in enumerate(user_sequences):
        train_seq = seq[:-1]
        target = seq[-1]
        train_seqs.append(train_seq)
        eval_targets.append(target)

        # Node id space: user u -> id u; item i -> id num_users + i
        user_node = user_idx
        for item in train_seq:
            item_node = num_users + item

            edges_src.append(user_node)
            edges_dst.append(item_node)
            edges_src.append(item_node)
            edges_dst.append(user_node)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index, num_users, eval_targets, train_seqs


class InterpretableGATLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads


        self.W = nn.Linear(in_dim, out_dim, bias=False)

        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, patch_dict=None, layer_idx=0):

        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # W^h h for every node, split across heads.
        Wh = self.W(x).view(N, self.num_heads, self.head_dim)

        alpha_src = (Wh * self.a_src.unsqueeze(0)).sum(dim=-1)  # [N, H]
        alpha_dst = (Wh * self.a_dst.unsqueeze(0)).sum(dim=-1)  # [N, H]

        e = alpha_src[src] + alpha_dst[dst]  # [E, H]
        e = self.leaky_relu(e)

        e_max = torch.full((N, self.num_heads), float('-inf'), device=x.device, dtype=e.dtype)
        e_max = e_max.scatter_reduce(0, dst.unsqueeze(1).expand(-1, self.num_heads),
                                      e, reduce='amax', include_self=True)

        e_max = torch.where(torch.isinf(e_max), torch.zeros_like(e_max), e_max)
        e = e - e_max[dst]
        exp_e = torch.exp(e)

        denom = torch.zeros((N, self.num_heads), device=x.device, dtype=exp_e.dtype)
        denom = denom.scatter_add(0, dst.unsqueeze(1).expand(-1, self.num_heads), exp_e)
        alpha = exp_e / (denom[dst] + 1e-16)  # [E, H]
        alpha = self.attn_dropout(alpha)


        msgs = alpha.unsqueeze(-1) * Wh[src]  # [E, H, d]
        head_outputs = torch.zeros((N, self.num_heads, self.head_dim),
                                    device=x.device, dtype=msgs.dtype)
        head_outputs = head_outputs.scatter_add(
            0, dst.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim), msgs
        )


        if patch_dict is not None:
            for (p_layer, p_head), patch_value in patch_dict.items():
                if p_layer == layer_idx:
                    head_outputs[:, p_head, :] = patch_value

        out = head_outputs.reshape(N, self.out_dim)
        out = self.out_dropout(out)
        return out, head_outputs, alpha


class InterpretableGATRec(nn.Module):
    """
    Multi-layer GAT recommender.

    Users and items share a single embedding table indexed by unified
    node ids (users 0..num_users-1, items num_users..num_users+num_items).
    Score(u, i) = final_emb[u] . final_emb[num_users + i].
    """
    def __init__(self, num_users, num_items, hidden_dim=64, num_layers=2,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items + 1  # +1 for padding item 0
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.node_emb = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.gat_layers = nn.ModuleList([
            InterpretableGATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(hidden_dim)

    def encode(self, edge_index, patch_dict=None, need_head_outputs=True, need_attn=False):

        x = self.node_emb.weight  # [num_nodes, D]
        per_layer_x = [x]           # residual-stream snapshots for DLA
        all_head_outputs = []
        all_alpha = []

        for l_idx, layer in enumerate(self.gat_layers):
            attn_out, head_outputs, alpha = layer(x, edge_index, patch_dict, layer_idx=l_idx)
            # Residual connection, matching the SASRec block structure --
            # gives DLA a clean residual stream to decompose.
            x = x + attn_out
            per_layer_x.append(x)
            all_head_outputs.append(head_outputs if need_head_outputs else None)
            all_alpha.append(alpha if need_attn else None)

        final = self.ln_final(x)
        if need_attn:
            return final, per_layer_x, all_head_outputs, all_alpha
        return final, per_layer_x, all_head_outputs

    def score(self, user_ids, item_ids, final_emb):
        """
        Score(u, i) = final_emb[u] . final_emb[num_users + i]. Vectorized
        over batches of (u, i) pairs.
        """
        user_vecs = final_emb[user_ids]                       # [B, D]
        item_vecs = final_emb[self.num_users + item_ids]      # [B, D]
        return (user_vecs * item_vecs).sum(dim=-1)


class BPRDataset(Dataset):
    """
    BPR triplets: (user, positive item, negative item). Positive is a real
    training interaction; negative is a random item the user has NOT
    interacted with. BPR loss then encourages score(u, pos) > score(u, neg).
    Cheaper than full-catalog cross-entropy for large graphs.
    """
    def __init__(self, train_seqs, num_items, seed=0):
        self.user_pos_pairs = []
        self.user_seen = {}
        for user_idx, seq in enumerate(train_seqs):
            seen = set(seq)
            self.user_seen[user_idx] = seen
            for item in seq:
                self.user_pos_pairs.append((user_idx, item))
        self.num_items = num_items
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.user_pos_pairs)

    def __getitem__(self, idx):
        user, pos = self.user_pos_pairs[idx]
        seen = self.user_seen[user]
        while True:
            neg = self.rng.randint(1, self.num_items)
            if neg not in seen:
                return user, pos, neg


@torch.no_grad()
def evaluate_recall_at_k(model, edge_index, eval_targets, train_seqs,
                          num_users, num_items, k=10, patch_dict=None):

    model.eval()
    with torch.autocast(device_type='cuda', enabled=USE_AMP):
        final_emb, _, _ = model.encode(edge_index, patch_dict=patch_dict,
                                        need_head_outputs=(patch_dict is not None))
    final_emb = final_emb.float()

    # Score ALL users against ALL items in one big matmul: [U, D] @ [D, I].
    user_vecs = final_emb[:num_users]                                     # [U, D]
    item_vecs = final_emb[num_users + 1: num_users + num_items + 1]       # [I, D]
    all_scores = user_vecs @ item_vecs.T                                   # [U, I]

    # Mask items each user has already interacted with.
    for u_idx, seq in enumerate(train_seqs):
        if seq:
            seen = torch.tensor(seq, device=all_scores.device) - 1  # item ids start at 1, columns at 0
            all_scores[u_idx, seen] = float('-inf')

    topk_indices = torch.topk(all_scores, k=k, dim=-1).indices
    targets = torch.tensor(eval_targets, device=all_scores.device) - 1
    hits = (topk_indices == targets.unsqueeze(1)).any(dim=-1).float()
    return hits.mean().item()


@torch.no_grad()
def run_direct_logit_attribution(model, edge_index, eval_targets, num_users,
                                   num_items, sample_size=256, seed=0):

    model.eval()
    with torch.autocast(device_type='cuda', enabled=USE_AMP):
        _, _, all_head_outputs = model.encode(edge_index, need_head_outputs=True)

    rng = np.random.default_rng(seed)
    sample_users = rng.choice(num_users, size=min(sample_size, num_users), replace=False)
    sample_users_t = torch.tensor(sample_users, device=DEVICE, dtype=torch.long)
    sample_targets = torch.tensor([eval_targets[u] for u in sample_users],
                                    device=DEVICE, dtype=torch.long)


    target_unembed = model.node_emb.weight[model.num_users + sample_targets].float()  # [B, D]

    num_layers = model.num_layers
    num_heads = model.num_heads
    dla_matrix = np.zeros((num_layers, num_heads))

    for l in range(num_layers):
        head_outs = all_head_outputs[l].float()
        user_head_outs = head_outs[sample_users_t]  # [B, H, head_dim]
        for h in range(num_heads):
            head_vec_per_user = user_head_outs[:, h, :]  # [B, head_dim]

            start = h * model.gat_layers[l].head_dim
            end = start + model.gat_layers[l].head_dim
            target_slice = target_unembed[:, start:end]
            logit_contribution = (head_vec_per_user * target_slice).sum(dim=-1)
            dla_matrix[l, h] = logit_contribution.mean().item()

    return dla_matrix


def mine_cooccurrence_patterns(train_seqs, num_items, max_patterns=30, seed=0):

    rng = random.Random(seed)

    item_partners = defaultdict(Counter)
    for seq in train_seqs:
        seen = list(set(seq))
        for i in range(len(seen)):
            for j in range(len(seen)):
                if i != j:
                    item_partners[seen[i]][seen[j]] += 1

    patterns = []
    user_indices = list(range(len(train_seqs)))
    rng.shuffle(user_indices)

    for user_idx in user_indices:
        if len(patterns) >= max_patterns:
            break
        seq = train_seqs[user_idx]
        if len(seq) < 3:
            continue
        found = False
        for target_item in seq:
            partners = item_partners.get(target_item, {})
            candidates = [p for p in seq if p != target_item and partners.get(p, 0) >= 2]
            if not candidates:
                continue
            # Pick the strongest co-occurring partner also in the history.
            neighbor_item = max(candidates, key=lambda p: partners.get(p, 0))
            clean_history = list(seq)
            # Corrupt: replace neighbor_item with a random unrelated item.
            corrupt_item = neighbor_item
            attempts = 0
            while (corrupt_item in seq or corrupt_item == target_item) and attempts < 20:
                corrupt_item = rng.randint(1, num_items)
                attempts += 1
            corrupt_history = [corrupt_item if x == neighbor_item else x for x in seq]
            patterns.append({
                'user_idx': user_idx,
                'clean_history': clean_history,
                'corrupt_history': corrupt_history,
                'target_item': target_item,
                'neighbor_item': neighbor_item,
                'corrupt_item': corrupt_item,
            })
            found = True
            break
        if not found:
            continue

    return patterns


def rebuild_edge_index_for_user(base_edge_index, user_idx, new_history,
                                  num_users, num_items, device):

    user_node = user_idx
    src, dst = base_edge_index[0], base_edge_index[1]
    keep_mask = (src != user_node) & (dst != user_node)
    kept_src = src[keep_mask]
    kept_dst = dst[keep_mask]

    new_srcs = []
    new_dsts = []
    for item in new_history:
        item_node = num_users + item
        new_srcs.extend([user_node, item_node])
        new_dsts.extend([item_node, user_node])
    new_srcs_t = torch.tensor(new_srcs, dtype=torch.long, device=device)
    new_dsts_t = torch.tensor(new_dsts, dtype=torch.long, device=device)

    final_src = torch.cat([kept_src, new_srcs_t])
    final_dst = torch.cat([kept_dst, new_dsts_t])
    return torch.stack([final_src, final_dst], dim=0)


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


@torch.no_grad()
def run_causal_activation_patching(model, base_edge_index, patterns,
                                     num_users, num_items, min_gap=1e-3):
    

    model.eval()
    num_layers = model.num_layers
    num_heads = model.num_heads

    per_head_recoveries = [[[] for _ in range(num_heads)] for _ in range(num_layers)]
    skipped = 0

    for p in patterns:
        user_idx = p['user_idx']
        target_item = p['target_item']

        clean_ei = rebuild_edge_index_for_user(
            base_edge_index, user_idx, p['clean_history'], num_users, num_items, DEVICE
        )
        corrupt_ei = rebuild_edge_index_for_user(
            base_edge_index, user_idx, p['corrupt_history'], num_users, num_items, DEVICE
        )

        with torch.autocast(device_type='cuda', enabled=USE_AMP):
            clean_final, _, clean_heads = model.encode(clean_ei, need_head_outputs=True)
            corrupt_final, _, _ = model.encode(corrupt_ei, need_head_outputs=False)

        user_t = torch.tensor([user_idx], device=DEVICE, dtype=torch.long)
        item_t = torch.tensor([target_item], device=DEVICE, dtype=torch.long)
        clean_score = model.score(user_t, item_t, clean_final.float()).item()
        corrupt_score = model.score(user_t, item_t, corrupt_final.float()).item()

        gap = clean_score - corrupt_score
        if abs(gap) < min_gap:
            skipped += 1
            continue

        for l in range(num_layers):
            for h in range(num_heads):

                patch_dict = {(l, h): clean_heads[l][:, h, :]}
                with torch.autocast(device_type='cuda', enabled=USE_AMP):
                    patched_final, _, _ = model.encode(
                        corrupt_ei, patch_dict=patch_dict, need_head_outputs=False
                    )
                patched_score = model.score(user_t, item_t, patched_final.float()).item()
                recovery = (patched_score - corrupt_score) / (gap + 1e-8)
                per_head_recoveries[l][h].append(recovery)

    median_matrix = np.zeros((num_layers, num_heads))
    iqr_matrix = np.zeros((num_layers, num_heads))
    ci_lo_matrix = np.zeros((num_layers, num_heads))
    ci_hi_matrix = np.zeros((num_layers, num_heads))
    trimmed_mean_matrix = np.zeros((num_layers, num_heads))
    for l in range(num_layers):
        for h in range(num_heads):
            vals = np.array(per_head_recoveries[l][h])
            if len(vals) > 0:
                median_matrix[l, h] = np.median(vals)
                q75, q25 = np.percentile(vals, [75, 25])
                iqr_matrix[l, h] = q75 - q25
                lo, hi = bootstrap_median_ci(vals, n_boot=2000, seed=l * num_heads + h)
                ci_lo_matrix[l, h] = lo
                ci_hi_matrix[l, h] = hi
                trimmed_mean_matrix[l, h] = trimmed_mean(vals)

    n_used = len(patterns) - skipped
    print(f"Patching used {n_used}/{len(patterns)} mined patterns "
          f"({skipped} skipped for near-zero clean/corrupt score gap).")
    return trimmed_mean_matrix, median_matrix, iqr_matrix, ci_lo_matrix, ci_hi_matrix, n_used


@torch.no_grad()
def run_zero_ablation_eval(model, edge_index, eval_targets, train_seqs,
                             num_users, num_items, target_layer, target_head, k=10):
    
    if target_layer in range(model.num_layers) and target_head in range(model.num_heads):
        head_dim = model.gat_layers[target_layer].head_dim
        zero_patch = torch.zeros((model.num_nodes, head_dim), device=DEVICE)
        patch_dict = {(target_layer, target_head): zero_patch}
    else:
        patch_dict = None
    return evaluate_recall_at_k(model, edge_index, eval_targets, train_seqs,
                                  num_users, num_items, k=k, patch_dict=patch_dict)


@torch.no_grad()
def run_per_layer_decomposition(model, edge_index, eval_targets, num_users,
                                   sample_size=256, seed=0):

    model.eval()
    with torch.autocast(device_type='cuda', enabled=USE_AMP):
        _, per_layer_x, _ = model.encode(edge_index, need_head_outputs=False)


    rng = np.random.default_rng(seed)
    sample_users = rng.choice(num_users, size=min(sample_size, num_users), replace=False)
    sample_users_t = torch.tensor(sample_users, device=DEVICE, dtype=torch.long)
    sample_targets = torch.tensor([eval_targets[u] for u in sample_users],
                                    device=DEVICE, dtype=torch.long)
    target_unembed = model.node_emb.weight[model.num_users + sample_targets].float()

    per_layer_contrib = []
    per_layer_contrib.append(
        ((per_layer_x[0][sample_users_t].float() * target_unembed).sum(dim=-1)).mean().item()
    )
    for l in range(model.num_layers):
        write = per_layer_x[l + 1][sample_users_t].float() - per_layer_x[l][sample_users_t].float()
        contrib = (write * target_unembed).sum(dim=-1).mean().item()
        per_layer_contrib.append(contrib)

    return per_layer_contrib


@torch.no_grad()
def run_head_specialization_probe(model, edge_index, num_users):

    model.eval()
    with torch.autocast(device_type='cuda', enabled=USE_AMP):
        _, _, _, all_alpha = model.encode(edge_index, need_head_outputs=False, need_attn=True)

    src, dst = edge_index[0], edge_index[1]
    num_nodes = model.num_nodes

    degree = torch.zeros(num_nodes, device=DEVICE)
    degree = degree.scatter_add(0, dst, torch.ones_like(dst, dtype=torch.float))
    max_degree = degree.max().clamp(min=1.0)
    src_popularity = (degree[src] / max_degree).cpu().numpy()  # [E]

    dst_is_item = (dst >= num_users).float().cpu().numpy()  # [E], 1 if updating an item node

    num_layers = model.num_layers
    num_heads = model.num_heads
    popularity_scores = np.zeros((num_layers, num_heads))
    item_update_scores = np.zeros((num_layers, num_heads))

    for l in range(num_layers):
        alpha = all_alpha[l].float().cpu().numpy()  # [E, H]
        for h in range(num_heads):
            w = alpha[:, h]
            w_total = w.sum() + 1e-8
            popularity_scores[l, h] = (w * src_popularity).sum() / w_total
            item_update_scores[l, h] = (w * dst_is_item).sum() / w_total

    return popularity_scores, item_update_scores


def print_gat_head_specialization(popularity_scores, item_update_scores):
    num_layers, num_heads = popularity_scores.shape
    print("\n[Head Specialization Probe]")
    print("Per-head scores (0-1 scale):")
    print(f"{'Head':<14}{'Popularity':>12}{'Item-update':>14}")
    for l in range(num_layers):
        for h in range(num_heads):
            print(f"L{l} H{h:<11}{popularity_scores[l,h]:>12.3f}{item_update_scores[l,h]:>14.3f}")

    top_pop = np.unravel_index(np.argmax(popularity_scores), popularity_scores.shape)
    top_item = np.unravel_index(np.argmax(item_update_scores), item_update_scores.shape)
    print(f"\nMost popularity-driven head: Layer {top_pop[0]}, Head {top_pop[1]} "
          f"({popularity_scores[top_pop]:.3f}) -- may be a popularity shortcut, not real signal")
    print(f"Most item-focused head:      Layer {top_item[0]}, Head {top_item[1]} "
          f"({item_update_scores[top_item]:.3f}) -- specializes in updating item representations")



def seq_to_titles(seq, idx2title):
    return " | ".join(idx2title.get(i, f"<unknown {i}>") for i in seq if i != 0)


@torch.no_grad()
def print_example_recommendations(model, edge_index, eval_targets, train_seqs,
                                     num_users, num_items, idx2title,
                                     num_examples=5, top_k=5, seed=0):
    model.eval()
    with torch.autocast(device_type='cuda', enabled=USE_AMP):
        final_emb, _, _ = model.encode(edge_index, need_head_outputs=False)
    final_emb = final_emb.float()

    rng = np.random.default_rng(seed)
    sample_users = rng.choice(num_users, size=min(num_examples, num_users), replace=False)

    print("\n" + "=" * 70)
    print("EXAMPLE RECOMMENDATIONS (held-out users, real item titles)")
    print("=" * 70)

    user_vecs = final_emb[:num_users]
    item_vecs = final_emb[num_users + 1: num_users + num_items + 1]

    for count, u in enumerate(sample_users):
        scores = user_vecs[u] @ item_vecs.T  # [num_items]
        # Mask seen items
        seen = torch.tensor(train_seqs[u], device=scores.device) - 1
        scores[seen] = float('-inf')

        topk_scores, topk_indices = torch.topk(scores, k=top_k)
        topk_item_ids = (topk_indices + 1).cpu().tolist()  # back to 1-indexed
        target = eval_targets[u]
        target_score = (user_vecs[u] * item_vecs[target - 1]).sum().item()

        print(f"\n--- Example {count + 1} ---")
        print(f"User's training history: {seq_to_titles(train_seqs[u], idx2title)}")
        print(f"Actual held-out next:    {idx2title.get(target, f'<unknown {target}>')}  "
              f"(model's score: {target_score:.2f})")
        print(f"Top-{top_k} predicted:")
        for rank, (s, iid) in enumerate(zip(topk_scores.tolist(), topk_item_ids), start=1):
            title = idx2title.get(iid, f"<unknown {iid}>")
            marker = "  <-- correct!" if iid == target else ""
            print(f"  {rank}. {title}  (score: {s:.2f}){marker}")
        print(f"In top-{top_k}: {'YES' if target in topk_item_ids else 'no'}")

    print("=" * 70)


if __name__ == "__main__":
    DATA_PATH = '/Movies_and_TV.jsonl.gz'
    META_PATH = '/meta_Movies_and_TV.jsonl.gz'
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    NUM_HEADS = 4
    NUM_EPOCHS = 500
    WARMUP_EPOCHS = 10
    PEAK_LR = 1e-3
    WEIGHT_DECAY = 1e-5
    BPR_BATCH_SIZE = 4096
    EVAL_EVERY = 10
    PATIENCE = 3


    user_seqs, item2idx, idx2item = parse_amazon_jsonl(
        DATA_PATH, min_item_freq=5, min_user_interactions=5,
        max_users=20_000, max_items=None
    )
    idx2title = load_item_titles(META_PATH, item2idx)
    num_items = len(item2idx)

    # 2. Build graph (train-only edges) + collect held-out targets
    edge_index, num_users, eval_targets, train_seqs = build_graph_from_split(user_seqs, num_items)
    edge_index = edge_index.to(DEVICE)
    print(f"\nGraph: {num_users:,} user nodes, {num_items:,} item nodes, "
          f"{edge_index.size(1):,} directed edges.")

    # 3. Model
    model = InterpretableGATRec(
        num_users=num_users, num_items=num_items,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, dropout=0.1
    ).to(DEVICE)

    # 4. Training: BPR loss with warmup + cosine LR + early stopping
    bpr_dataset = BPRDataset(train_seqs, num_items)
    bpr_loader = DataLoader(bpr_dataset, batch_size=BPR_BATCH_SIZE, shuffle=True,
                              num_workers=2 if DEVICE.type == 'cuda' else 0,
                              pin_memory=DEVICE.type == 'cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, (NUM_EPOCHS - WARMUP_EPOCHS))
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_recall = -1.0
    best_state = None
    epochs_without_improvement = 0

    print("\n--- Training GAT-Rec ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for users, pos_items, neg_items in bpr_loader:
            users = users.to(DEVICE, non_blocking=True)
            pos_items = pos_items.to(DEVICE, non_blocking=True)
            neg_items = neg_items.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', enabled=USE_AMP):

                final_emb, _, _ = model.encode(edge_index, need_head_outputs=False)
                pos_scores = model.score(users, pos_items, final_emb)
                neg_scores = model.score(users, neg_items, final_emb)
                # BPR loss: -log sigmoid(pos - neg)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = total_loss / max(1, n_batches)

        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:
            val_recall = evaluate_recall_at_k(model, edge_index, eval_targets, train_seqs,
                                                num_users, num_items, k=10)
            print(f"Epoch {epoch+1} | BPR Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | "
                  f"Held-out Recall@10: {val_recall:.4f}")
            if val_recall > best_recall:
                best_recall = val_recall
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping: no improvement for {PATIENCE} checks. "
                          f"Best: {best_recall:.4f}")
                    break
        else:
            print(f"Epoch {epoch+1} | BPR Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    if best_state is not None:
        print(f"\nReloading best checkpoint (Recall@10 = {best_recall:.4f}) "
              f"for interpretability experiments.")
        model.load_state_dict(best_state)

    # 5. Direct Logit Attribution
    print("\n--- Running Direct Logit Attribution (DLA) ---")
    dla_matrix = run_direct_logit_attribution(
        model, edge_index, eval_targets, num_users, num_items, sample_size=256
    )
    plt.figure(figsize=(6, 4))
    plt.imshow(dla_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Direct Logit Contribution (approx., pre-final-LN)')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.title('DLA Heatmap (GAT recsys)')
    plt.savefig('movies_gat_dla_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n[DLA Numerical Analysis]")
    flat_dla = dla_matrix.flatten()
    top_indices = np.argsort(flat_dla)[::-1]
    for i in range(3):
        idx = top_indices[i]
        l, h = np.unravel_index(idx, dla_matrix.shape)
        print(f"  Rank {i+1}: Layer {l}, Head {h} | Logit Contribution: {dla_matrix[l, h]:+.4f}")

    # 6. Mine co-occurrence patterns and run causal patching
    print("\n--- Mining co-occurrence patterns from training graph ---")
    patterns = mine_cooccurrence_patterns(train_seqs, num_items, max_patterns=30)
    print(f"Found {len(patterns)} usable patterns.")
    if patterns:
        print("\nExample mined patterns (real titles):")
        for p in patterns[:3]:
            print(f"  User's clean history:   {seq_to_titles(p['clean_history'], idx2title)}")
            print(f"  User's corrupt history: {seq_to_titles(p['corrupt_history'], idx2title)}")
            print(f"  Target item:            {idx2title.get(p['target_item'], p['target_item'])}")
            print(f"  Swapped neighbor:       {idx2title.get(p['neighbor_item'], p['neighbor_item'])} "
                  f"-> {idx2title.get(p['corrupt_item'], p['corrupt_item'])}")
            print()

    print("\n--- Running Causal Activation Patching ---")
    if not patterns:
        print("No usable patterns found -- skipping.")
    else:
        pt_tmean, pt_median, pt_iqr, pt_lo, pt_hi, n_used = run_causal_activation_patching(
            model, edge_index, patterns, num_users, num_items
        )

        plt.figure(figsize=(6, 4))
        plt.imshow(pt_median, cmap='viridis', aspect='auto')
        plt.colorbar(label='Median Score Recovery Ratio')
        plt.xlabel('Head Index')
        plt.ylabel('Layer Index')
        plt.title(f'Causal Patching (median over {n_used} patterns, GAT recsys)')
        plt.savefig('movies_gat_patching_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\n[Causal Patching Numerical Analysis]")
        print(f"Top 3 heads by median recovery (95% bootstrap CI, over {n_used} patterns):")
        flat_pt = pt_median.flatten()
        top_pt = np.argsort(flat_pt)[::-1]
        for i in range(3):
            idx = top_pt[i]
            l, h = np.unravel_index(idx, pt_median.shape)
            print(f"  Rank {i+1}: Layer {l}, Head {h} | "
                  f"Median: {pt_median[l, h]:.2%} "
                  f"[95% CI: {pt_lo[l, h]:.2%}, {pt_hi[l, h]:.2%}] | "
                  f"IQR: {pt_iqr[l, h]:.2%} | Trimmed mean: {pt_tmean[l, h]:.2%}")

    # 7. Zero-ablation sweep
    print("\n--- Running Head Zero-Ablation Sweep ---")
    baseline_recall = evaluate_recall_at_k(model, edge_index, eval_targets, train_seqs,
                                             num_users, num_items, k=10)
    print(f"Baseline Recall@10: {baseline_recall:.4f}")
    for l in range(model.num_layers):
        for h in range(model.num_heads):
            ab_recall = run_zero_ablation_eval(model, edge_index, eval_targets, train_seqs,
                                                 num_users, num_items, target_layer=l, target_head=h)
            drop = baseline_recall - ab_recall
            print(f"Ablating Head (Layer {l}, Head {h}) -> Recall@10: {ab_recall:.4f} (Drop: {drop:+.4f})")

    # 8. Per-layer contribution decomposition (GAT-native)
    print("\n--- Running Per-Layer Contribution Decomposition ---")
    layer_contribs = run_per_layer_decomposition(
        model, edge_index, eval_targets, num_users, sample_size=256
    )
    print("Mean dot product of each layer's write-to-residual with target unembedding:")
    print(f"  Initial embeddings: {layer_contribs[0]:+.4f}")
    for l in range(model.num_layers):
        print(f"  Layer {l} write:     {layer_contribs[l + 1]:+.4f}")

    plt.figure(figsize=(6, 4))
    xs = ['init'] + [f'layer {l}' for l in range(model.num_layers)]
    plt.bar(xs, layer_contribs)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.ylabel('Mean logit contribution to correct target')
    plt.title('Per-layer decomposition of the residual stream (GAT recsys)')
    plt.savefig('movies_gat_per_layer_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()


    print("\n--- Running Head Specialization Probe")
    popularity_scores, item_update_scores = run_head_specialization_probe(
        model, edge_index, num_users
    )
    print_gat_head_specialization(popularity_scores, item_update_scores)


    print_example_recommendations(model, edge_index, eval_targets, train_seqs,
                                     num_users, num_items, idx2title,
                                     num_examples=5, top_k=5)
