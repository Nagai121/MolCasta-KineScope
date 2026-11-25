"""Summarize competing candidates using their same-frame hydrogen geometry."""
import numpy as np


def add_explicit_context(part):
    pair = part["pair"]
    explicit = part["explicit"]
    quality = explicit[:, 3]
    context = np.empty((len(pair), 8), dtype=np.float32)
    for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
        if start == end:
            continue
        own = quality[start:end]
        for axis in range(2):
            _, group = np.unique(pair[start:end, axis], return_inverse=True)
            groups = int(group.max()) + 1
            total = np.bincount(group, weights=own, minlength=groups)
            top = np.zeros(groups, dtype=np.float32)
            np.maximum.at(top, group, own)
            is_top = own == top[group]
            top_count = np.bincount(group, weights=is_top, minlength=groups)
            second = np.zeros_like(top)
            np.maximum.at(second, group[~is_top], own[~is_top])
            rival = np.where(is_top & (top_count[group] == 1), second[group], top[group])
            offset = 4 * axis
            context[start:end, offset:offset + 4] = np.column_stack((
                rival,
                np.log1p(np.maximum(total[group] - own, 0.)),
                own / np.maximum(total[group], 1e-8),
                own - rival,
            ))
    part["explicit_context"] = np.column_stack((context, explicit))
