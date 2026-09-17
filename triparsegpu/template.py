"""Stage 7: per-component token consensus -> template string."""
import cupy as cp
import numpy as np
from triparsegpu.compile import kernel

_TPB = 128


def consensus(tok_hash, line_tok_offs, line_tok_count, label, contents):
    """Returns (line_template_ids int64[nlines], templates list[str])."""
    nlines = int(label.size)

    # group lines by component: sort labels, derive segments
    comp_order = cp.argsort(label, kind="stable").astype(cp.int64)
    sl = label[comp_order]
    newc = cp.empty(nlines, cp.bool_)
    newc[0] = True
    newc[1:] = sl[1:] != sl[:-1]
    comp_start = cp.flatnonzero(newc).astype(cp.int64)
    comp_end = cp.concatenate([comp_start[1:], cp.array([nlines], cp.int64)])
    ncomp = int(comp_start.size)
    comp_id_of_line = cp.empty(nlines, cp.int64)
    comp_id_of_line[comp_order] = cp.cumsum(newc, dtype=cp.int64) - 1

    comp_len = line_tok_count[comp_order[comp_start]].astype(cp.int32)
    max_len = int(comp_len.max()) if ncomp else 1

    tpl_hash = cp.zeros(ncomp * max_len, cp.uint64)
    k = kernel("template", "consensus")
    k((ncomp,), (_TPB,), (tok_hash, line_tok_offs, comp_order,
                          comp_start, comp_end, comp_len,
                          cp.int32(max_len), tpl_hash))

    # materialise template text on the host from one representative line each
    th = tpl_hash.get().reshape(ncomp, max_len)
    lens = comp_len.get()
    reps = comp_order[comp_start].get()
    templates = []
    for c in range(ncomp):
        toks = contents[reps[c]].split()
        out = []
        for k_, t in enumerate(toks[: lens[c]]):
            if th[c, k_] != 0:
                out.append(t)
            else:
                # keep leading/trailing punctuation: "2," -> "<*>,"
                i, j = 0, len(t)
                while i < j and not (t[i].isalnum()):
                    i += 1
                while j > i and not (t[j-1].isalnum()):
                    j -= 1
                out.append(t[:i] + "<*>" + t[j:])
        templates.append(" ".join(out))

    return comp_id_of_line, templates
