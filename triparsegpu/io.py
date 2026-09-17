"""Log loading: strip the header per the logparser log_format, return the
Content bytes as one flat buffer plus line offsets (ready for the GPU)."""
import re
import numpy as np

# log_format strings from logpai/logparser Drain/benchmark.py
FORMATS = {
    "BGL":         "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
    "HDFS":        "<Date> <Time> <Pid> <Level> <Component>: <Content>",
    "Thunderbird": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\\[<PID>\\])?: <Content>",
    "OpenStack":   "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \\[<ADDR>\\] <Content>",
}


def format_regex(fmt):
    """Convert a logparser log_format into a compiled regex with named groups.
    Same construction as logparser's generate_logformat_regex."""
    headers, parts = [], re.split(r"(<[^<>]+>)", fmt)
    out = ""
    for i, p in enumerate(parts):
        if i % 2 == 0:
            out += re.sub(" +", r"\\s+", p)
        else:
            name = p.strip("<>")
            out += f"(?P<{name}>.*?)"
            headers.append(name)
    return re.compile("^" + out + "$"), headers


def iter_contents(path, dataset, limit=None):
    """Yield the Content field of each line. Lines that don't match the
    format are yielded whole (and counted as misses)."""
    rx, _ = format_regex(FORMATS[dataset])
    misses = 0
    with open(path, errors="ignore") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.rstrip("\n")
            m = rx.match(line)
            if m:
                yield m.group("Content")
            else:
                misses += 1
                yield line
    iter_contents.misses = misses


def load_contents(path, dataset, limit=None):
    """Return (contents list, miss count)."""
    rx, _ = format_regex(FORMATS[dataset])
    contents, misses = [], 0
    with open(path, errors="ignore") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.rstrip("\n")
            m = rx.match(line)
            if m:
                contents.append(m.group("Content"))
            else:
                contents.append(line)
                misses += 1
    return contents, misses


def to_buffer(contents):
    """Flatten to a single uint8 buffer + int64 line offsets, for H2D copy."""
    buf = bytearray()
    offs = [0]
    for c in contents:
        buf += c.encode("utf-8", "ignore") + b"\n"
        offs.append(len(buf))
    return np.frombuffer(bytes(buf), np.uint8), np.asarray(offs, np.int64)


# Same regexes the Drain benchmark applies before parsing (logpai/logparser).
PREPROCESS = {
    # (pattern, replacement) — keep the literal prefix so templates match the
    # ground truth style: blk_123 -> blk_<*>, core.7 -> core.<*>
    "BGL":         [(r"core\.\d+", "core.<*>")],
    "HDFS":        [(r"blk_-?\d+", "blk_<*>"),
                    (r"(\d+\.){3}\d+:\d+", "<*>:<*>"),
                    (r"(\d+\.){3}\d+", "<*>")],
    "Thunderbird": [(r"(\d+\.){3}\d+", "<*>")],
    "OpenStack":   [(r"((\d+\.){3}\d+,?)+", "<*>"), (r"/.+?\s", "<*> "), (r"\d+", "<*>")],
}


def preprocess(contents, dataset):
    """Substitute the dataset's parameter regexes, keeping literal prefixes
    so template text matches the benchmark's style."""
    pats = [(re.compile(p), r) for p, r in PREPROCESS.get(dataset, [])]
    out = []
    for line in contents:
        for p, r in pats:
            line = p.sub(r, line)
        out.append(line)
    return out
