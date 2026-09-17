import json, pandas as pd, pytest
from triparsegpu import metrics as M

def test_perfect():
    gt = [0, 0, 1, 1, 2]
    assert M.grouping_accuracy(gt, gt) == 1.0
    assert M.fga(gt, gt) == 1.0

def test_relabelled_is_still_perfect():
    # ids are arbitrary labels; only the partition matters
    assert M.grouping_accuracy([0, 0, 1, 1], [9, 9, 7, 7]) == 1.0

def test_split_group():
    # GT has one group of 4; prediction splits it -> 0 correct
    assert M.grouping_accuracy([0, 0, 0, 0], [0, 0, 1, 1]) == 0.0

def test_merged_groups():
    # GT has two groups; prediction merges them -> 0 correct
    assert M.grouping_accuracy([0, 0, 1, 1], [0, 0, 0, 0]) == 0.0

def test_partial():
    # group {0,1} correct, group {2,3} split -> 2 of 4
    assert M.grouping_accuracy([0, 0, 1, 1], [5, 5, 6, 7]) == 0.5

def test_parsing_accuracy():
    assert M.parsing_accuracy(["a <*>", "b"], ["a <*>", "b"]) == 1.0
    assert M.parsing_accuracy(["a <*>", "b"], ["a <*>", "c"]) == 0.5

@pytest.mark.parametrize("D", ["BGL", "HDFS", "Thunderbird", "OpenStack"])
def test_drain3_scores_on_real_data(D):
    """Drain3 against LogHub ground truth: sanity, not a hard threshold."""
    res = json.load(open(f"results/baseline/{D}_2k.json"))
    df = pd.read_csv(f"data/loghub/{D}/{D}_2k.log_structured.csv")
    ga = M.grouping_accuracy(df.EventId.tolist(), res["ids"])
    print(f"\n{D}: Drain3 GA = {ga:.4f}  ({len(set(res['ids']))} clusters vs {df.EventId.nunique()} GT)")
    assert 0.0 <= ga <= 1.0
