import importlib.util
import os

from unmask.data.hydrafake import SPLITS

HERE = os.path.dirname(os.path.abspath(__file__))


def test_published_averages_are_consistent():
    """Avg of Table 1 (Tan et al.) = (ID + 18 OOD subsets) / 19 for every quoted row."""
    path = os.path.join(HERE, '..', 'tools', 'eval_hydrafake.py')
    spec = importlib.util.spec_from_file_location('eval_hydrafake', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, (idv, cm, cf, cd, avg) in mod.PAPER_TABLE1.items():
        assert len(cm) == len(cf) == len(cd) == 6
        assert abs((idv + sum(cm + cf + cd)) / 19 - avg) <= 0.151, name


def test_23_subsets():
    assert [len(v) for v in SPLITS.values()] == [5, 6, 6, 6]
