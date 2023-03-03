from enum import Enum, auto


class MetricType(Enum):
    init = auto()
    step = auto()
    final = auto()
    error = auto()
    model = auto()


def calculate_ap(pid2label, ranked_pids, cutoff=0.5):
    num_rel = 0
    total_precision = 0.0
    for i, pid in enumerate(ranked_pids):
        label = pid2label[pid]
        if label >= cutoff:
            num_rel += 1
            total_precision += num_rel / (i + 1.0)

    return (total_precision / num_rel) if num_rel > 0 else 0.0
