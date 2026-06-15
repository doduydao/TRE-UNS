import re
import sys
from statistics import mean


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python analyze_reasoning_log.py <log_file>")
        return 1

    log_file = sys.argv[1]
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # tqdm often writes many metric snapshots on one physical line.
    pattern = re.compile(
        r"Loss=(?P<Loss>-?\\d+\\.\\d+),\\s*"
        r"L_CE=(?P<L_CE>-?\\d+\\.\\d+),\\s*"
        r"L_KL=(?P<L_KL>-?\\d+\\.\\d+),\\s*"
        r"L_EP=(?P<L_EP>-?\\d+\\.\\d+),\\s*"
        r"E_rule_P=(?P<E_rule_P>-?\\d+\\.\\d+),\\s*"
        r"E_rule_Q=(?P<E_rule_Q>-?\\d+\\.\\d+),\\s*"
        r"VP=(?P<VP>-?\\d+\\.?\\d*),\\s*"
        r"VQ=(?P<VQ>-?\\d+\\.?\\d*)"
    )

    keys = ["Loss", "L_CE", "L_KL", "L_EP", "E_rule_P", "E_rule_Q", "VP", "VQ"]
    values = {k: [] for k in keys}

    for match in pattern.finditer(text):
        for k in keys:
            values[k].append(float(match.group(k)))

    n = len(values["Loss"])
    print(f"Parsed metric snapshots: {n}")

    if n > 0:
        for k in keys:
            arr = values[k]
            print(f"{k:>8} | mean={mean(arr):.6f} | min={min(arr):.6f} | max={max(arr):.6f}")

        eps = 1e-12
        print("\nNon-zero counts:")
        for k in ["E_rule_P", "E_rule_Q", "VP", "VQ", "L_EP"]:
            arr = values[k]
            c = sum(1 for x in arr if abs(x) > eps)
            print(f"{k:>8} != 0: {c}/{n}")

    oom_count = text.count("OutOfMemoryError")
    cuda_oom_count = text.count("CUDA out of memory")
    print(f"\nOOM lines: OutOfMemoryError={oom_count}, CUDA out of memory={cuda_oom_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
