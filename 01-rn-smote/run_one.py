import sys
import time

sys.path.insert(0, "/home/a1medessam/Desktop/papers2code/rnsmote_project/src")
import datasets  # noqa: E402
import evaluate  # noqa: E402

RESULTS_PATH = "/home/a1medessam/Desktop/papers2code/rnsmote_project/results_raw.csv"


def main(name, n_repeats, classifiers=None):
    X, y = datasets.load(name)
    t0 = time.time()
    df = evaluate.evaluate_dataset(name, X, y, n_repeats=n_repeats, n_splits=10,
                                    classifiers=classifiers)
    dt = time.time() - t0
    header = not __import__("os").path.exists(RESULTS_PATH)
    df.to_csv(RESULTS_PATH, mode="a", header=header, index=False)
    print(f"{name} (clf={classifiers or 'ALL'}, reps={n_repeats}): "
          f"{dt:.1f}s, {len(df)} rows appended -> {RESULTS_PATH}")


if __name__ == "__main__":
    name = sys.argv[1]
    n_repeats = int(sys.argv[2])
    clfs = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    main(name, n_repeats, clfs)
