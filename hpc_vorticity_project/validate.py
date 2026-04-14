import sys
import numpy as np


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 validate.py ref_vorticity.raw test_vorticity.raw width height [tolerance]")
        sys.exit(1)

    ref_path = sys.argv[1]
    test_path = sys.argv[2]
    width = int(sys.argv[3])
    height = int(sys.argv[4])
    tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-5

    count = width * height
    ref = np.fromfile(ref_path, dtype=np.float32, count=count)
    test = np.fromfile(test_path, dtype=np.float32, count=count)

    if ref.size != count or test.size != count:
        print("File size mismatch. Expected", count, "float32 values.")
        sys.exit(1)

    diff = np.abs(ref - test)
    max_diff = float(diff.max())
    bad = int(np.sum(diff > tol))

    print("max abs diff:", max_diff)
    print("values above tolerance:", bad)

    if bad == 0:
        print("VALIDATION PASSED")
        sys.exit(0)
    else:
        print("VALIDATION FAILED")
        sys.exit(2)


if __name__ == "__main__":
    main()
