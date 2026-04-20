import sys
import numpy as np
import matplotlib.pyplot as plt


def save_image(data, width, height, title, out_png):
    image = data.reshape((height, width))
    plt.figure(figsize=(12, 5))
    plt.imshow(image)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X index (column)")
    plt.ylabel("Y index (row)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    if len(sys.argv) < 6:
        print("Usage: python3 visualize.py vorticity.raw magnitude.raw width height output_prefix")
        sys.exit(1)

    vort_path = sys.argv[1]
    mag_path = sys.argv[2]
    width = int(sys.argv[3])
    height = int(sys.argv[4])
    prefix = sys.argv[5]

    count = width * height
    vort = np.fromfile(vort_path, dtype=np.float32, count=count)
    mag = np.fromfile(mag_path, dtype=np.float32, count=count)

    if vort.size != count or mag.size != count:
        print("Input size mismatch")
        sys.exit(1)

    save_image(vort, width, height, "Vorticity", prefix + "_vorticity.png")
    save_image(mag, width, height, "Magnitude", prefix + "_magnitude.png")
    print("Saved", prefix + "_vorticity.png", "and", prefix + "_magnitude.png")


if __name__ == "__main__":
    main()
