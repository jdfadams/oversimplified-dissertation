import matplotlib.pyplot as plt
import numpy as np


def rectangle(x, y, ppu):
    x0, x1 = x
    length = int((x1 - x0) * ppu)
    x = np.linspace(x0, x1, length).reshape((1, length))
    y0, y1 = y
    length = int((y1 - y0) * ppu)
    y = np.linspace(y0, y1, length).reshape((length, 1))
    return x + y * 1j


def julia(c: complex, max_iters=50, view=rectangle((-2, 2), (-2, 2), 500)):
    z = view.copy()
    c = np.full(z.shape, c, dtype=complex)
    m = np.full(z.shape, True, dtype=bool)
    escape_time = np.zeros(z.shape, dtype=int)
    for iters in range(max_iters):
        z[m] = z[m] ** 2 + c[m]
        m[np.abs(z) > 2] = False
        escape_time[m] = iters
    return escape_time


def mandelbrot(max_iters=50, view=rectangle((-2, 0.5), (-1.5, 1.5), 500)):
    c = view
    z = np.zeros(c.shape, dtype=complex)
    m = np.full(c.shape, True, dtype=bool)
    escape_time = np.zeros(c.shape, dtype=int)
    for iters in range(max_iters):
        z[m] = z[m] ** 2 + c[m]
        m[np.abs(z) > 2] = False
        escape_time[m] = iters
    return escape_time


def draw(escape_time):
    plt.imshow(np.flipud(escape_time), cmap='magma')
    plt.axis('off')
    plt.show()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('julia')
    subparsers.add_parser('mandelbrot')
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    if args.command == 'julia':
        c = -0.167382584 - 1.041230161j  # period 12
        draw(julia(c))
    else:
        draw(mandelbrot())


if __name__ == '__main__':
    main()
