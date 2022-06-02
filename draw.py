import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

JULIA_PARAMS = {
    'rabbit': (-0.1226+0.7449j, 50),
    'p3': (-1.754877666246693+0j, 20),
    'p4': (-0.156520166833755+1.032247108922832j, 50),
    'p12': (-0.167349208205021+1.041178661132973j, 50),
    'p16': (-0.152906328119694+1.039662099471381j, 50),
    'cantor': (-0.4+0.6j, 250),
}

MANDELBROT_PARAMS = {
    'default': (-0.7+0j, 1.5, 500, 150),
    'p4': (-0.156520166833755+1.032247108922832j, 0.2, 2000, 1000),
}


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


def mandelbrot(max_iters=150, view=rectangle((-2.2, 0.8), (-1.5, 1.5), 500)):
    c = view
    z = np.zeros(c.shape, dtype=complex)
    m = np.full(c.shape, True, dtype=bool)
    escape_time = np.zeros(c.shape, dtype=int)
    for iters in range(max_iters):
        z[m] = z[m] ** 2 + c[m]
        m[np.abs(z) > 2] = False
        escape_time[m] = iters
    return escape_time


def draw(escape_time, *, cmap='magma'):
    escape_time = ma.log(escape_time).filled(0)
    plt.imshow(np.flipud(escape_time), cmap=cmap)
    plt.axis('off')
    plt.show()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    sp = subparsers.add_parser('julia')
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument('-c', type=complex)
    g.add_argument('-w', choices=JULIA_PARAMS.keys())
    sp = subparsers.add_parser('mandelbrot')
    sp.add_argument('w', nargs='?', choices=MANDELBROT_PARAMS.keys(), default='default')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == 'julia':
        if args.c:
            draw(julia(args.c))
        elif args.w:
            c, n = JULIA_PARAMS[args.w]
            draw(julia(c, max_iters=n))
    elif args.command == 'mandelbrot':
        center, e, d, n = MANDELBROT_PARAMS[args.w]
        x, y = center.real, center.imag
        draw(mandelbrot(max_iters=n, view=rectangle((x - e, x + e), (y - e, y + e), d)))


if __name__ == '__main__':
    main()
