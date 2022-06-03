import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

CANTOR = -0.4+0.6j
P3 = -1.754877666246693+0j
P4 = -0.156520166833755+1.032247108922832j
P12 = -0.167349208205021+1.041178661132973j
P16 = -0.152906328119694+1.039662099471381j
RABBIT = -0.1226+0.7449j

JULIA_PARAMS = {
    'rabbit': (RABBIT, 50),
    'p3': (P3, 20),
    'p4': (P4, 50),
    'p12': (P12, 50),
    'p12_zoom': (P12, 100, 0, 0.25, 800),
    'p16': (P16, 50),
    'cantor': (CANTOR, 250),
}

MANDELBROT_PARAMS = {
    'default': (-0.7+0j, 1.5, 800, 150),
    'p3': (P3, 0.2, 800, 1000),
    'p4': (P4, 0.175, 800, 1000),
    'p12': (P12, 0.0025, 800, 1000),
}


def square(center, e, l):
    x, y = center.real, center.imag
    x0, x1 = x - e, x + e
    x = np.linspace(x0, x1, l).reshape((1, l))
    y0, y1 = y - e, y + e
    y = np.linspace(y0, y1, l).reshape((l, 1))
    return x + y * 1j


def julia(c: complex, max_iters=50, view=square(0, 2, 800)):
    z = view.copy()
    c = np.full(z.shape, c, dtype=complex)
    m = np.full(z.shape, True, dtype=bool)
    escape_time = np.zeros(z.shape, dtype=int)
    for iters in range(max_iters):
        z[m] = z[m] ** 2 + c[m]
        m[np.abs(z) > 2] = False
        escape_time[m] = iters
    return escape_time


def mandelbrot(max_iters=150, view=square(-0.7, 1.5, 800)):
    c = view
    z = np.zeros(c.shape, dtype=complex)
    m = np.full(c.shape, True, dtype=bool)
    escape_time = np.zeros(c.shape, dtype=int)
    for iters in range(max_iters):
        z[m] = z[m] ** 2 + c[m]
        m[np.abs(z) > 2] = False
        escape_time[m] = iters
    return escape_time


def draw(escape_time, *, cmap='magma', fname=None):
    escape_time = ma.log(escape_time).filled(0)
    escape_time_flipped = np.flipud(escape_time)
    plt.imshow(escape_time_flipped, cmap=cmap)
    plt.axis('off')
    if fname:
        plt.imsave(fname, escape_time_flipped, cmap=cmap)
    else:
        plt.show()


def draw_julia(name):
    c, n, *rect_params = JULIA_PARAMS[name]
    if rect_params:
        j = julia(c, max_iters=n, view=square(*rect_params))
    else:
        j = julia(c, max_iters=n)
    draw(j, fname=f'k_{name}.png')


def draw_mandelbrot(name):
    center, e, d, n = MANDELBROT_PARAMS[name]
    m = mandelbrot(max_iters=n, view=square(center, e, d))
    draw(m, fname=f'm_{name}.png')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    sp = subparsers.add_parser('julia')
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument('-c', type=complex)
    g.add_argument('-w', choices=JULIA_PARAMS.keys())
    sp = subparsers.add_parser('mandelbrot')
    sp.add_argument('-w', choices=MANDELBROT_PARAMS.keys(), default='default')
    subparsers.add_parser('all')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == 'julia':
        if args.c:
            draw(julia(args.c), fname=f'k_{args.c}.png')
        elif args.w:
            draw_julia(args.w)
    elif args.command == 'mandelbrot':
        draw_mandelbrot(args.w)
    elif args.command == 'all':
        for name in JULIA_PARAMS:
            draw_julia(name)
        for name in MANDELBROT_PARAMS:
            draw_mandelbrot(name)


if __name__ == '__main__':
    main()
