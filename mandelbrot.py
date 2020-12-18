# -*- coding: utf-8 -*-
"""
Iteractive plotting the Mandelbrot set

How to
======

Select the domain to zoom in by dragging a rectangle in the plot.
Double click to update the plot.

Left click to zoom out.

Up-key to increase max iters by factor of 2.

Down-key to reduce max iters by a factor of 10.

Press R to toggle rectangle selection.

Todo
----
 - Optimise the algorithm. Very slow for max_iters of 1000.

 - Fix the annoying rectangle selection that persists after zoom.

Notes
-----

Parallel impact was massive, in Linux.

Tests:
    [1] Linux, Ryzen 7 2700 8 cores, 16 threads CPU @ 3.3 GHz
    [2] Win10, Xeon E-2176M, 6 cores, 12 threads CPU @ 2.7 GHz

Naive single thread implementation in [1] took 25s.

Multithreaded implementation:
    Threads   Time [s]
               [1]     [2]
       1        26      19
       8        8.5     8.2
      16        5.2     7.4
      64        2.5     13
     128        2.4     35
     256        2.4     96**
     512        *      256***

* Crash - Too many open files.
** win32api permission denied error after closing the plot.
*** same as ** and system extremely slow when running.

@author: rarossi
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton
from multiprocessing import Pool, cpu_count, Array

# Plot resolution, in pixels in x and y directions
if len(sys.argv) >= 3:
    resolution = [int(d) for d in sys.argv[1:3]]
else:
    resolution = 1920, 1080

# Centre and range of the plot, x and y
centrex, centrey = 0, 0
rangex, rangey = 4, 2

# Initial max iterations for the Mandelbrot plotting algorithm.
# The higher the more details in the plot, and slower to calculate.
# This will be changed iteractivelly by Up- and Down-keys.
current_max_iters = 99


if sys.platform == 'linux':
    # `thread_progress` is a shared memory for all threads to report progress.
    # It is initialised with zeros upon each call to `mandel()`.
    # Progress report only for linux since in windows it is a pain and inefficient.
    global thread_progress

# `nprocs` is also shared as a constant to calculate progress.
global nprocs
if len(sys.argv) == 4:
    nprocs = int(sys.argv[3])
else:
    nprocs = 8 * cpu_count()


def onclick(event):
    """
    Handle mouse clicks:

        - Zoom in upon double-left-click.

        - Zoom out upon right click.

    """
    global plotted_domain, selected_domain

    if event.button == 1 and event.dblclick:
        update(selected_domain)

    if event.button == 3:
        x1, x2, y1, y2 = plotted_domain
        # TODO improve this zoom out logic
        selected_domain = np.array(
                (x1 - 0.5, x2 + 0.5,
                 y1 - 0.5, y2 + 0.5))
        update(selected_domain)


def line_select_callback(eclick, erelease):
    """
    Handle callbacks of RectangleSelector.

    Set the global `selected_domain` to the selected rectangle.

    `eclick` and `erelease` are the press and release events.
    """

    x1, x2 = sorted((eclick.xdata, erelease.xdata))
    y1, y2 = sorted((eclick.ydata, erelease.ydata))

    if eclick.button == MouseButton.LEFT:
        global selected_domain
        selected_domain = np.array((x1, x2, y1, y2))
        print('Selected: {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(*selected_domain), end='\r')


def toggle_selector(event):
    """
    Handle keyboard input.

    Up:
        Increase `max_iters` by factor of 2.

    Down:
        Decrease `max_iters` by factor of 10.

    Rr:
        Toggle rectangle selection.

    """
    global current_max_iters, plotted_domain
    if event.key == 'up':
        current_max_iters *= 2
        update(plotted_domain)
    if event.key == 'down':
        current_max_iters /= 10
        update(plotted_domain)
    if event.key in ['R', 'r']:
        toggle_selector.RS.set_active(not toggle_selector.RS.active)
        print('%sctivated rectangle selection.' % ('A' if toggle_selector.RS.active else 'Dea'))


def mandel_worker(arg_list):
    """
    Calculate Mandelbrot set for a part of the domain.

    This is a worker function meant for parallel processing, hence called by
    a `multiprocessing.Pool` in the function `mandel()`.

    `arg_list` must contain a tuple:
        domain, resolution, threshold, max_iters, thread_id

    Where:
        domain:
            (x1, x2, y1, y2)


        resolution:
            (resolution_x, resolution_y)

        threshold:
            Threshold value for the Mandelbrot set algorithm.

        max_iters:
            Maximum number of interactions (bail out) for the algorithm.

        thread_id:
            Id of the thread, used to report progress.

    Returns
    -------
    Grid of numbers to be plotted by `matplotlib.pyplot.imshow`.

    """

    domain, resolution, threshold, max_iters, thread_id = arg_list

    x = np.linspace(domain[0], domain[1], resolution[0])
    y = np.linspace(domain[2], domain[3], resolution[1])

    xy_count_iters = []
    for y0 in y:
        x_count_iters = []
        for x0 in x:
            x1, y1 = 0, 0
            i = 0

            while x1*x1 + y1*y1 <= threshold and i < max_iters:
                xtmp = x1*x1 - y1*y1 + x0
                y1 = 2*x1*y1 + y0
                x1 = xtmp
                i += 1
            x_count_iters.append(i)

        xy_count_iters.append(x_count_iters)

        # Show progress, Linux only
        if sys.platform == 'linux':
            global thread_progress, nprocs
            thread_progress[thread_id] = int(100 * (y0 - domain[2]) / (domain[3] - domain[2]))
            print('%3.0f%%' % (sum(thread_progress)/nprocs), end='\r')

    return np.array(xy_count_iters)


def mandel(domain=(-2.5, 1, -1, 1),  # x1, x2, y1, y2
           resolution=(1920, 1080),
           threshold=4,
           max_iters=99,
           ):
    """
    Parallel calculation of the Mandelbrot set

    Parameters
    ----------
    domain : tuple (float, float, float, float), optional
        Boundary of the domain (x1, x2, y1, y2).
        The default is (-2.5, 1, -1, 1).

    resolution : tuple (int, int), optional
        Plot resolution x and y.
        The default is (1920, 1080).

    threshold : int, optional
        Threshold value for the Mandelbrot set algorithm.
        The default is 4.

    max_iters : int, optional
        Maximum number of interactions (bail out) for the algorithm.
        The default is 99.

    Returns
    -------
    numpy.array
        Grid of numbers to be plotted by `matplotlib.pyplot.imshow`.

    """

    global nprocs

    if sys.platform == 'linux':
        global thread_progress
        thread_progress = Array('i', [0] * nprocs)

    # Divide domain and resolution y to run in parallel
    x1, x2, y1, y2 = domain
    resx, resy = resolution

    delta_resy = resy // nprocs
    remain_resy = resy % nprocs

    chunks_resy = [(i+1)*delta_resy for i in range(nprocs)]
    chunks_resy[-1] += remain_resy

    chunks = []
    pixel_start_y = 1
    for i, pixel_end_y in enumerate(chunks_resy):
        y1i = y1 + (y2 - y1) * pixel_start_y / resy
        y2i = y1 + (y2 - y1) * pixel_end_y / resy
        chunk = [(x1, x2, y1i, y2i),
                 (resx, pixel_end_y - pixel_start_y + 1),
                 threshold,
                 max_iters,
                 i,
                 ]
        chunks.append(chunk)
        pixel_start_y = pixel_end_y + 1

    pool = Pool(processes=nprocs)
    ret = pool.map(mandel_worker, chunks)
    return np.concatenate(ret)


def update(domain):
    """
    Update the plot.

    Parameters
    ----------
    domain :
        See `mandel`.

    """
    t0 = time.time()

    global plotted_domain
    plotted_domain = domain
    print('Max iters: {} Domain: {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(current_max_iters, *domain))
    grid = mandel(resolution=resolution, domain=domain, max_iters=current_max_iters)
    plt.imshow(grid, cmap=plt.cm.RdBu, interpolation='bilinear', origin='lower', extent=domain)
    plt.draw()

    et = time.time() - t0
    print('Update elapsed time %.3f s' % et)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            drawtype='box', useblit=True,
            button=[1],  # react to left button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.connect('key_press_event', toggle_selector)

    selected_domain = np.array(
            (centrex - rangex, centrex + rangex,
             centrey - rangey, centrey + rangey))

    update(selected_domain)
    plt.show()
