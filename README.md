# High_Performance_Computing

This code implements a weighted 5-point stencil on a rectangular grid.  The
value in each cell of the grid is updated based on an average of the values in
the neighbouring North, South, East and West cells. This code was optimised to 
run in the shortest time on BlueCrystal, the University of Bristols super computer.

The grid is initialised into a checkerboard pattern, with each square of the
checkerboard being 64x64 pixels. The stencil operation reads from one grid and
writes to a temporary grid.  The stencil is run twice for every iteration, with
the final result being held in the original array.  The results are quantised to
integers in the range [0,255] and output as a binary image file that may be
viewed graphically.

The only output of each run is the runtime of the iteration loop of the program.
Initialisation and output are not timed.


