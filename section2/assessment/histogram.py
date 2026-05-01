# Add your solution here
@cuda.jit
def cuda_histogram(x, xmin, xmax, histogram_out):
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    start  = cuda.grid(1)      # each thread starts at its own unique index
    stride = cuda.gridsize(1)  # stride = total threads in grid

    for i in range(start, x.shape[0], stride):  # grid stride loop
        bin_number = np.int32((x[i] - xmin) / bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            cuda.atomic.add(histogram_out, bin_number, 1)  # race-condition safe