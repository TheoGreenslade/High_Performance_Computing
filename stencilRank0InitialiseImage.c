#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, const int height,
             float* image, float* tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("Rank %d \n", rank);

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Calculate section of image allocated to this process
  int NumOfRows = ny / nprocs;
  
  int remainder = ny % nprocs;

  if (rank < remainder) {
    NumOfRows++;
  }

  printf("Number of rows %d \n",NumOfRows);

  int start = rank * NumOfRows;
  int end = start + NumOfRows + 2;

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = NumOfRows + 2;

  // Allocate the image
  float* image = malloc(sizeof(float) * width * height);
  float* tmp_image = malloc(sizeof(float) * width * height);

  // Set the input image
  if (rank == 0){
    float* imageToSend = malloc(sizeof(float) * width * (ny * 2));
    init_image(nx, ny, width, height, imageToSend, tmp_image);
  }
  else {
    float* Reciev = malloc(sizeof(image) - sizeof(float)* width * (height - 1));
    MPI_Recv(image, sizeof(float) * width * height, MPI_CHAR, (rank-1),  MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //initialise tmpimage as 0s
  }

  if ((rank != nprocs-1) && (rank != 0)) {
    float* imageToSend = malloc(sizeof(image) - sizeof(float)* width * (height - 1));
    MPI_Send(imageToSend, sizeof(imageToSend), MPI_CHAR, rank+1, 1, MPI_COMM_WORLD);
  }

  // Call the stencil kernel
  MPI_Barrier(MPI_COMM_WORLD);
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    stencil(nx, ny, height, image, tmp_image);
    stencil(nx, ny, height, tmp_image, image);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double toc = wtime();

  // Output
  if (rank == 0){
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");
  }
  

  output_image(OUTPUT_FILE, nx, ny, width, height, image);
  free(image);
  free(tmp_image);

  MPI_Finalize();
}

void stencil(const int nx, const int ny, const int height,
             float* image, float* tmp_image)
{
  float tenth = 0.1;
  float threeFifths = 0.6;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      tmp_image[j + i * height] =  image[j     + i       * height] * threeFifths;
      tmp_image[j + i * height] += image[j     + (i - 1) * height] * tenth;
      tmp_image[j + i * height] += image[j     + (i + 1) * height] * tenth;
      tmp_image[j + i * height] += image[j - 1 + i       * height] * tenth;
      tmp_image[j + i * height] += image[j + 1 + i       * height] * tenth;
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0;
      tmp_image[j + i * height] = 0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255 * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}