#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, const int height, const int width,
             float* image, float* tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);
void swapHalos(int rank, int nprocs, float* image, int width, int height);
void initiliaseArrays(int width, int height, float* image, float* tmp_image);
void divideInitialImage(float*image, int nx, int ny, int nprocs, int height, int width, int remainder,int rank);
void collectImage(int rank, int nx, int ny, int nprocs, double toc, double tic, int remainder, int height, int width, float* image,int NumOfCols);

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
  int NumOfCols = nx / nprocs;
  int remainder = nx % nprocs;
  int start;

  if (rank < remainder) {
    NumOfCols++;
    start = rank * NumOfCols + rank;
  }
  else{
    start = rank * NumOfCols + remainder;
  }

  int end = start + NumOfCols + 1;
  int width = NumOfCols + 2;
  int height = ny + 2;

  // Allocate the image
  float* image = malloc(sizeof(float) * width * height);
  float* tmp_image = malloc(sizeof(float) * width * height);

  initiliaseArrays(width, height, image, tmp_image);

  divideInitialImage(image, nx, ny, nprocs, height, width, remainder,rank);

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    stencil(nx, ny, height, width, image, tmp_image);
    swapHalos(rank, nprocs, tmp_image, width, height);
    stencil(nx, ny, height, width, tmp_image, image);
    swapHalos(rank, nprocs, image, width, height);
  }
  double toc = wtime();

  collectImage(rank, nx, ny, nprocs, toc, tic, remainder, height, width, image, NumOfCols);
  MPI_Finalize();
}

void stencil(const int nx, const int ny, const int height, const int width,
             float* image, float* tmp_image)
{
  for (int j = 1; j < height - 1; ++j) {
    for (int i = 1; i < width - 1; ++i) {
      tmp_image[j + i * height] =  image[j     + i       * height] * 0.6f;
      tmp_image[j + i * height] += image[j     + (i - 1) * height] * 0.1f;
      tmp_image[j + i * height] += image[j     + (i + 1) * height] * 0.1f;
      tmp_image[j + i * height] += image[j - 1 + i       * height] * 0.1f;
      tmp_image[j + i * height] += image[j + 1 + i       * height] * 0.1f;
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0;
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

void swapHalos(int rank, int nprocs, float* image, int width, int height){
  float* haloLeftSend = malloc(sizeof(float) * height);
  float* haloLeftRecv = malloc(sizeof(float) * height);
  float* haloRightSend= malloc(sizeof(float) * height);
  float* haloRightRecv= malloc(sizeof(float) * height);
  
  for (int j = 0; j < height; ++j) {
      haloLeftSend[j] =  image[j + height];
      haloRightSend[j] =  image[j + (width-2) * height];
  }

  int RightRank = rank + 1;
  if (rank == nprocs - 1) {
    RightRank =  MPI_PROC_NULL;
  }
  else {
    RightRank = rank + 1;
  }
  
  int LeftRank;
  if (rank == 0){
    LeftRank =  MPI_PROC_NULL;
  }
  else{
    LeftRank = rank - 1;
  }
  

  MPI_Sendrecv(haloLeftSend, height, MPI_FLOAT, LeftRank, 1, haloRightRecv, height, 
      MPI_FLOAT,RightRank, MPI_ANY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(haloRightSend, height, MPI_FLOAT,RightRank, 1, haloLeftRecv, height, 
      MPI_FLOAT,LeftRank, MPI_ANY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);


  if (rank > 0){
    for (int j = 0; j < height; ++j) {
      image[j] = haloLeftRecv[j];
    }
  }

  if (rank != nprocs-1){
    for (int j = 0; j < height; ++j) {
      image[j + (width-1) * height] = haloRightRecv[j];
    }
  }
}

void initiliaseArrays(int width, int height, float* image, float* tmp_image) {
  for (int j = 0; j < height; ++j) {
    for (int i = 1; i < width; ++i) {
      tmp_image[j + i * height] = 0;
      image[j + i * height] = 0;
    }
  }
}

void divideInitialImage(float*image, int nx, int ny, int nprocs, int height, int width, int remainder,int rank){
  if (rank == 0){
    float* whole_image = malloc(sizeof(float) * (nx+2) * (ny+2));
    init_image(nx, ny, nx+2, ny+2, whole_image);

    for (int r = 1; r < nprocs; r++){
      int send_NumOfCols = (nx / nprocs) + 2;
      int send_start;
      if (r < remainder){
        send_start = (r * (send_NumOfCols-2)) + r;
        send_NumOfCols++;
      } else {
        send_start = (r * (send_NumOfCols-2)) + remainder ;
      }
      float* send_image = malloc(sizeof(float) * send_NumOfCols * height);

      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < send_NumOfCols; ++i) {
          send_image[j + i * height] =  whole_image[j     + (i+send_start)    * height];
        }
      }
      MPI_Send(send_image, (send_NumOfCols) * height, MPI_FLOAT, r, 1, MPI_COMM_WORLD);
    }

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          image[(j) + (i) * height] =  whole_image[j     + i       * height];
        }
      }

  } else {
    int recv_NumOfCols = (nx / nprocs)+2;
    if (rank < remainder){
        recv_NumOfCols++;
    }
    MPI_Recv(image, (recv_NumOfCols) * height, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void collectImage(int rank, int nx, int ny, int nprocs, double toc, double tic, int remainder, int height, int width, float* image, int NumOfCols){
  if (rank == 0){
    printf("------------------------------------\n");
    printf(" image:%d processes:%d runtime: %lf s\n",nx,nprocs, toc - tic);
    printf("------------------------------------\n");

    float* outputimage = malloc(sizeof(float) * (nx + 2) * (ny + 2));
    for (int r = 1; r < nprocs; r++){
      //recieve section of image from each rank and put into output image
      int recv_NumOfCols = nx / nprocs +2;
      int recv_start;
      if (r < remainder){
        recv_start = r * (recv_NumOfCols-2) + r+1;
        recv_NumOfCols++;
      } else {
        recv_start = r * (recv_NumOfCols-2) + remainder+1;
      }
      int recv_end = recv_start + recv_NumOfCols - 1;
      float* recv_image = malloc(sizeof(float) * (recv_NumOfCols+2) * height);
      MPI_Recv(recv_image, (recv_NumOfCols+2) * height, MPI_FLOAT, r,MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int j = 0; j < height; ++j) {
        for (int i = 1; i < recv_NumOfCols-1; ++i) {
          outputimage[j + (recv_start-1+i) * height] =  recv_image[j     + i       * height];
        }
      }
      //store in image
    }

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < NumOfCols+1; ++i) {
          outputimage[j + i * height] =  image[j     + i       * height];
        }
    }

    output_image(OUTPUT_FILE, nx, ny, nx+2, ny+2, outputimage);
  }
  else{
    // send image to rank 0
    MPI_Send(image, width*height, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
      
    
  }
}