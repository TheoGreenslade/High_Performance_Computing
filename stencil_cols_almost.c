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

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  // printf("3");

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
  printf("Initalisation: Rank: %d start:%d end:%d \n", rank, start, end);

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = NumOfCols + 2;
  int height = ny + 2;

  // Allocate the image
  float* image = malloc(sizeof(float) * width * height);
  float* tmp_image = malloc(sizeof(float) * width * height);

  // // Set the input image
  if (rank == 0){
    float* whole_image = malloc(sizeof(float) * (nx+2) * (ny+2));
    init_image(nx, ny, nx+2, ny+2, whole_image);

    for (int r = 1; r < nprocs; r++){
      int send_NumOfCols = (nx / nprocs) + 2;
      int send_start;
      if (r < remainder){
        send_NumOfCols++;
        send_start = r * send_NumOfCols + r;
      } else {
        send_start = r * send_NumOfCols + remainder;
      }
      int send_end = send_start + send_NumOfCols + 1;
      float* send_image = malloc(sizeof(float) * (send_NumOfCols+2) * height);
      printf("Send: Rank: %d start:%d end:%d \n", r, send_start, send_end);
      // printf("size of array:%d",(send_NumOfCols+2) * height);
      printf("send num of cols:%d\n",send_NumOfCols);
      for (int j = 0; j < height+1; ++j) {
        for (int i = 0; i < send_NumOfCols+1; ++i) {
          // printf("%d: r:%d in second for j:%d i:%d index:%d\n",rank,r,j,i,j + i * height);
          send_image[j + i * height] =  whole_image[(j+send_start)     + i       * height];
        }
      }
      // printf("%d: send buffer for %d:%d\n",rank,r,(send_NumOfCols+1) * height);
      MPI_Send(send_image, (send_NumOfCols+1) * height, MPI_FLOAT, r, 1, MPI_COMM_WORLD);
      // printf("Send: Rank: %d start:%d end:%d \n", r, send_start, send_end);
      printf("sent to rank:%d\n",r);
    }

    for (int j = 0; j < height+1; ++j) {
        for (int i = 0; i < width+1; ++i) {
          image[j + i * height] =  whole_image[(j+1)     + (i+1)       * height];
        }
      }
    // printf("Send: Rank: 0 start:0 end:%d \n", NumOfCols+1);

  } else {
    int recv_NumOfCols = (nx / nprocs)+2;
    if (rank < remainder){
        recv_NumOfCols++;
    }
    // printf("%d: recv buffer:%d\n",rank,(recv_NumOfCols+1) * height);
    MPI_Recv(image, (recv_NumOfCols+1) * height, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("recv by rank:%d\n",rank);
  }

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    // printf("rank:%d iteration:%d\n",rank,t);
    stencil(nx, ny, height, width, image, tmp_image);
    swapHalos(rank, nprocs, image, width, height);
    stencil(nx, ny, height, width, tmp_image, image);
    swapHalos(rank, nprocs, image, width, height);
  }
  double toc = wtime();
  printf("%d: stencil done\n",rank);

  if (rank == 0){
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");

    float* outputimage = malloc(sizeof(float) * (nx + 2) * (ny + 2));
    // printf("%d: before loop\n",rank);
    for (int r = 1; r < nprocs; r++){
      //recieve section of image from each rank and put into output image
      int recv_NumOfCols = nx / nprocs;
      int recv_start;
      if (r < remainder){
        recv_NumOfCols++;
        recv_start = r * recv_NumOfCols + r+1;
      } else {
        recv_start = r * recv_NumOfCols + remainder+1;
      }
      int recv_end = recv_start + recv_NumOfCols - 1;
      float* recv_image = malloc(sizeof(float) * recv_NumOfCols * height);
      printf("Recv: Rank: %d start:%d end:%d \n", r, recv_start, recv_end);
      MPI_Recv(recv_image, recv_NumOfCols * ny, MPI_FLOAT, r,MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("rank:%d r:%d recv start:%d recv end:%d\n",rank,r,recv_start,recv_end);
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < recv_NumOfCols; ++i) {
          outputimage[j + (recv_start+i) * height] =  recv_image[j     + i       * height];
        }
      }
      //store in image
    }

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < NumOfCols; ++i) {
          outputimage[(j+1) + (i+1) * height] =  image[j     + i       * height];
        }
    }

    output_image(OUTPUT_FILE, nx, ny, nx+2, ny+2, outputimage);
  }
  else{
    // send image to rank 0
    MPI_Send(image, NumOfCols * ny, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
  }

  // free(image);
  // free(tmp_image);

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
  float* halo1send = malloc(sizeof(float) * height);
  float* halo1recv = malloc(sizeof(float) * height);
  float* halo2send= malloc(sizeof(float) * height);
  float* halo2recv= malloc(sizeof(float) * height);

  for (int i = 0; i < height; ++i) {
      halo1send[i] =  image[i];
      halo2send[i] =  image[i + (width - 1) * height];
  }

  int nextRank;
  if (rank = nprocs - 1){
    nextRank =  MPI_PROC_NULL;
  }
  else{
    nextRank = rank + 1;
  }

  int prevRank;
  if (rank = 0){
    prevRank =  MPI_PROC_NULL;
  }
  else{
    prevRank = rank - 1;
  }


  MPI_Sendrecv(halo1send, height, MPI_FLOAT, prevRank, 1, halo2recv, height, 
      MPI_FLOAT,nextRank, MPI_ANY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(halo2send, height, MPI_FLOAT,nextRank, 1, halo1recv, height, 
      MPI_FLOAT,prevRank, MPI_ANY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int i = 0; i < height; ++i) {
      image[i] = halo1recv[i];
      image[i + (width - 1) * height] = halo2recv[i];
  }
}