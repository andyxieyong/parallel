//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include <fstream>

__device__
bool isExterior(const uchar4* img, int pos)
{
    return img[pos].x == 255 && img[pos].y == 255 && img[pos].z == 255;
}

__global__
void calculateMask(const uchar4* img,
                   const int numRows, 
                   const int numCols,
                   int* mask)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= numCols || row >= numRows) {
        return;
    }

    int pos = numCols * row + col;
    bool exterior = isExterior(img, pos);
    if (exterior) {
        mask[pos] = 0;
        return;
    }

    int rowUp = max(0, row-1);
    int rowDown = min(numRows, row+1);
    int colLeft = max(0, col-1);
    int colRight = min(numCols, col+1);

    if (isExterior(img, numCols * rowUp + col) ||
        isExterior(img, numCols * rowDown + col) ||
        isExterior(img, numCols * row + colLeft) ||
        isExterior(img, numCols * row + colRight)) {
        mask[pos] = 1;    
    } else {
        mask[pos] = 2;
    }
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, 
                const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
    size_t numElems = numRowsSource * numColsSource;
    const int block = 16;
    const dim3 blockSize(block, block, 1);
    const dim3 gridSize((numColsSource + block - 1) / block, 
                        (numRowsSource + block - 1) / block, 
                        1);

    int* d_mask;
    uchar4* d_sourceImg;
    checkCudaErrors(cudaMalloc(&d_mask, sizeof(int) * numElems));
    checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * numElems));
    checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * numElems, cudaMemcpyHostToDevice));
    calculateMask<<<gridSize, blockSize>>>(d_sourceImg, numRowsSource, numColsSource, d_mask);

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
}
