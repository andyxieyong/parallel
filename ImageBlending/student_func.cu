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

__device__
bool isExterior(const uchar4* img, int pos)
{
    return img[pos].x == 255 && img[pos].y == 255 && img[pos].z == 255;
}

__global__
void calculateMask(const uchar4* img,
                   const int numCols,
                   const int numRows,
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

__global__
void separateChannels(const uchar4* img,
                      float* red, float* green, float* blue,
                      int numCols, int numRows)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= numCols || row >= numRows) {
        return;
    }

    int pos = numCols * row + col;
    red[pos] = img[pos].x;
    green[pos] = img[pos].y;
    blue[pos] = img[pos].z;
}

__device__
float getSum1Float(const float* dstImg, int* mask, float* prev, int pos)
{
    if (mask[pos] == 2) {
        return prev[pos];
    } else if (mask[pos] == 1) {
        return dstImg[pos];
    } else {
        return 0.0f;
    }
}

__global__
void jacobi(const float* srcImg, const float* dstImg, int* mask,
            float* prev, float* next,
            int numCols, int numRows)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= numCols || row >= numRows) {
        return;
    }

    int pos = numCols * row + col;
    if (mask[pos] != 2) {
        next[pos] = dstImg[pos];
        return;
    }

    int neighborUp = numCols * max(0, row-1) + col;
    int neighborDown = numCols * min(numRows, row+1) + col;    
    int neighborLeft = numCols * row + max(0, col-1);
    int neighborRight = numCols * row + min(numCols, col+1);
    
    float sum1 = 0.0f;
    sum1 += getSum1Float(dstImg, mask, prev, neighborUp);
    sum1 += getSum1Float(dstImg, mask, prev, neighborDown);
    sum1 += getSum1Float(dstImg, mask, prev, neighborLeft);
    sum1 += getSum1Float(dstImg, mask, prev, neighborRight);
    float sum2 = 0.0f;
    sum2 += srcImg[pos] - srcImg[neighborUp];
    sum2 += srcImg[pos] - srcImg[neighborDown];
    sum2 += srcImg[pos] - srcImg[neighborLeft];
    sum2 += srcImg[pos] - srcImg[neighborRight];
    float newVal = (sum1 + sum2) / 4.0f;
    next[pos] = min(255.0f, max(0.0f, newVal));
}

__global__
void combineChannels(uchar4* out, float* red, float* green, float* blue, int numCols, int numRows)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= numCols || row >= numRows) {
        return;
    }

    int pos = numCols * row + col;
    out[pos].x = red[pos];
    out[pos].y = green[pos];
    out[pos].z = blue[pos];
}

void your_blend(const uchar4* const h_sourceImg, //IN
                const size_t numRowsSource, 
                const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
    int numElems = numRowsSource * numColsSource;
    const int block = 16;
    const dim3 blockSize(block, block, 1);
    const dim3 gridSize((numColsSource + block - 1) / block, 
                        (numRowsSource + block - 1) / block, 
                        1);

    uchar4* d_sourceImg;
    checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * numElems));
    checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * numElems, cudaMemcpyHostToDevice));
    uchar4* d_destImg;
    checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4) * numElems));
    checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * numElems, cudaMemcpyHostToDevice));
    
    int* d_mask; 
    checkCudaErrors(cudaMalloc(&d_mask, sizeof(int) * numElems));        
    calculateMask<<<gridSize, blockSize>>>(d_sourceImg, numColsSource, numRowsSource, d_mask);

    float* d_srcRed;
    float* d_srcGreen;
    float* d_srcBlue;
    checkCudaErrors(cudaMalloc(&d_srcRed, sizeof(float) * numElems));
    checkCudaErrors(cudaMalloc(&d_srcGreen, sizeof(float) * numElems));
    checkCudaErrors(cudaMalloc(&d_srcBlue, sizeof(float) * numElems));
    separateChannels<<<gridSize, blockSize>>>(d_sourceImg, d_srcRed, d_srcGreen, d_srcBlue, numColsSource, numRowsSource);

    float* d_dstRed;
    float* d_dstGreen;
    float* d_dstBlue;
    checkCudaErrors(cudaMalloc(&d_dstRed, sizeof(float) * numElems));
    checkCudaErrors(cudaMalloc(&d_dstGreen, sizeof(float) * numElems));
    checkCudaErrors(cudaMalloc(&d_dstBlue, sizeof(float) * numElems));
    separateChannels<<<gridSize, blockSize>>>(d_destImg, d_dstRed, d_dstGreen, d_dstBlue, numColsSource, numRowsSource);

    float* d_nextRed;
    float* d_nextGreen;
    float* d_nextBlue;
    float* d_prevRed;
    float* d_prevGreen;
    float* d_prevBlue;
    checkCudaErrors(cudaMalloc(&d_nextRed, sizeof(float) * numElems));
    checkCudaErrors(cudaMalloc(&d_nextGreen, sizeof(float) * numElems));
    checkCudaErrors(cudaMalloc(&d_nextBlue, sizeof(float) * numElems));
    checkCudaErrors(cudaMalloc(&d_prevRed, sizeof(float) * numElems));
    checkCudaErrors(cudaMalloc(&d_prevGreen, sizeof(float) * numElems));
    checkCudaErrors(cudaMalloc(&d_prevBlue, sizeof(float) * numElems));    
    checkCudaErrors(cudaMemcpy(d_prevRed, d_srcRed, sizeof(float) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_prevGreen, d_srcGreen, sizeof(float) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_prevBlue, d_srcBlue, sizeof(float) * numElems, cudaMemcpyDeviceToDevice));
    
    for (int i = 0; i < 800; ++i) {
        jacobi<<<gridSize, blockSize>>>(d_srcRed, d_dstRed, d_mask, d_prevRed, d_nextRed, numColsSource, numRowsSource);
        jacobi<<<gridSize, blockSize>>>(d_srcGreen, d_dstGreen, d_mask, d_prevGreen, d_nextGreen, numColsSource, numRowsSource);
        jacobi<<<gridSize, blockSize>>>(d_srcBlue, d_dstBlue, d_mask, d_prevBlue, d_nextBlue, numColsSource, numRowsSource);
        checkCudaErrors(cudaMemcpy(d_prevRed, d_nextRed, sizeof(float) * numElems, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_prevGreen, d_nextGreen, sizeof(float) * numElems, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_prevBlue, d_nextBlue, sizeof(float) * numElems, cudaMemcpyDeviceToDevice));
    }

    combineChannels<<<gridSize, blockSize>>>(d_sourceImg, d_nextRed, d_nextGreen, d_nextBlue, numColsSource, numRowsSource);
    checkCudaErrors(cudaMemcpy(h_blendedImg, d_sourceImg, sizeof(uchar4) * numElems, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_sourceImg));
    checkCudaErrors(cudaFree(d_destImg));
    checkCudaErrors(cudaFree(d_mask));
    checkCudaErrors(cudaFree(d_srcRed));
    checkCudaErrors(cudaFree(d_srcGreen));
    checkCudaErrors(cudaFree(d_srcBlue));
    checkCudaErrors(cudaFree(d_dstRed));
    checkCudaErrors(cudaFree(d_dstGreen));
    checkCudaErrors(cudaFree(d_dstBlue));
    checkCudaErrors(cudaFree(d_nextRed));
    checkCudaErrors(cudaFree(d_nextGreen));
    checkCudaErrors(cudaFree(d_nextBlue));
    checkCudaErrors(cudaFree(d_prevRed));
    checkCudaErrors(cudaFree(d_prevGreen));
    checkCudaErrors(cudaFree(d_prevBlue));
}
