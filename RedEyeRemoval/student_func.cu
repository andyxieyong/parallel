//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <assert.h>
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!

*/

__global__
void scan(unsigned int* const d_inputVals, 
          unsigned int* const d_scan, 
          unsigned int bit, 
          unsigned int block,
          size_t numElems)
{
    unsigned int offset = blockDim.x * block;
    int id = threadIdx.x + offset;
    if (id >= numElems) {
        return;
    }

    d_scan[id] = (d_inputVals[id] & bit) == 0 ? 1 : 0;
    __syncthreads();

    // Inclusive Hillis-Steele scan
    unsigned int value = 0;
	for (int i = 1; i < blockDim.x; i <<= 1) {
        value = id >= i+offset ? d_scan[id-i] + d_scan[id] : d_scan[id];
		__syncthreads();
		d_scan[id] = value;
		__syncthreads();
    }
    if (offset > 0) {
        d_scan[id] += d_scan[offset - 1];
    }
}

__global__
void move(unsigned int* const d_inputVals,
          unsigned int* const d_inputPos,
          unsigned int* const d_outputVals,
          unsigned int* const d_outputPos,
          unsigned int* const d_scan,
          const size_t numElems,          
          unsigned int bit)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= numElems) {
        return;
    }
    unsigned int startPos = d_scan[numElems-1]; 
    unsigned int index = 0;
    if ((d_inputVals[id] & bit) == 0) {
        index = id == 0 ? 0 : d_scan[id-1];
    } else {
        index = startPos + id - (id == 0 ? 0 : d_scan[id-1]);
    }
    assert(index < numElems);
    d_outputVals[index] = d_inputVals[id];
    d_outputPos[index] = d_inputPos[id];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    const dim3 gridSize((numElems / 1024) + 1);
    const dim3 blockSize(1024);

    unsigned int* d_scan;
    checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int) * numElems));

    for (unsigned int i = 0; i < 8 * sizeof(unsigned int); ++i) {
        unsigned int bit = 1 << i;
        for (unsigned int j = 0; j < gridSize.x; ++j) {
            // Todo: these should be run in parallel and merged!
            scan<<<1, blockSize>>>(d_inputVals, d_scan, bit, j, numElems);
        }
        move<<<gridSize, blockSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_scan, numElems, bit);

        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGetLastError());
    }
    cudaFree(d_scan);
    checkCudaErrors(cudaGetLastError());
}
