//Udacity HW 4
//Radix Sorting

#include "utils.h"
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

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__
void predicate(unsigned int* const d_inputVals, int* d_predicate, int bit, size_t numElems)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= numElems) {
        return;
    }
    
    d_predicate[id] = d_inputVals[id] & bit == 0 ? 1 : 0;
}

__global__
void scan(int* d_predicate, int* d_scan, int numElems)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= numElems) {
        return;
    }

    d_scan[id] = d_predicate[id];
    __syncthreads();

    // Inclusive Hillis-Steele scan
    unsigned int value = 0;
	for (int i = 1; i < numElems; i <<= 1) {
        value = id >= i ? d_scan[id-i] + d_scan[id] : d_scan[id];
		__syncthreads();
		d_scan[id] = value;
		__syncthreads();
	}
}

__global__
void move(unsigned int* const d_inputVals,
          unsigned int* const d_inputPos,
          const size_t numElems,
          int* d_predicate, 
          int* d_scan)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= numElems) {
        return;
    }
    int startPos = d_scan[numElems-1];
 
    // TODO
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    const dim3 gridSize((numElems / 512) + 1);
    const dim3 blockSize(512);

    int* d_predicate;
    int* d_scan;
    checkCudaErrors(cudaMalloc((void**) &d_predicate, sizeof(int) * numElems));
    checkCudaErrors(cudaMalloc((void**) &d_scan, sizeof(int) * numElems));

    unsigned int limit = std::numeric_limits<unsigned int>::max();
    for (unsigned int bit = 1; bit < limit; bit <<= 1) {
        predicate<<<gridSize, blockSize>>>(d_inputVals, d_predicate, bit, numElems);
        scan<<<gridSize, blockSize>>>(d_predicate, d_scan, numElems);
    }
    
    checkCudaErrors(cudaFree(d_predicate));
    checkCudaErrors(cudaFree(d_scan));
}
