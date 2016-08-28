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

// Create scan sections (e.g. 0...1023) according to current bit value.
__global__
void scanSections(unsigned int* const d_inputVals, 
                  unsigned int* const d_scan, 
                  unsigned int bit,
                  size_t numElems)
{
    unsigned int offset = blockDim.x * blockIdx.x;
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
}

// Take the highest value (=last value) of each scan section (0...1023,
// 1024...2047) and create another scan array from them so the values
// can be added to the original scan array.
__global__
void scanHighestValues(unsigned int* d_scan, unsigned int* d_scanHighest, unsigned int blockSize)
{
    int id = threadIdx.x;
    d_scanHighest[id] = id == 0 ? 0 : d_scan[id * blockSize - 1];
    __syncthreads();

    unsigned int value = 0;
	for (int i = 1; i < blockDim.x; i <<= 1) {
        value = id >= i ? d_scanHighest[id-i] + d_scanHighest[id] : d_scanHighest[id];
		__syncthreads();
		d_scanHighest[id] = value;
		__syncthreads();
	}
}

// Add the result of scanHighestValues-function so scan sections
// (i.e. all input values) are merged to one continous scan array.
__global__
void scanMerge(unsigned int* d_scan, unsigned int* d_scanHighest, unsigned int blockSize, size_t numElems)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= numElems) {
        return;
    }
    unsigned int highestId = id / blockSize;
    d_scan[id] += d_scanHighest[highestId];
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
    unsigned int* d_scanHighest;
    checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMalloc(&d_scanHighest, sizeof(unsigned int) * gridSize.x));

    for (unsigned int i = 0; i < 8 * sizeof(unsigned int); ++i) {
        unsigned int bit = 1 << i;
        // Since the maximum number of threads per block is 1024, an
        // array of hundreds of thousands values can not be scanned at
        // once. Thus scan is done in three parts:
        // 1. Create scan array for each section according to block size (e.g. 0-1023, 1024-2047 etc.)
        // 2. Create another scan array from the last values of each block (highest values)
        // 3. Add the highest scan values to the original scan array to form one continous scan array
        scanSections<<<gridSize, blockSize>>>(d_inputVals, d_scan, bit, numElems);
        scanHighestValues<<<1, gridSize>>>(d_scan, d_scanHighest, blockSize.x);
        scanMerge<<<gridSize, blockSize>>>(d_scan, d_scanHighest, blockSize.x, numElems);

        move<<<gridSize, blockSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_scan, numElems, bit);

        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }
    checkCudaErrors(cudaFree(d_scan));
    checkCudaErrors(cudaFree(d_scanHighest));
}
