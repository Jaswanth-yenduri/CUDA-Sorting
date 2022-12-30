/**
 * 
 * @authors
 * Jaswanth Yenduri (jyendur@siue.edu)  800746158
 * Likitha Vinjam   (lvinjam@siue.edu)  800748958
 * Manisha Reddy Tummala  (mtummal@siue.edu) 800722182
 * 
 */

/*
 * @file thrust.cu
*/

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
using namespace std;


/**********************************************************
* **********************************************************
* error checking stufff
***********************************************************
***********************************************************/
// Enable this for error checking
#define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )


inline void __cudaSafeCall( cudaError err, const char *file, const int line ) {
    #ifdef CUDA_CHECK_ERROR
    #pragma warning( push )
    #pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
    do {
        if ( cudaSuccess != err ) {
            fprintf( stderr,
            "cudaSafeCall() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while ( 0 );
    #pragma warning( pop )
    #endif // CUDA_CHECK_ERROR
    return;
}

inline void __cudaCheckError( const char *file, const int line ) {
    #ifdef CUDA_CHECK_ERROR
    #pragma warning( push )
    #pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
    do {
        cudaError_t err = cudaGetLastError();
        if ( cudaSuccess != err ) {
            fprintf( stderr,
            "cudaCheckError() failed at %s:%i : %s.\n",
            file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }

        // More careful checking. However, this will affect performance.
        // Comment if not needed.
        err = cudaDeviceSynchronize();
        if( cudaSuccess != err ) {
            fprintf( stderr,
            "cudaCheckError() with sync failed at %s:%i : %s.\n",
            file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while ( 0 );

    #pragma warning( pop )
    #endif // CUDA_CHECK_ERROR
    return;
}
/***************************************************************
* **************************************************************
* end of error checking stuff
****************************************************************
***************************************************************/


// function takes an array pointer, and the number of rows and cols in the array, and
// allocates and intializes the array to a bunch of random numbers
// Note that this function creates a 1D array that is a flattened 2D array
// to access data item data[i][j], you must can use data[(i*rows) + j]
int * makeRandArray( const int size, const int seed ) {
    srand( seed );
    int * array = new int[ size ];

    for( int i = 0; i < size; i ++ )
        array[i] = std::rand() % 1000000;

    return array;
}


//*******************************//
//your kernel here!!!!!!!!!!!!!!!!!
//*******************************//
__global__ void matavgKernel( int * device_nums, int N ) {
}


int main( int argc, char* argv[] )
{
    int * array; // the poitner to the array of rands
    int size, seed; // values for the size of the array and the seed for generating random numbers
    bool printSorted = false;

    // check the command line args
    if( argc < 4 ){
        std::cerr << "usage: " << argv[0]
                    << " [amount of random nums to generate] [seed value for rand]"
                    << " [1 to print sorted array, 0 otherwise]" 
                    << std::endl;
        exit( -1 );
    }

    // convert cstrings to ints
    {
        std::stringstream ss1( argv[1] );
        ss1 >> size;
    }

    {
        std::stringstream ss1( argv[2] );
        ss1 >> seed;
    }

    {
        int sortPrint;
        std::stringstream ss1( argv[3] );
        ss1 >> sortPrint;
        if( sortPrint == 1 )
            printSorted = true;
    }

    // get the random numbers
    array = makeRandArray( size, seed );

    if( printSorted ){
        ///////////////////////////////////////////////
        /// Your code to print the sorted array here //
        ///////////////////////////////////////////////

        for( int i = 0; i < size; i ++ )
            cout << array[i] << " ";
        cout << endl;
    }

    /***********************************
    * create a cuda timer to time execution
    **********************************/
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord( startTotal, 0 );
    /***********************************
    * end of cuda timer creation
    **********************************/

    thrust::device_vector<int> d_vec(array, array+size);
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::copy(d_vec.begin(), d_vec.end(), array);
    /////////////////////////////////////////////////////////////////////
    ///////////////////////    YOUR CODE HERE     ///////////////////////
    /////////////////////////////////////////////////////////////////////
    /*
    * You need to implement your kernel as a function at the top of this file.
    * Here you must
    * 1) allocate device memory
    * 2) set up the grid and block sizes
    * 3) call your kenrnel
    * 4) get the result back from the GPU
    *
    *
    * to use the error checking code, wrap any cudamalloc functions as follows:
    * CudaSafeCall( cudaMalloc( &pointer_to_a_device_pointer,
    * length_of_array * sizeof( int ) ) );
    * Also, place the following function call immediately after you call your kernel
    * ( or after any other cuda call that you think might be causing an error )
    * CudaCheckError();
    */

    /***********************************
    * Stop and destroy the cuda timer
    **********************************/
    cudaEventRecord( stopTotal, 0 );
    cudaEventSynchronize( stopTotal );
    cudaEventElapsedTime( &timeTotal, startTotal, stopTotal );
    cudaEventDestroy( startTotal );
    cudaEventDestroy( stopTotal );
    /***********************************
    * end of cuda timer destruction
    **********************************/

    std::cerr << "Total time in seconds: " << timeTotal / 1000.0 << std::endl;

    if( printSorted ){
        ///////////////////////////////////////////////
        /// Your code to print the sorted array here //
        ///////////////////////////////////////////////

        for( int i = 0; i < size; i ++ )
            cout << array[i] << " ";
        cout << endl;
    }
}
