#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // for random numbers
#include <omp.h>
#include <vector>
#include <cstring> // for strcpy
#include <string>

#define BLOCK_SIZE 4
int N = 10000;

using namespace std;

typedef struct {
  int width;
  int height;
  int stride;
  int** elements;
} Matrix;


//__device__
Matrix GetSubMatrix(int **B, int row, int col, int N)
{
  Matrix Asub;
  Asub.width    = BLOCK_SIZE;
  Asub.height   = BLOCK_SIZE;
  Asub.stride   = N;
  Asub.elements = new int*[BLOCK_SIZE];
  for(int i=0; i < BLOCK_SIZE; i++){
    Asub.elements[i] = &B[BLOCK_SIZE * row+i][BLOCK_SIZE * col];
      std::cout << BLOCK_SIZE * row+i << "," << BLOCK_SIZE * col << std::endl;
    }
  return Asub;
}

using namespace std;

// -------------------------------------------------- pairing
int paired(char a1, char a2)
{
  if(a1 == 'A' && a2 == 'U')
    return 1;
  if(a1 == 'U' && a2 == 'A')
    return 1;
  if(a1 == 'G' && a2 == 'C')
    return 1;
  if(a1 == 'C' && a2 == 'G')
    return 1;

  return 0;
}

__device__ int _paired(char a, char b) {
  if ((a == 'A' && b == 'U') || (a == 'U' && b == 'A') || (a == 'C' && b == 'G') || (a == 'G' && b == 'C')) {
    return 1;
  }
  return 0;
}

// --------------------------------------------------
// KERNEL

__global__ void myKernel(int **B, int N, int c0, char* seqq)
{
        int c1 = blockIdx.x + c0;
        int bb = BLOCK_SIZE;
        extern __shared__ int S[BLOCK_SIZE][BLOCK_SIZE];

        if(c1 <= min((N - 1) / bb, (N + c0 - 2 )/ bb))
        //for (int c1 = c0; c1 <= min((N - 1) / 16, (N + c0 - 2 )/ 16); c1 += 1) // parallel loop  blocks
        {
            int _si = c1*bb;
            int _sj = _si-c0*bb;

            /*
            for (int m = 0; m < (N / BLOCK_SIZE); ++m) {


        // Get sub-matrix Asub of A
               Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
*/



          //int i = c1*16;
          //int j = i-c0*16;

        //    for(int j=0; j<bb; j++)
        //      for(int i=0; i<N; i++)
            ////      S[j*N + i] = B[j+_sj][i];


            // shared block from i to i+16-1, j to j+16-1
            //cout << "\n -------- NEW BLOCK CORNER " << j << "," << i << endl;


            for (int c2 = max(1, bb * c0 - bb - 1);
                 c2 <= min(bb * c0 + bb - 1, N + bb * c0 - bb * c1 - 1); c2 += 1) { // serial loop
                if (c0 >= 1) {
                    //    #pragma omp parallel for
                    int lb = max(bb * c1, -bb * c0 + bb * c1 + c2);
                    int ub = min(min(N - 1, bb * c1 + bb-1), -bb * c0 + bb * c1 + c2 + bb-1);
                    int c3 = threadIdx.x+ lb;
                    if(c3<=ub) {
                      register int z = B[-c2 + c3][c3];
                     // for (int c3 = max(16 * c1, -16 * c0 + 16 * c1 + c2); c3 <= min(min(N - 1, 16 * c1 + 15), -16 * c0 + 16 * c1 + c2 + 15); c3 += 1) {   // parallel loop threads
                      int bound = (c2 / bb) *bb -1;

                      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                      for (int c4 = 0; c4 < bound; c4 += 1) // serial
                              // tu  mozna zmniejszyc o 1  -c2 + c3 + c4
                             // policz tylko dla srodkowych blokow w x i y czyli bez bloku sj, si
                              z = max(B[-c2 + c3][-c2 + c3 + c4  /* !!! */ - 1] + B[-c2 + c3 + c4 + 1 /* !!! */ - 1][c3], z);
                      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


                          // przelicz bez y, tylko dla 1 watka ostatna kolumne i rzad
                          if(threadIdx.y ==0){
                           int c4 = bound;
                           z = max(B[-c2 + c3][-c2 + c3 + c4] + B[-c2 + c3 + c4 + 1][c3], z);
                            // przelicz biezacy blok

                            for (int c4 = bound+1; c4 < c2; c4 += 1)
                              z = max(B[-c2 + c3][-c2 + c3 + c4 ] + B[-c2 + c3 + c4 + 1][c3], z);

                          B[-c2 + c3][c3] = max(z,
                                                B[-c2 + c3 + 1][c3 - 1] + _paired(seqq[-c2 + c3], seqq[c3]));
                          }
                      }

                } else {
                    //  #pragma omp parallel for
                   // printf("%i %i\n", _sj, _si);
                  int lb = bb * c1 + c2;
                  int ub = min(N - 1, bb * c1 + bb-1);
                  int c3 = threadIdx.x + lb;  // threadIdx.x
                  if(c3<=ub) {    // mozna dac ostra wtedy policzy  bez czwartej
                  //for (int c3 = 16 * c1 + c2; c3 <= min(N - 1, 16 * c1 + 15); c3 += 1) {   // parallel loop threads
                    register int z = B[-c2 + c3][c3];
                        for (int c4 = 0; c4 < c2; c4 += 1) {  // serial
                            z = max(B[-c2 + c3][-c2 + c3 + c4] + B[-c2 + c3 + c4 + 1][c3],  z);
                        }
                        B[-c2 + c3][c3] = max(z,
                                              B[-c2 + c3 + 1][c3 - 1] + _paired(seqq[-c2 + c3], seqq[c3]));
                        if(c1==0)
                        printf("%i %i %i\n", -c2+c3, c3, B[-c2 + c3][c3]);
                    }

                }
            }
        }

}




// --------------------------------------------------


int main() {

  //string seq = "GUACGUACGUACGUACGUACGUACGUACGUAC";
  string seq = "GUACGUACGUACGUACGUAC";
  int N = seq.length();

  int n = N, i,j,k;

  char *seqq = new char[N+1];
  std::strcpy(seqq, seq.c_str());          // Copy the string content   // use random data for given big N, comment this

  int* flatArray_S = new int[n * n];
  int* flatArray_S_CPU = new int[n * n];

  // Allocate 2D host array for CPU and GPU
  int** S = new int*[n];
  int** S_CPU = new int*[n];

  for(int i = 0; i < n; i++) {
    S[i] = &flatArray_S[i * n];
    S_CPU[i] = &flatArray_S_CPU[i * n];
  }
  // initialization
  for(i=0; i<N; i++) {
    for(j=0; j<N; j++){
      S[i][j] = INT_MIN;
      S_CPU[i][j] = INT_MIN;
    }
  }
  for(i=0; i<N; i++){
    S[i][i] = 0;
    S_CPU[i][i] = 0;
    if(i+1 < N) {
      S[i][i + 1] = 0;
      S[i+1][i] = 0;
      S_CPU[i][i+1] = 0;
      S_CPU[i+1][i] = 0;
    }
  }
  // -----------------------------

  // cuda memory allocation
  int* flat_d_S;
  int** d_S;
  char *d_sequence;

  double start_time = omp_get_wtime();
  cudaMalloc(&d_sequence, n);
  cudaMalloc(&flat_d_S, n * n * sizeof(int));
  cudaMalloc(&d_S, n * sizeof(int*));

  int* h_S[n];  // copy flat_d_S pointers to vector on host and copy to d_S vector of pointers
  for(int i = 0; i < n; i++) {
    h_S[i] = flat_d_S + i * n;
  }
  cudaMemcpy(d_S, h_S, n * sizeof(int*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sequence, seqq, n, cudaMemcpyHostToDevice);
  // Copy host data to device before entering the loop
  cudaMemcpy(flat_d_S, &S[0][0], n * n * sizeof(int), cudaMemcpyHostToDevice);

  int numBlocks = (n) / BLOCK_SIZE;
  int bb = BLOCK_SIZE;

  //numBlocks = min((N - 1) / 16, (N + c0 - 2 )/ 16) - c0;
  for (int c0 = 0; c0 <= (N - 1)/bb; c0 += 1)  // serial loop
  {
    //for (int c1 = c0; c1 <= min((N - 1) / 16, (N + c0 - 2 )/ 16); c1 += 1) // parallel loop  blocks
    numBlocks = min((N - 1) / bb, (N + c0 - 2 )/ bb) - c0 + 1;
    myKernel<<<numBlocks, BLOCK_SIZE>>>(d_S, n, c0, d_sequence);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error on kernel launch: %s\n", cudaGetErrorString(err));
  }



  cudaMemcpy(&S[0][0], flat_d_S, n * n * sizeof(int), cudaMemcpyDeviceToHost);

  double end_time = omp_get_wtime();
  double elapsed_time = end_time - start_time;
  printf("Time taken: %f seconds\n", elapsed_time);

  printf("gpu ended\n");


  cout << endl << endl;
  if(1==1)
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      if(S[i][j] < 0)
        cout << "";
      else
        cout << S[i][j];
      cout << "\t";
    }
    cout << "\n";
  }
  cout << endl;

  Matrix C_SUB =  GetSubMatrix(S , 1, 1, N);

  for(i=0; i<BLOCK_SIZE; i++){
    for(j=0; j<BLOCK_SIZE; j++){
      if(C_SUB.elements[i][j] < 0)
        cout << "";
      else
        cout << C_SUB.elements[i][j];
      cout << "\t";
   }
  cout << "\n";
  }
  cout << endl;

 // kontrola z cpu
  for (i = N-1; i >= 0; i--) {
    for (j = i+1; j < N; j++) {
      for (k = 0; k < j-i; k++) {
        S_CPU[i][j] = max(S_CPU[i][k+i] + S_CPU[k+i+1][j], S_CPU[i][j]);
      }

      S_CPU[i][j] = max(S_CPU[i][j], S_CPU[i+1][j-1] + paired(seqq[i],seqq[j]));

      //  cout << i << "|" << j << "|" << seqq[i] << seqq[j] << "|" << S[i][j] << " , " << paired(seqq[i],seqq[j])  << "| " << S[i+1][j-1]<< endl;

    }
  }


  for(i=0; i<N; i++)
    for(j=0; j<N; j++)
      if(S[i][j] != S_CPU[i][j]){
        cout << "error" << endl;
        exit(0);
      }


  delete[] S;
  delete[] S_CPU;

  cudaFree(d_S);
  cudaFree(flat_d_S);

  return 0;
}