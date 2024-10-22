#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // for random numbers
#include <omp.h>
#include <vector>
#include <cstring> // for strcpy
#include <string>
#include <ctime>     // for time()

#define BLOCK_SIZE 32
int N = 1000;

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
        __shared__ int C[BLOCK_SIZE][BLOCK_SIZE];

        if(c1 <= min((N - 1) / bb, (N + c0 - 2 )/ bb))
        //for (int c1 = c0; c1 <= min((N - 1) / 16, (N + c0 - 2 )/ 16); c1 += 1) // parallel loop  blocks
        {
            int _sj = c1-c0;
            int _si = c1;


         for (int m = _sj+1; m < _si; ++m) {

           // Thread row and column
               int row = threadIdx.y;
               int col = threadIdx.x;

              __shared__ int * A_elements[BLOCK_SIZE];
              __shared__ int * B_elements[BLOCK_SIZE];

              A_elements[row] = &B[BLOCK_SIZE * _sj+row][BLOCK_SIZE * m -1];
              B_elements[row] = &B[BLOCK_SIZE * m +row][BLOCK_SIZE * _si];

             if(row < BLOCK_SIZE && col < BLOCK_SIZE){

              register int Cvalue = 0;

              __syncthreads();

              #pragma unroll
              for (int e = 0; e < BLOCK_SIZE; e++)
              {
                  Cvalue = max(A_elements[row][e] + B_elements[e][col], Cvalue);
              }

              __syncthreads();

                C[row][col] = max(C[row][col], Cvalue);

            }

           }

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

                      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                      if(1==1){


                        if(threadIdx.y ==0){

                          int _j = (-c2+c3) % BLOCK_SIZE;
                          int _i = c3 % BLOCK_SIZE;


                          for (int c4 = 0; c4 < bb-1; c4 += 1)  // blocks 0 (triangles)
                            z = max(B[-c2 + c3][-c2 + c3 + c4 ] + B[-c2 + c3 + c4 + 1][c3], z);

                          z = max(z, C[_j][_i]); // middle blocks

                         int fragment = (c1 == N/BLOCK_SIZE-1); // last column

                        for (int c4 =  c2 - bb - fragment; c4 < c2; c4 += 1)   // current tile
                          z = max(B[-c2 + c3][-c2 + c3 + c4] + B[-c2 + c3 + c4 + 1][c3], z);

                          B[-c2 + c3][c3] = max(z,
                                               B[-c2 + c3 + 1][c3 - 1] +  _paired(seqq[-c2 + c3] , seqq[c3] ));
                          }
                      }

                      else // original generated code
                        {
                        for (int c4 = 0; c4 < c2; c4 += 1) {  // serial
                          z = max(B[-c2 + c3][-c2 + c3 + c4] + B[-c2 + c3 + c4 + 1][c3],  z);
                        }
                        B[-c2 + c3][c3] = max(z,
                                              B[-c2 + c3 + 1][c3 - 1] + _paired(seqq[-c2 + c3], seqq[c3]));
                        }
                      }

                } else {
                    //  #pragma omp parallel for
                  int lb = bb * c1 + c2;
                  int ub = min(N - 1, bb * c1 + bb-1);
                  int c3 = threadIdx.x + lb;  // threadIdx.x
                  if(c3<=ub) {
                  //for (int c3 = 16 * c1 + c2; c3 <= min(N - 1, 16 * c1 + 15); c3 += 1) {   // parallel loop threads
                    register int z = B[-c2 + c3][c3];
                        for (int c4 = 0; c4 < c2; c4 += 1) {  // serial
                            z = max(B[-c2 + c3][-c2 + c3 + c4] + B[-c2 + c3 + c4 + 1][c3],  z);
                        }
                        B[-c2 + c3][c3] = max(z,
                                              B[-c2 + c3 + 1][c3 - 1] + _paired(seqq[-c2 + c3], seqq[c3]));

                    }

                }
            }
        }

}


// --------------------------------------------------


int main() {



 // string seq = "UCGCUACCAUUGCUUCUAGACCUACGAAAUAGUCUCAUCUCUACGGCAGUAGUGCAUCUGUGUCGCGCUGUUCGUGAACCGAGACGUUGCAAGUCUUGUGUCAUUUAGGCGUAUGCACUGCUCUCCCU";
   string seq = "GUACGUACGUACGUACGUAC";
  seq = "CUGGUUUAUGUCACCCAGCAGCAGACCCUCCUUUACCGAAAGAUGAUGCUCGUAUUAUUGUACG";
  N += BLOCK_SIZE - N % BLOCK_SIZE;
 //int N = seq.length();


  int n = N, i,j,k;

  char *seqq = new char[N+1];
  if(N>1) // no debug
   {
    char znaki[] = {'C', 'G', 'U', 'A'};
    srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < N; i++) {
      seqq[i] = znaki[rand() % 4];  // Losowy wybór z zestawu 'C', 'G', 'U', 'A'
    }
   }
   cout << seqq << endl;
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
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  //numBlocks = min((N - 1) / 16, (N + c0 - 2 )/ 16) - c0;
  for (int c0 = 0; c0 <= (N - 1)/bb; c0 += 1)  // serial loop
  {
    //for (int c1 = c0; c1 <= min((N - 1) / 16, (N + c0 - 2 )/ 16); c1 += 1) // parallel loop  blocks
    numBlocks = min((N - 1) / bb, (N + c0 - 2 )/ bb) - c0 + 1;
    myKernel<<<numBlocks, dimBlock>>>(d_S, n, c0, d_sequence);


    cudaError_t errSync  = cudaDeviceSynchronize();

    // Sprawdzenie błędów związanych z wywołaniem kernela (np. błędne parametry wywołania)
    cudaError_t errAsync = cudaGetLastError();

    // Sprawdzenie, czy pojawiły się błędy
    if (errSync != cudaSuccess) {
      printf("Cuda synchronization error: %s\n", cudaGetErrorString(errSync));
      exit(1);
    }

    if (errAsync != cudaSuccess) {
      printf("Cuda asynchronous kernel error: %s\n", cudaGetErrorString(errAsync));
      exit(1);
    }

  }

  cudaMemcpy(&S[0][0], flat_d_S, n * n * sizeof(int), cudaMemcpyDeviceToHost);

  double end_time = omp_get_wtime();
  double elapsed_time = end_time - start_time;
  printf("Time taken: %f seconds\n", elapsed_time);

  printf("gpu ended\n");


  cout << endl << endl;
  if(1==0)
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


 // cpu control   loop uday dynamic tiling paper
  //if(1==0)
  for (i = N-1; i >= 0; i--) {
    for (j = i+1; j < N; j++) {
      for (k = 0; k < j-i; k++) {
        S_CPU[i][j] = max(S_CPU[i][k+i] + S_CPU[k+i+1][j], S_CPU[i][j]);
      }

      S_CPU[i][j] = max(S_CPU[i][j], S_CPU[i+1][j-1] + paired(seqq[i],seqq[j]));

    }
  }

  for(i=0; i<N; i++)
    for(j=0; j<N; j++)
      if(S[i][j] != S_CPU[i][j]){
        cout << i <<" " <<  j << ":" << S[i][j] << " " << S_CPU[i][j] << endl;
        cout << "error" << endl;
        exit(1);

      }


  delete[] S;
  delete[] S_CPU;

  cudaFree(d_S);
  cudaFree(flat_d_S);

  return 0;
}