#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // for random numbers
#include <omp.h>
#include <vector>
#include <cstring> // for strcpy
#include <string>
#include <ctime>     // for time()

#define BLOCK_SIZE 32
int N = 33000;

using namespace std;

// -------------------------------------------------- pairing
short paired(char a1, char a2)
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

__device__ short _paired(char a, char b) {
  if ((a == 'A' && b == 'U') || (a == 'U' && b == 'A') || (a == 'C' && b == 'G') || (a == 'G' && b == 'C')) {
    return 1;
  }
  return 0;
}

// --------------------------------------------------
// KERNEL

__global__ void myKernel(short **B, short N, short c0, char* seqq)
{
        register short c1 = blockIdx.x + c0;
        register short bb = BLOCK_SIZE;
        __shared__ short C[BLOCK_SIZE][BLOCK_SIZE];

        if(c1 <= min((N - 1) / bb, (N + c0 - 2 )/ bb))
        //for (short c1 = c0; c1 <= min((N - 1) / 16, (N + c0 - 2 )/ 16); c1 += 1) // parallel loop  blocks
        {
            register short _sj = c1-c0;
            register short _si = c1;


         for (short m = _sj+1; m < _si; ++m) {

           // Thread row and column
               register short row = threadIdx.y;
               register short col = threadIdx.x;

             __shared__ short A_elements[BLOCK_SIZE][BLOCK_SIZE];
             __shared__ short B_elements[BLOCK_SIZE][BLOCK_SIZE];
 
              A_elements[row][col] = B[BLOCK_SIZE * _sj+row][BLOCK_SIZE * m -1 + col];
              B_elements[row][col] = B[BLOCK_SIZE * m +row][BLOCK_SIZE * _si + col];

             if(row < BLOCK_SIZE && col < BLOCK_SIZE){

              register short Cvalue = 0;

              __syncthreads();

              #pragma unroll
              for (short e = 0; e < BLOCK_SIZE; e++)
              {
                  Cvalue = max(A_elements[row][e] + B_elements[e][col], Cvalue);
              }

              __syncthreads();

                C[row][col] = C[row][col] > Cvalue ? C[row][col] : Cvalue ;

            }

           }

            for (short c2 = max(1, bb * c0 - bb - 1);
                 c2 <= min(bb * c0 + bb - 1, N + bb * c0 - bb * c1 - 1); c2 += 1) { // serial loop
                if (c0 >= 1) {
                    //    #pragma omp parallel for
                    short lb = max(bb * c1, -bb * c0 + bb * c1 + c2);
                    short ub = min(min(N - 1, bb * c1 + bb-1), -bb * c0 + bb * c1 + c2 + bb-1);
                    short c3 = threadIdx.x+ lb;
                    if(c3<=ub) {

                      register short z = B[-c2 + c3][c3];
                     // for (short c3 = max(16 * c1, -16 * c0 + 16 * c1 + c2); c3 <= min(min(N - 1, 16 * c1 + 15), -16 * c0 + 16 * c1 + c2 + 15); c3 += 1) {   // parallel loop threads

                      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                      if(1==1){


                        if(threadIdx.y ==0){

                          short _j = (-c2+c3) % BLOCK_SIZE;
                          short _i = c3 % BLOCK_SIZE;


                          for (short c4 = 0; c4 < bb-1; c4 += 1)  // blocks 0 (triangles)
                            z = max(B[-c2 + c3][-c2 + c3 + c4 ] + B[-c2 + c3 + c4 + 1][c3], z);

                          z = z > C[_j][_i] ? z : C[_j][_i]; // middle blocks

                         short fragment = (c1 == N/BLOCK_SIZE-1); // last column

                        for (short c4 =  c2 - bb - fragment; c4 < c2; c4 += 1)   // current tile
                          z = max(B[-c2 + c3][-c2 + c3 + c4] + B[-c2 + c3 + c4 + 1][c3], z);

                          B[-c2 + c3][c3] = max(z,
                                               B[-c2 + c3 + 1][c3 - 1] +  _paired(seqq[-c2 + c3] , seqq[c3] ));
                          }
                      }

                      else // original generated code
                        {
                        for (short c4 = 0; c4 < c2; c4 += 1) {  // serial
                          z = max(B[-c2 + c3][-c2 + c3 + c4] + B[-c2 + c3 + c4 + 1][c3],  z);
                        }
                        B[-c2 + c3][c3] = max(z,
                                              B[-c2 + c3 + 1][c3 - 1] + _paired(seqq[-c2 + c3], seqq[c3]));
                        }
                      }

                } else {
                    //  #pragma omp parallel for
                  short lb = bb * c1 + c2;
                  short ub = min(N - 1, bb * c1 + bb-1);
                  short c3 = threadIdx.x + lb;  // threadIdx.x
                  if(c3<=ub) {
                  //for (short c3 = 16 * c1 + c2; c3 <= min(N - 1, 16 * c1 + 15); c3 += 1) {   // parallel loop threads
                    register short z = B[-c2 + c3][c3];
                        for (short c4 = 0; c4 < c2; c4 += 1) {  // serial
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
 //short N = seq.length();


 int n = N, i,j,k;

  char *seqq = new char[N+1];
  if(N>1) // no debug
   {
    char znaki[] = {'C', 'G', 'U', 'A'};
    srand(static_cast<unsigned short>(time(0)));

    for (short i = 0; i < N; i++) {
      seqq[i] = znaki[rand() % 4];  // Losowy wybór z zestawu 'C', 'G', 'U', 'A'
    }
   }
   cout << seqq << endl;
  std::strcpy(seqq, seq.c_str());          // Copy the string content   // use random data for given big N, comment this

  short* flatArray_S = new short[n * n];
  short* flatArray_S_CPU = new short[n * n];

  // Allocate 2D host array for CPU and GPU
  short** S = new short*[n];
  short** S_CPU = new short*[n];

  for(short i = 0; i < n; i++) {
    S[i] = &flatArray_S[i * n];
    S_CPU[i] = &flatArray_S_CPU[i * n];
  }
  // initialization
  for(i=0; i<N; i++) {
    for(j=0; j<N; j++){
      S[i][j] = -1;
      S_CPU[i][j] = -1;
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
  short* flat_d_S;
  short** d_S;
  char *d_sequence;

  double start_time = omp_get_wtime();
  cudaMalloc(&d_sequence, n);
  cudaMalloc(&flat_d_S, n * n * sizeof(short));
  cudaMalloc(&d_S, n * sizeof(short*));

  short* h_S[n];  // copy flat_d_S pointers to vector on host and copy to d_S vector of pointers
  for(short i = 0; i < n; i++) {
    h_S[i] = flat_d_S + i * n;
  }
  cudaMemcpy(d_S, h_S, n * sizeof(short*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sequence, seqq, n, cudaMemcpyHostToDevice);
  // Copy host data to device before entering the loop
  cudaMemcpy(flat_d_S, &S[0][0], n * n * sizeof(short), cudaMemcpyHostToDevice);

  short numBlocks = (n) / BLOCK_SIZE;
  short bb = BLOCK_SIZE;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  //numBlocks = min((N - 1) / 16, (N + c0 - 2 )/ 16) - c0;
  for (short c0 = 0; c0 <= (N - 1)/bb; c0 += 1)  // serial loop
  {
    //for (short c1 = c0; c1 <= min((N - 1) / 16, (N + c0 - 2 )/ 16); c1 += 1) // parallel loop  blocks
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

  cudaMemcpy(&S[0][0], flat_d_S, n * n * sizeof(short), cudaMemcpyDeviceToHost);

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
