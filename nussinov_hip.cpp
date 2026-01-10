#include <iostream>
#include <cstdlib> // for random numbers
#include <omp.h>
#include <vector>
#include <cstring> // for strcpy
#include <string>
#include <ctime>     // for time()
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 32
int N = 40000;

using namespace std;

// -------------------------------------------------- pairing
__host__ __device__ short _paired(char a, char b) {
  if ((a == 'A' && b == 'U') || (a == 'U' && b == 'A') || (a == 'C' && b == 'G') || (a == 'G' && b == 'C')) {
    return 1;
  }
  return 0;
}


#define HIP_CHECK(call) do {                                 \
  hipError_t _e = (call);                                    \
  if (_e != hipSuccess) {                                    \
    fprintf(stderr, "HIP error %s:%d: %s\n",                 \
            __FILE__, __LINE__, hipGetErrorString(_e));      \
    std::exit(1);                                            \
  }                                                          \
} while(0)
    
    
    
__global__ void myKernel(short **B, short N, short c0, char* seqq)
{
    
        short row = threadIdx.y;
        short col = threadIdx.x;
               
        short c1 = blockIdx.x + c0;
        short bb = BLOCK_SIZE;
        __shared__ short C[BLOCK_SIZE][BLOCK_SIZE];
        C[row][col] = 0;
        

        if(c1 <= min((N - 1) / bb, (N + c0 - 2 )/ bb))
        //for (short c1 = c0; c1 <= min((N - 1) / 16, (N + c0 - 2 )/ 16); c1 += 1) // parallel loop  blocks
        {
            short _sj = c1-c0;
            short _si = c1;


         for (short m = _sj+1; m < _si; ++m) {

           // Thread row and column
   
               

             __shared__ short A_elements[BLOCK_SIZE][BLOCK_SIZE];
             __shared__ short B_elements[BLOCK_SIZE][BLOCK_SIZE];
 
              A_elements[row][col] = B[BLOCK_SIZE * _sj+row][BLOCK_SIZE * m -1 + col];
              B_elements[row][col] = B[BLOCK_SIZE * m +row][BLOCK_SIZE * _si + col];

             if(row < BLOCK_SIZE && col < BLOCK_SIZE){

              short Cvalue = 0;

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

                      short z = B[-c2 + c3][c3];
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
                    short z = B[-c2 + c3][c3];
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


int main(){
    
  string seq = "GUACGUACGUACGUACGUAC";
  seq = "CUGGUUUAUGUCACCCAGCAGCAGACCCUCCUUUACCGAAAGAUGAUGCUCGUAUUAUUGUACG";
  N += BLOCK_SIZE - N % BLOCK_SIZE;
    
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
  HIP_CHECK( hipMalloc(&d_sequence, n));
  HIP_CHECK( hipMalloc(&flat_d_S, n * n * sizeof(short)));
  HIP_CHECK( hipMalloc(&d_S, n * sizeof(short*)));
  
    short* h_S[n];  // copy flat_d_S pointers to vector on host and copy to d_S vector of pointers
  for(short i = 0; i < n; i++) {
    h_S[i] = flat_d_S + i * n;
  }
  
  HIP_CHECK( hipMemcpy(d_S, h_S,
                     (size_t)n * sizeof(short*),
                     hipMemcpyHostToDevice) );

 HIP_CHECK( hipMemcpy(d_sequence, seqq,
                     (size_t)n,                 // jeśli seqq to char* i n jest w bajtach
                     hipMemcpyHostToDevice) );

  // Copy host data to device before entering the loop
  HIP_CHECK( hipMemcpy(flat_d_S, &S[0][0],
                     (size_t)n * (size_t)n * sizeof(short),
                     hipMemcpyHostToDevice) );
                     
  short numBlocks = (n) / BLOCK_SIZE;
  short bb = BLOCK_SIZE;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  
  for (short c0 = 0; c0 <= (N - 1) / bb; c0 += 1)  // serial loop
    {
        numBlocks = (short)(min((short)((N - 1) / bb),
                                 (short)((N + c0 - 2) / bb)) - c0 + 1);

        // HIP kernel launch:
        hipLaunchKernelGGL(myKernel,
                           dim3((unsigned int)numBlocks, 1, 1),
                           dim3((unsigned int)dimBlock.x, (unsigned int)dimBlock.y, 1),
                           0, 0,
                           d_S, n, c0, d_sequence);

        hipError_t errSync  = hipDeviceSynchronize();
        hipError_t errAsync = hipGetLastError();

        if (errSync != hipSuccess) {
            printf("HIP synchronization error: %s\n", hipGetErrorString(errSync));
            std::exit(1);
        }

        if (errAsync != hipSuccess) {
            printf("HIP asynchronous kernel error: %s\n", hipGetErrorString(errAsync));
            std::exit(1);
        }
    }
    

    HIP_CHECK(
    hipMemcpy(&S[0][0],
              flat_d_S,
              (size_t)n * (size_t)n * sizeof(short),
              hipMemcpyDeviceToHost)
    );
    
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

      S_CPU[i][j] = max(S_CPU[i][j], S_CPU[i+1][j-1] + _paired(seqq[i],seqq[j]));

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

  HIP_CHECK(hipFree(d_S));
  HIP_CHECK(hipFree(flat_d_S));
        
    return 0;
}
