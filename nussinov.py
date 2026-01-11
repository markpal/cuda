import numpy as np
from numba import cuda
from numba import int16
import time

# to remove in future
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

BLOCK_SIZE = 32
N = 128

# -------------------------
def random_rna_seq_ascii(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    seq = np.random.choice(np.array([b'A', b'C', b'G', b'U']), size=N)
    return seq.view(np.uint8)
# -------------------------

# -------------------------
# Kernel CUDA
# -------------------------
@cuda.jit(device=True)
def _paired(a, b):
    if ((a == ord('A') and b == ord('U')) or
        (a == ord('U') and b == ord('A')) or
        (a == ord('C') and b == ord('G')) or
        (a == ord('G') and b == ord('C'))):
        return int16(1)
    return int16(0)
    
    
@cuda.jit
def myKernel(B, n, c0, seqq):
    c1 = cuda.blockIdx.x + c0
    bb = int16(BLOCK_SIZE)
    
    row = cuda.threadIdx.y
    col = cuda.threadIdx.x
    
    C = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=np.int16)
    A_elements = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=np.int16)
    B_elements = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=np.int16)
    
    C[row, col] = 0
    
    c1_max = min((N - 1) // bb, (N + c0 - 2) // bb)
    
    if c1 <= c1_max:
        _sj = c1 - c0
        _si = c1
        
        # non-problematic, can be par by max reduction in C[row, col]
        for m in range(_sj + 1, _si):
            A_elements[row, col] = B[bb * _sj + row, bb * m - 1 + col]
            B_elements[row, col] = B[bb * m + row, bb * _si + col]
        
            
            
            if row < bb and col < bb:
                Cvalue = int16(0)
                
                cuda.syncthreads()

                for e in range(bb):
                    tmp = int16(A_elements[row, e] + B_elements[e, col])
                    if tmp > Cvalue:
                        Cvalue = tmp
                        
                cuda.syncthreads()
                   
                old = C[row, col]
                if Cvalue > old:
                    C[row, col] = Cvalue
                    
        # problematic
        start = max(1, bb * c0 - bb - 1)
        end   = min(bb * c0 + bb - 1, N + bb * c0 - bb * c1 - 1)

        for c2 in range(start, end + 1):
            if c0 >= 1:
                lb = int16(max(bb * c1, -bb * c0 + bb * c1 + c2))
                ub = int16(min(min(N - 1, bb * c1 + bb - 1), -bb * c0 + bb * c1 + c2 + bb - 1))
                
                c3 = cuda.threadIdx.x + lb
                
                if c3 <= ub:
                    z = int16(B[-c2 + c3, c3])
                    
                    if cuda.threadIdx.y == 0:
                        _i = c3 % BLOCK_SIZE
                        _j = (c3 - c2) % BLOCK_SIZE
                        
                        for c4 in range(bb - 1):  # blocks 0 (triangles)
                            z = max(B[-c2 + c3, -c2 + c3 + c4] + B[-c2 + c3 + c4 + 1, c3], z)
            
                        z = z if z > C[_j, _i] else C[_j, _i]
                        
                        fragment = 1 if c1 == N // BLOCK_SIZE - 1 else 0
                        
                        for c4 in range(c2 - bb - fragment, c2):
                            z = max(B[-c2 + c3, -c2 + c3 + c4] + B[-c2 + c3 + c4 + 1, c3], z)
                            
                        B[-c2 + c3][c3] = max(z, B[-c2 + c3 + 1][c3 - 1] + _paired(seqq[-c2 + c3], seqq[c3]));
                        
            else:
                lb = bb * c1 + c2
                ub = min(N - 1, bb * c1 + bb - 1)
                
                c3 = cuda.threadIdx.x + lb
                
                if c3 <= ub:
                    z = B[-c2 + c3, c3]

                    for c4 in range(c2):
                        z = max(B[-c2 + c3, -c2 + c3 + c4] + B[-c2 + c3 + c4 + 1, c3], z)

                    B[-c2 + c3, c3] = max(z, B[-c2 + c3 + 1, c3 - 1] + _paired(seqq[-c2 + c3], seqq[c3]))

# END ------------------------------------------------------------------------------------------------------
                        
# -------------------------
# Kod hosta (CPU)
# -------------------------
def paired_host_ascii(a, b):
    return 1 if ((a == ord('A') and b == ord('U')) or
                 (a == ord('U') and b == ord('A')) or
                 (a == ord('C') and b == ord('G')) or
                 (a == ord('G') and b == ord('C'))) else 0


seq = random_rna_seq_ascii(N)
print(bytes(seq).decode("ascii"))

S = np.zeros((N, N), dtype=np.int16)
S_CPU = np.zeros((N, N), dtype=np.int16)


t0 = time.perf_counter()
# kopiowanie do GPU
dS = cuda.to_device(S)
dseq = cuda.to_device(seq)

numBlocks = N // BLOCK_SIZE
bb = BLOCK_SIZE

print("Blocks: " + str(numBlocks) + ", BB: " + str(BLOCK_SIZE))

dimBlock = (BLOCK_SIZE, BLOCK_SIZE)

for c0 in range((N - 1) // bb + 1):
    numBlocks = min((N - 1) // bb, (N + c0 - 2) // bb) - c0 + 1
    
    myKernel[(numBlocks,), dimBlock](dS, N, c0, dseq)
    cuda.synchronize()
    
dS.copy_to_host(S)

t1 = time.perf_counter()
print("czas [s] =", t1 - t0)

for i in range(N - 1, -1, -1):
    for j in range(i + 1, N):
        for k in range(0, j - i):
            S_CPU[i, j] = max(S_CPU[i, k + i] + S_CPU[k + i + 1, j], S_CPU[i, j])

        S_CPU[i, j] = max(S_CPU[i, j], S_CPU[i + 1, j - 1] + paired_host_ascii(seq[i], seq[j]))


for i in range(N):
    for j in range(N):
        if S[i, j] != S_CPU[i, j]:
            print(i, j, ":", S[i, j], S_CPU[i, j])
            print("error")
            raise SystemExit(1)



