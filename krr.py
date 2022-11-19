import argparse
parser = argparse.ArgumentParser(description='Traing gnn')
parser.add_argument('--gpu','-g',dest='gpu',default=0)
args = parser.parse_args()
print(args)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import cupy
import gc
from cuml import KernelRidge

def run():
    M,N = 20000,1000
    S = 10000000
    x = cupy.random.rand(M,N)
    y = cupy.random.rand(M)
    for _ in range(S):
        for _ in range(S):
            krr = KernelRidge(kernel='rbf')
            krr.fit(x,y)
            del krr
            gc.collect()

if __name__ == '__main__':
    run()
