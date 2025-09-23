from aku import Aku

from tsu.gemm import run_gemm

aku = Aku()


@aku.register
def hello():
    print('hello, world!')


aku.register(run_gemm)

if __name__ == '__main__':
    aku.run()
