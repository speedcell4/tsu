from aku import Aku

from tsu.lse import run_lse

aku = Aku()


@aku.register
def hello():
    print('hello, world!')


aku.register(run_lse)

if __name__ == '__main__':
    aku.run()
