from aku import Aku

from tsu.lse import lse

aku = Aku()


@aku.register
def hello():
    print('hello, world!')


aku.register(lse)

if __name__ == '__main__':
    aku.run()
