from aku import Aku

from tsu.flash_attn import flash_attn

aku = Aku()


@aku.register
def hello():
    print('hello, world!')


aku.register(flash_attn)

if __name__ == '__main__':
    aku.run()
