from aku import Aku

aku = Aku()


@aku.register
def hello():
    print('hello, world!')


if __name__ == '__main__':
    aku.run()
