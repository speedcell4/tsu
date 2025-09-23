from aku import Aku

from tsu.log_mm import run_log_mm

aku = Aku()


@aku.register
def hello():
    print('hello, world!')


aku.register(run_log_mm)

if __name__ == '__main__':
    aku.run()
