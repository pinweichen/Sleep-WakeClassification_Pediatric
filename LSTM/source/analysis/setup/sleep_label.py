from enum import Enum


class SleepWakeLabel(Enum):
    wake = 0
    sleep = 1


class ThreeClassLabel(Enum):
    wake = 0
    nrem = 1
    rem = 2


class MultiClassLabel(Enum):
    wake = 0
    n1 = 1
    n2 = 2
    n3 = 3
    r = 4


class FourClassLabel(Enum):
    wake = 0
    light = 1
    deep = 2
    rem = 3

