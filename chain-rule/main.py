x0 = 100
x1 = 100
x2 = 1.1
x3 = 0.7
x = 90

def g1(x7):
    return x7*x7

def g2(x5):
    return x5 - x

def g3(x4, x6):
    return x4 * x6

def g4(x0, x1):
    return x0 + x1

def g5(x2, x3):
    return x2 + x3

def distance(x0, x1, x2, x3):
    x4 = g5(x0, x1)
    x6 = g4(x2, x3)
    x5 = g3(x4, x6)
    print ("x5:", x5)
    x7 = g2(x5)
    print("x7:", x7)
    x8 = g1(x7)
    return x8


print(distance(x0, x1, x2, x3))