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
    # print ("x5:", x5)
    x7 = g2(x5)
    # print("x7:", x7)
    x8 = g1(x7)
    return x0, x1, x2, x3, x4, x5, x6, x7, x8

def calc_x8_speed():
    return 1

def calc_x7_speed(x8_speed, x7):
    return x8_speed * x7 * 2

def calc_x5_speed(x7_speed):
    return x7_speed

def calc_x4_speed(x5_speed, x6):
    return x5_speed * x6

def cal_x6_speed(x5_speed, x4):
    return x5_speed * x4

def calc_x0_speed(x4_speed):
    return x4_speed

def calc_x1_speed(x4_speed):
    return x4_speed

def calc_x2_speed(x6_speed):
    return x6_speed

def calc_x3_speed(x6_speed):
    return x6_speed

print(distance(x0, x1, x2, x3)[-1])
little_step = 0.000000001

for i in range(100000):

    
    x0, x1, x2, x3, x4, x5, x6, x7, x8 = distance(x0, x1, x2, x3)

    x8_speed = calc_x8_speed()
    x7_speed = calc_x7_speed(x8_speed, x7)
    x5_speed = calc_x5_speed(x7_speed)
    x4_speed = calc_x4_speed(x5_speed, x6)
    x6_speed = cal_x6_speed(x5_speed, x4)
    x0_speed = calc_x0_speed(x4_speed)
    x1_speed = calc_x1_speed(x4_speed)
    x2_speed = calc_x2_speed(x6_speed)
    x3_speed = calc_x3_speed(x6_speed)

    # print ("x8_speed : ", x8_speed)
    # print ("x7_speed : ", x7_speed)
    # print ("x5_speed : ", x5_speed)
    # print ("x4_speed : ", x4_speed)
    # print ("x6_speed : ", x6_speed)
    # print ("x0_speed : ", x0_speed)
    # print ("x1_speed : ", x1_speed)
    # print ("x2_speed : ", x2_speed)
    # print ("x3_speed : ", x3_speed)

    x0 += -x0_speed * little_step
    x1 += -x1_speed * little_step
    x2 += -x2_speed * little_step
    x3 += -x3_speed * little_step

    # print ("x0 : ", x0)
    # print ("x1 : ", x1)
    # print("x2 : ", x2)
    # print("x3 : ", x3)

    print("x8 : ", x8)

print(x0, x1, x2, x3)
x0, x1, x2, x3, x4, x5, x6, x7, x8 = distance(x0, x1, x2, x3)
print("x5 : ", x5)