from numpy import *


def compute_error_for_given_points(b,m,points):
    # compute error for given points
    sum_of_error = 0;
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]

        sum_of_error += (y - ((m*x) + b)) ** 2

    sum_of_error = sum_of_error/float(len(points))
    return sum_of_error


def step_gradient(current_b,current_m,learning_rate,points):
    # for the complete array of points update bias and slope
    # partial derivative of 1/N Sigma((y-(mx+b))**2)
    # with m = 2/N Sigma(x(y-(mx+b)))
    # with b = 2/N Sigma((y-(mx+b)))

    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2/N) * (y - ((current_m * x) + current_b))
        m_gradient += -(2/N) * x * (y - ((current_m * x) + current_b))

    b = current_b - (learning_rate * b_gradient)
    m = current_m - (learning_rate * m_gradient)

    return [b,m]



def gradient_descent_runner(points,learning_rate,initial_b,initial_m,num_iterations):
    b = initial_b
    m = initial_m
    for i in range(num_iterations):
        b,m = step_gradient(b,m,learning_rate,array(points))

    return [b,m]


def run():
    points = genfromtxt('data.csv',delimiter=',')
    # learning rate is the hyper parameter alpha
    learning_rate = 0.0001

    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    [b,m] = gradient_descent_runner(points,learning_rate,initial_b,initial_m,num_iterations)
    print b
    print m
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_given_points(b, m, points))

run()
