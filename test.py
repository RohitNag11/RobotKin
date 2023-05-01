import numpy as np

theta = 0.5


def R(t): return np.array([[np.cos(t), -np.sin(t), 0],
                           [np.sin(t), np.cos(t), 0],
                           [0, 0, 1]])


# print(R(theta))

# theta = 2

# print(R(theta))


class MyClass:
    def __init__(self, theta):
        self.theta = theta

    @property
    def R(self):
        return np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                         [np.sin(self.theta), np.cos(self.theta), 0],
                         [0, 0, 1]])


obj = MyClass(1)
print(obj.R)

obj.theta = 2
print(obj.R)
