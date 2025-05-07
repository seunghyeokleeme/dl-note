import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [10, 5, 12, 14, 15, 16, 25, 18, 17]

a, b = 7/8, 73/8

x_line = np.linspace(0, 16, 100)
y_line = a * x_line + b

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 15)
plt.ylim(0, 30)

plt.scatter(x, y)
plt.plot(x_line, y_line, 'g-')
plt.vlines(2, 5, 10.875, color='red', linestyles='solid', linewidth=2)
plt.vlines(7, 15.25, 25, color='red', linestyles='solid', linewidth=2)
plt.show()
#

x_line = np.linspace(0.1, 100, 1000)
y_line = (y[2] - (x_line * x[2]))**2

plt.plot(x_line, y_line, 'r-', label ='(e_3)^2')
plt.plot(5, (y[2]-(5*x[2]))**2, 'bo')
plt.plot(6, (y[2]-(6*x[2]))**2, 'bo')
plt.fill_between([5, 6], [(y[2]-(5*x[2]))**2, (y[2]-(6*x[2]))**2], alpha=0.5)

plt.xlabel('a')
plt.ylabel('(e_3)^2')
plt.xlim(0, 10)
plt.ylim(0, 50)
plt.legend()

plt.show()

x_line = np.linspace(0.1, 100, 1000)
y_line = np.abs(y[2] - (x_line * x[2]))

plt.plot(x_line, y_line, 'r-', label ='|e_3|')
plt.plot(5, np.abs(y[2] - (5 * x[2])), 'bo')
plt.plot(6, np.abs(y[2] - (6 * x[2])), 'bo')
plt.fill_between([5, 6], [np.abs(y[2] - (5 * x[2])), np.abs(y[2] - (6 * x[2]))], alpha=0.5)

plt.xlabel('a')
plt.ylabel('|e_3|')
plt.xlim(0, 10)
plt.ylim(0, 15)
plt.legend()

plt.show()