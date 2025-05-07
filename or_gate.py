import matplotlib.pyplot as plt
import numpy as np

def OR_Gate(x1:int, x2:int) -> int:
    w1, w2, theta = 0.5, 0.5, 0.3
    temp = x1*w1 + x2*w2
    if temp > theta:
        return 1
    else:
        return 0

# OR 게이트 입력 데이터 생성
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([OR_Gate(x[i][0], x[i][1]) for i in range(len(x))])

# 시각화
plt.figure(figsize=(8, 6))

# 데이터 포인트 그리기
for i in range(len(x)):
    if y[i] == 0:
        plt.plot(x[i][0], x[i][1], 'ro', markersize=10, label='0' if i == 0 else "")
    else:
        plt.plot(x[i][0], x[i][1], 'bo', markersize=10, label='1' if i == 1 else "")

# 결정 경계선 그리기
w1, w2, theta = 0.5, 0.5, 0.3
x_line = np.linspace(-0.5, 1.5, 100)
y_line = (theta - w1 * x_line) / w2
y_line_2 = (0.6 - w1 * x_line) / w2

plt.plot(x_line, y_line, 'g-', label='Decision Boundary')
plt.plot(x_line, y_line_2, 'r-', label='Wrong Decision Boundary')

# 그래프 꾸미기
plt.grid(True)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('OR Gate Visualization (w1 = 0.5, w2 = 0.5, theta = 0.3)')
plt.legend()
plt.show()
