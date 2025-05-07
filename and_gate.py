import matplotlib.pyplot as plt
import numpy as np

# 시각화
plt.figure(figsize=(8, 6))

# 데이터 포인트 그리기
x0 = [0, 0, 1]  # x 좌표
y0 = [0, 1, 0]  # y 좌표
x1 = [1]        # x 좌표
y1 = [1]        # y 좌표

plt.plot(x0, y0, 'ro')  # 0인 점들 (빨간색)
plt.plot(x1, y1, 'bo')  # 1인 점 (파란색)

# 결정 경계선 그리기
w1, w2, theta = 0.5, 0.5, 0.7
x_line = np.linspace(-0.5, 1.5, 100)
y_line = (theta - w1 * x_line) / w2

plt.plot(x_line, y_line, 'g-', label='Decision Boundary')

# 그래프 꾸미기
plt.grid(True)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('AND Gate Visualization (w1 = 0.5, w2 = 0.5, theta = 0.7)')
plt.legend()
plt.show()