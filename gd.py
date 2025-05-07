import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fn1(x, y):
    return x**2 + y**2;

# NumPy 라이브러리를 사용한 편미분 계산
def gradient(f, point, h=1e-5):
    """
    함수의 그래디언트(기울기 벡터)를 계산하는 함수
    
    Parameters:
    f: 미분할 함수
    point: 미분을 계산할 지점의 좌표 (x, y)
    h: 미분 계산에 사용할 작은 값
    
    Returns:
    그래디언트 벡터
    """
    grad = np.zeros_like(point, dtype=float)
    
    for i in range(len(point)):
        # 각 변수에 대해 편미분 계산
        point_plus_h = point.copy()
        point_plus_h[i] += h
        
        # 중앙 차분법(central difference)을 사용하여 더 정확한 미분값 계산
        point_minus_h = point.copy()
        point_minus_h[i] -= h
        
        grad[i] = (f(*point_plus_h) - f(*point_minus_h)) / (2 * h)
    
    return grad

def gradient_descent(start_point, learning_rate=0.1, num_iterations=100):
    """
    경사 하강법을 수행하는 함수
    
    Parameters:
    start_point: 시작점 (x, y)
    learning_rate: 학습률
    num_iterations: 반복 횟수
    
    Returns:
    경로의 좌표들
    """
    point = np.array(start_point, dtype=float)
    path = [point.copy()]
    
    for _ in range(num_iterations):
        grad = gradient(fn1, point)
        point = point - learning_rate * grad
        path.append(point.copy())
    
    return np.array(path)

def plot_3d_surface():
    """
    x^2 + y^2 함수의 3D 표면과 경사 하강법의 경로를 그리는 함수
    """
    # x, y 좌표 생성
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # z 값 계산
    Z = fn1(X, Y)
    
    # 3D 그래프 생성
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 표면 플롯 생성
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
    
    # 경사 하강법 수행
    start_point = [1, 1]  # 시작점
    path = gradient_descent(start_point, learning_rate=0.1, num_iterations=5)
    
    # 경로 그리기
    path_x = path[:, 0]
    path_y = path[:, 1]
    path_z = fn1(path_x, path_y)
    
    # 시작점과 경로 표시
    ax.plot(path_x, path_y, path_z, 'r-', linewidth=2, label='Gradient Descent Path')
    ax.scatter(path_x[0], path_y[0], path_z[0], color='red', s=100, label='Start Point')
    ax.scatter(path_x[-1], path_y[-1], path_z[-1], color='green', s=100, label='End Point')
    
    # 레이블 추가
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Loss')
    ax.set_title('Gradient Descent on x^2 + y^2')
    
    # 범례 추가
    ax.legend()
    
    plt.show()

# 3D 그래프 그리기
if __name__ == "__main__":
    plot_3d_surface()
