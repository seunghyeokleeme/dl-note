import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

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

def gradient_descent(start_point, learning_rate=1, num_iterations=100):
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
    x^2 + y^2 함수의 3D 표면과 경사 하강법의 경로를 애니메이션으로 그리는 함수
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
    start_point = [4, 4]  # 시작점
    path = gradient_descent(start_point, learning_rate=0.01)
    
    # 경로 데이터
    path_x = path[:, 0]
    path_y = path[:, 1]
    path_z = fn1(path_x, path_y)
    
    # 시작점 표시
    start_point_plot = ax.scatter([], [], [], color='red', s=100, label='Current Point')
    path_line, = ax.plot([], [], [], 'ro-', linewidth=2, label='Gradient Descent Path')
    
    # 레이블 추가
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Loss')
    ax.set_title('Gradient Descent Animation (learning rate = 0.01)')
    
    # 그래프 범위 설정
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 50)
    
    # 범례 추가
    ax.legend()
    
    def init():
        start_point_plot._offsets3d = ([], [], [])
        path_line.set_data([], [])
        path_line.set_3d_properties([])
        return start_point_plot, path_line
    
    def animate(i):
        # 현재까지의 경로 표시
        path_line.set_data(path_x[:i+1], path_y[:i+1])
        path_line.set_3d_properties(path_z[:i+1])
        
        # 현재 위치 표시
        start_point_plot._offsets3d = ([path_x[i]], [path_y[i]], [path_z[i]])
        
        return start_point_plot, path_line
    
    # 애니메이션 생성
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(path), interval=100,
        blit=True
    )
    
    # GIF로 저장
    anim.save('gradient_descent_3.gif', writer='pillow', fps=10)
    
    plt.show()

# 3D 그래프 그리기
if __name__ == "__main__":
    plot_3d_surface()
