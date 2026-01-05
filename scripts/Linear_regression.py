import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X=np.linspace(0,100,50,True,False)
Y=X*2+3+np.random.randn(50)*2

#plt.scatter(X,Y,label="data")
#plt.legend()#图例增加，与上方的label对应，函数的参数添加位置就行，每个数据链都可以添加！！！#
#plt.show()

w = np.random.randn()
b = np.random.randn()
lr = 0.00001  # 学习率/太大会导致梯度爆炸
epochs = 1000

for epoch in range(epochs):
    # 前向计算 y_pred
    Y_pred = w * X + b
    # 计算损失 (均方误差)
    loss = np.mean((Y - Y_pred) ** 2)
    # 计算梯度
    dw = -2 * np.mean((Y - Y_pred) * X)
    db = -2 * np.mean(Y - Y_pred)
    # 更新参数
    w -= lr * dw
    b -= lr * db
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss={loss:.4f}, w={w:.2f}, b={b:.2f}")

plt.scatter(X,Y,label="original data")
plt.plot(X,w*X+b,color="red",label="fitted line")
plt.legend()
plt.show()
