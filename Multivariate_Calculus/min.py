import numpy as np
from scipy import optimize

def f(x,y):
    return np.exp(-(2*x*x + y*y - x*y)/2)

def g(x,y):
    return x*x + 3*(y+1)**2 - 1

def dfdx(x,y):
    return 0.5*(-4*x + y)*f(x,y)

def dfdy(x,y):
    return 0.5*(-2*y + x)*f(x,y)

def dgdx(x,y):
    return 2*x

def dgdy(x,y):
    return 6*(y+1)

def DL(vars):
    x,y,lmb = vars
    return np.array([
        dfdx(x,y) - lmb*dgdx(x,y),
        dfdy(x,y) - lmb*dgdy(x,y),
        -g(x,y)
    ])

# 產生一圈初始值
ts = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
inits = [(np.cos(t), -1 + (1/np.sqrt(3))*np.sin(t), 0.0) for t in ts]

sols = []
for guess in inits:
    sol = optimize.root(DL, guess).x
    # 去重（四捨五入避免重複）
    key = tuple(np.round(sol, 6))
    if key not in {tuple(np.round(s,6)) for s in sols}:
        sols.append(sol)

# 列出四個駐點，找 f 最小者
vals = [(x, y, f(x,y)) for x,y,_ in sols]
vals_sorted = sorted(vals, key=lambda t: t[2])  # 依 f 值由小到大
for x,y,fx in vals_sorted:
    print(f"x={x:.6f}, y={y:.6f}, f={fx:.6f}")

best_x = vals_sorted[0][0]
print("\nGlobal minimum x =", round(best_x, 2))
