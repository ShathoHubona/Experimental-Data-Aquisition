import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
start, stop, n_vals = -2, 2, 800

x_vals = np.linspace(start, stop, n_vals)
y_vals = np.linspace(-1, 1, n_vals)
x, y = np.meshgrid(x_vals, y_vals)

N = 10

E0 = 1
dx = 0.1
dy = 0.1
lambdaa = 0.0006
I1 = 0
L = 500


for i in range(N):
    I1 += np.exp(((-1j*2*np.pi)/lambdaa)*(x*i*10))

I1f = np.abs(I1)**2

z = I1f*(E0**2)*((dx**2)*(dy**2)/((lambdaa**2)*(L**2))*((np.sinc((np.pi*x*dx)/(lambdaa*L)))**2)*((np.sinc((np.pi*y*dy)/(lambdaa*L)))**2))

cp = plt.contourf(x, y, z)
plt.colorbar(cp)


#ax.contour(X, Y, Z)

ax.set_title('Contour Plot')


plt.show()
