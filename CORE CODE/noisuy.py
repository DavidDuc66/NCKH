#import epyt
from epyt import epanet
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
d = epanet('FILEGOCCUADUY.inp') 

#xây dựng tọa độ và cao độ
x_values = d.getNodeCoordinates('x')
x_array = np.array(list(x_values.values()))
y_values = d.getNodeCoordinates('y')
y_array = np.array(list(y_values.values()))
x_array=x_array.reshape(-1, 1)
y_array=y_array.reshape(-1, 1)
xmin=np.min(x_array)
xmax=np.max(x_array)
ymin=np.min(y_array)
ymax=np.max(y_array)
dx=xmax-xmin
dy=ymax-ymin
xleft=xmin-dx/2
ydown=ymin-dy/2
xright=xmax+dx/2
yup=ymax+dy/2
A=np.array([xleft, yup])
B=np.array([xright, yup])
C=np.array([xright, ydown])
D=np.array([xleft, ydown])
E=np.array([607009.524, 1191846.24])

points = np.array([A, B, C, D, E])  # Các tọa độ x, y của điểm biên
values = np.array([50, 185.185, 50, 200.200, 207])  # Giá trị cao độ tại các điểm biên

# Các điểm bạn cần nội suy
SN=d.getNodeCount()
toado = np.empty((SN, 0))
toado = np.concatenate((x_array,y_array ), axis=1);
caodo = d.getNodeElevations()
points_need_interpolation =toado

# Nội suy giá trị cao độ tại các điểm cần nội suy
interpolated_values = griddata(points, values, points_need_interpolation, method='cubic')

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Tọa độ và cao độ của 5 điểm
points = points_need_interpolation  # Các tọa độ x, y
values = interpolated_values  # Giá trị cao độ tại các điểm

# Xác định biên của trục x và y
x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

# Mở rộng biên thêm một khoảng
x_range = x_max - x_min
y_range = y_max - y_min

x_min -= 0.1 * x_range
x_max += 0.1 * x_range
y_min -= 0.1 * y_range
y_max += 0.1 * y_range 

# Tạo lưới mới để nội suy cao độ
grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]  # Tạo grid với kích thước 100x100

# Nội suy giá trị cao độ cho grid
grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

# Vẽ bề mặt 3D
fig = plt.figure(figsize=(12, 8))  # Điều chỉnh kích thước hình ảnh bằng cách sử dụng figsize
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='k')

# Hiển thị các điểm gốc
ax.scatter(points[:, 0], points[:, 1], values, color='red', s=30, label='Điểm gốc')

# Thêm tiêu đề và nhãn
ax.set_title('Nội suy cao độ với 5 điểm và biên mở rộng')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Cao độ')

plt.legend()
plt.show()

# # Tạo lưới mới để nội suy cao độ
# grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]  # Tạo grid với kích thước 100x100

# # Nội suy giá trị cao độ cho grid
# grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

# # Vẽ bề mặt 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='k')

# # Hiển thị các điểm gốc
# ax.scatter(points[:, 0], points[:, 1], values, color='red', s=30, label='Điểm gốc')

# # Thêm tiêu đề và nhãn
# ax.set_title('Nội suy cao độ với 5 điểm và biên mở rộng')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Cao độ')

# plt.legend()
# plt.show()
##############################################################################################

# #Hiển thị kết quả trên đồ thị
# plt.figure(figsize=(8, 6))
# plt.imshow(griddata(points, values, (np.mgrid[xleft:xright:100j, ydown:yup:100j]), method='linear').T, 
#            extent=(xleft, xright, ydown, yup), origin='lower', cmap='viridis')
# plt.scatter(points[:, 0], points[:, 1], c=values, cmap='viridis', edgecolor='black', s=100, label='Điểm biên')
# plt.scatter(points_need_interpolation[:, 0], points_need_interpolation[:, 1], 
#             c=interpolated_values, cmap='viridis', edgecolor='blue', s=50, label='Điểm nội suy')
# plt.colorbar(label='Cao độ')
# plt.legend()
# plt.title('Nội suy cao độ từ điểm biên')
# plt.show()

