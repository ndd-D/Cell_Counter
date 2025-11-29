import cv2
import numpy as np

img = cv2.imread('image.png')

'''
    原图-灰度图-二值化-开运算-膨胀（背景图）-距离变换-
    前景图-不确定区域-标记前、背、不确定-分水岭开始-标记分割
'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0)
_, threshold = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernal = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernal, 2)

bg = cv2.dilate(opening, kernal, 5)

distance = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

_, fg = cv2.threshold(distance, 0.2* distance.max(), 255, cv2.THRESH_BINARY)

fg = np.uint8(fg)
unsure = cv2.subtract(bg, fg)

ret, markers = cv2.connectedComponents(fg)
markers += 1
markers[unsure == 255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

# ========== 修正后的计数和标记 ==========

# 直接分析分水岭后的 markers
unique_labels = np.unique(markers)
print("所有标签值:", unique_labels)

# 分水岭算法后的标记含义：
# -1: 边界
# 0: 背景（在分水岭中通常不会被保留）
# 1: 原始背景（但分水岭会修改）
# >1: 不同的对象区域

# 正确统计方法：只统计大于1的标签（对象区域）
object_labels = unique_labels[unique_labels > 1]
object_count = len(object_labels)

print(f"检测到的对象数量: {object_count}")
print(f"对象标签: {object_labels}")

# # 创建彩色标记图像
colored_markers = np.zeros_like(img)

# 为每个对象分配不同颜色并标记中心点
for i, label in enumerate(object_labels):
    # 生成随机颜色
    color = np.random.randint(50, 255, 3).tolist()
    
    # 给该标签区域上色
    mask = (markers == label)
    colored_markers[mask] = color

    # 计算并标记对象中心
    y_coords, x_coords = np.where(mask)
    if len(x_coords) > 0 and len(y_coords) > 0:
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        # 在彩色标记图上画中心点和编号
        cv2.circle(colored_markers, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.putText(colored_markers, str(i+1), (center_x-5, center_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(colored_markers, str(mask.sum()), (center_x-5, center_y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # 在原始图像上也标记编号
        cv2.putText(img, str(i+1), (center_x-5, center_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# 标记边界为红色（保持原样）
boundary_mask = (markers == -1)
colored_markers[boundary_mask] = [0, 0, 255]
cv2.imwrite("result2.png",img)
# 显示结果
cv2.imshow('threshold', threshold)
# cv2.imshow('opening', opening)
# cv2.imshow('bg', bg)
# cv2.imshow('fg', fg)
# cv2.imshow('unsure', unsure)
cv2.imshow('Original Image with Markers', img)
cv2.imshow('Colored Regions', colored_markers)

print("按任意键关闭窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()
