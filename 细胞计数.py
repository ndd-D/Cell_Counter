import cv2
import numpy as np

img=cv2.imread('image.png')

'''
    原图-灰度图-二值化-开运算-膨胀（背景图）-距离变换-
    前景图-不确定区域-标记前、背、不确定-分水岭开始-标记分割
'''
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


_,threshold=cv2.threshold(gray,80,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


kernal=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(threshold,cv2.MORPH_OPEN,kernal,2)


bg=cv2.dilate(opening,kernal,2)


distance=cv2.distanceTransform(opening,cv2.DIST_L2,3)


_,fg=cv2.threshold(distance,0.2*distance.max(),255,cv2.THRESH_BINARY)


fg=np.uint8(fg)
unsure=cv2.subtract(bg,fg)

ret,markers=cv2.connectedComponents(fg)
markers+=1
markers[unsure==255]=0


markers=cv2.watershed(img,markers)
img[markers==-1]=[255,0,0]

areas=[]
contours, h = cv2.findContours(bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    # cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
    area = cv2.contourArea(cnt)
    areas.append(area)
print(areas)
areas = np.array(areas)
# 计算分位数（例如：25%、50%、75%、90%分位数）
q25, q50, q75, q95 = np.percentile(areas, [25, 50, 75, 95])
noise_threshold=q25
adhere_threshold=q95
print(areas)
print(f"noise_threshold: {noise_threshold}, adhere_threshold: {adhere_threshold}")

count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < noise_threshold:
        continue
    elif area > adhere_threshold:
        print("two")
        # 记录当前计数（标注用）
        current_count = count + 1  # 第一个粘连细胞的数字
        count += 2  # 计数+2（表示2个细胞）
        # 计算轮廓中心
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # 标注粘连细胞（可以标两个数字，或用特殊符号）
        cv2.putText(img, str(current_count), (cX-15, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(img, str(current_count+1), (cX+15, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    else:
        count += 1
        # 单个细胞标注
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(img, str(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    #标注面积
    # cv2.putText(img, f"{area:.2f}", (cX, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (256,256,0), 1)
print(count)
# cv2.imwrite("result.bmp",img)
while True:
    cv2.imshow('img',img)
    cv2.imshow('threshold',threshold)
    cv2.imshow('bg',bg)
    cv2.imshow('fg',fg)
#     cv2.imshow('unsure',unsure)
#     cv2.imshow('distance',distance)

    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()