---
uid: 20230418220752
title: 图像处理入门
tags: []
---

# 图像处理入门

## 图像数据读取展示

```python
import cv2 #opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

img=cv2.imread('cat.jpg')
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)       #任意键退出
    cv2.destroyAllWindows()

#保存
cv2.imwrite('mycat.png',img)

```

## 视频读取与展示

- cv2.VideoCapture 可以捕获摄像头，用数字来控制不同的设备，例如 0,1。
- 如果是视频文件，直接指定好路径即可。

```python
video = cv2.VideoCapture('test.mp4')
while video.isOpened():
    ret, frame = video.read()
    if frame is None:
        break
    if ret == True:
        # 将每一帧转化成灰度图
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
       	#每一帧等待100ms或者按退出键退出
        if cv2.waitKey(100) & 0xFF == 27:
            break
video.release()
cv2.destroyAllWindows()
```

## 图像操作

### 截取图像部分数据

```python
#截取图像[纵，横]
img=cv2.imread('cat.jpg')
cat=img[0:50,0:200]
cv_show('cat',cat)
```

### 颜色通道提取

```python
# 将img的颜色提取出来形成一个数组
b,g,r=cv2.split(img)
# 将三个颜色通道合成图像
img=cv2.merge((b,g,r))
```

### 边界填充

- BORDER_REPLICATE：复制法，也就是复制最边缘像素。
- BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制
- BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
- BORDER_WRAP：外包装法
- BORDER_CONSTANT：常量法，常数值填充。

```python
top_size,bottom_size,left_size,right_size = (50,50,50,50)
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
```

![](https://www.windilycloud.cn/img/202201132347909.png)

![](https://www.windilycloud.cn/img/202201132347856.png)

### 数值计算

作用于各个像素的颜色通道

### 图像融合

只能同样大小的图片才能融合，按透明度权重拼成一张图片![](https://www.windilycloud.cn/img/202201132347489.png)

```python
img_dog = cv2.resize(img_dog, (500, 414))
res = cv2.addWeighted(img_cat, 0.4, img_dog, 0.6, 0)
#按倍数更改大小
res = cv2.resize(img, (0, 0), fx=4, fy=4)
```

# 图像基本处理

## 灰度图

```python
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img_gray.shape
```

## HSV

- H - 色调（主波长）。
- S - 饱和度（纯度/颜色的阴影）。
- V 值（强度）

```python
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
```

## 图像阈值

#### ret, dst = cv2.threshold(src, thresh, maxval, type)

- src： 输入图，只能输入单通道图像，通常来说为灰度图
- dst： 输出图
- thresh： 阈值
- maxval： 当像素值超过了阈值（或者小于阈值，根据 type 来决定），所赋予的值
- type：二值化操作的类型，包含以下 5 种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
- cv2.THRESH_BINARY 超过阈值部分取 maxval（最大值），否则取 0
- cv2.THRESH_BINARY_INV THRESH_BINARY 的反转
- cv2.THRESH_TRUNC 大于阈值部分设为阈值，否则不变
- cv2.THRESH_TOZERO 大于阈值部分不改变，否则设为 0
- cv2.THRESH_TOZERO_INV THRESH_TOZERO 的反转

```python
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
```

![](https://www.windilycloud.cn/img/202201132347476.png)

## 图像平滑

用一个 3\*3 的核来处理图像,res = np.hstack((dilate_1,dilate_2,dilate_3)) 用来将图像相减表现变化趋势

### 均值滤波

```
blur = cv2.blur(img, (3, 3))
```

### 方框滤波

```
box = cv2.boxFilter(img,-1,(3,3), normalize=True)
```

### 高斯滤波

```
aussian = cv2.GaussianBlur(img, (5, 5), 1)
```

### 中值滤波

```
median = cv2.medianBlur(img, 5)
```

## 形态学

### 腐蚀操作

```
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
```

### 膨胀操作

```
kernel = np.ones((3,3),np.uint8)
dige_dilate = cv2.dilate(img,kernel,iterations = 1)
```

### 开运算与闭运算

- 开：先腐蚀，再膨胀

- 闭：先膨胀，再腐蚀

```
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

### 梯度运算

先膨胀，再腐蚀，二者相减得梯度

```
dilate = cv2.dilate(pie,kernel,iterations = 5)
erosion = cv2.erode(pie,kernel,iterations = 5)
res = np.hstack((dilate,erosion))

gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
```

### 顶帽与黑帽

- 顶帽 = 原始输入 - 开运算结果
- 黑帽 = 闭运算 - 原始输入

```
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackhat  = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, kernel)
```

### 算子

算子也是用来算梯度的

- sobel 算子

  dst = cv2.Sobel(src, ddepth, dx, dy, ksize)

  - ddepth: 图像的深度
  - dx 和 dy 分别表示水平和竖直方向
  - ksize 是 Sobel 算子的大小

  ```python
  #先读取为灰度图
  img = cv2.imread('pie.png',cv2.IMREAD_GRAYSCALE)
  #分别计算x和y方向的梯度再求和
  sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
  sobelx = cv2.convertScaleAbs(sobelx)
  cv_show(sobelx,'sobelx')

  sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
  sobely = cv2.convertScaleAbs(sobely)
  cv_show(sobely,'sobely')

  sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
  cv_show(sobelxy,'sobelxy')

  #不建议同时求x和y方向的梯度，处理得没分别处理的好
  sobelxy=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
  sobelxy = cv2.convertScaleAbs(sobelxy)
  cv_show(sobelxy,'sobelxy')
  ```

- scharr 算子

- laplacian 算子

- 不同算子的差异

  ```python
  #不同算子的差异
  img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
  sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
  sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
  sobelx = cv2.convertScaleAbs(sobelx)
  sobely = cv2.convertScaleAbs(sobely)
  sobelxy =  cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

  scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
  scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
  scharrx = cv2.convertScaleAbs(scharrx)
  scharry = cv2.convertScaleAbs(scharry)
  scharrxy =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0)

  laplacian = cv2.Laplacian(img,cv2.CV_64F)
  laplacian = cv2.convertScaleAbs(laplacian)

  res = np.hstack((sobelxy,scharrxy,laplacian))
  cv_show(res,'res')
  ```

### 边缘检测

- 1. 使用高斯滤波器，以平滑图像，滤除噪声。
- 2. 计算图像中每个像素点的梯度强度和方向。
- 3. 应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
- 4. 应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
- 5. 通过抑制孤立的弱边缘最终完成边缘检测。

高斯滤波器

![](https://www.windilycloud.cn/img/202201132347245.png)

计算梯度和方向

![](https://www.windilycloud.cn/img/202201132347758.png)

非极大值抑制

![](https://www.windilycloud.cn/img/202201132347379.png)

![](https://www.windilycloud.cn/img/202201132347690.png)

双阈值检测

![](https://www.windilycloud.cn/img/202201132347421.png)

```python
img=cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)

v1=cv2.Canny(img,80,150)
v2=cv2.Canny(img,50,100)

res = np.hstack((v1,v2))
cv_show(res,'res')
```

### 图像金字塔

向下采样方法

![](https://www.windilycloud.cn/img/202201132347332.png)

向上采样方法

![](https://www.windilycloud.cn/img/202201132347012.png)

```
up=cv2.pyrUp(img)
cv_show(up,'up')
down=cv2.pyrDown(img)
cv_show(down,'down')
```

### 图像轮廓

```
findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
```

mode: 轮廓检索模式

- RETR_EXTERNAL ：只检索最外面的轮廓；
- RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
- RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
- RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次;

method: 轮廓逼近方法

- CHAIN_APPROX_NONE：以 Freeman 链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
- CHAIN_APPROX_SIMPLE: 压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分

为了更高的准确率，使用二值图像。

绘制轮廓

```python
img = cv2.imread('contours.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv_show(thresh,'thresh')

contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
# 注意需要copy,要不原图会变。。。
draw_img = img.copy()
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
cv_show(res,'res')
```

![](https://www.windilycloud.cn/img/202201132347592.png)

这轮廓用来找物体应该易于和环境分开才行

轮廓特征 (对于那种形不成必和轮廓的没有效果)

```python
cnt = contours[0]
#面积
cv2.contourArea(cnt)
#周长，True表示闭合的
cv2.arcLength(cnt,True)
```

轮廓近似

```
approxPolyDP(curve, epsilon, closed[, approxCurve]) -> approxCurve
```

```python
img = cv2.imread('contours2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

draw_img = img.copy()
res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)
cv_show(res,'res')

epsilon = 0.15*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
cv_show(res,'res')
```

边界矩形

```python
img = cv2.imread('contours.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv_show(img,'img')

area = cv2.contourArea(cnt)
x, y, w, h = cv2.boundingRect(cnt)
rect_area = w * h
extent = float(area) / rect_area
print ('轮廓面积与边界矩形比',extent)
```

外接圆

```python
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,(0,255,0),2)
cv_show(img,'img')
```

### 傅里叶变换

作用：

- 高频：变化剧烈的灰度分量，例如边界
- 低频：变化缓慢的灰度分量，例如一片大海
- 低通滤波器：只保留低频，会使得图像模糊
- 高通滤波器：只保留高频，会使得图像细节增强

运用：

- opencv 中主要就是 cv2.dft() 和 cv2.idft()，输入图像需要先转换成 np.float32 格式。
- 得到的结果中频率为 0 的部分会在左上角，通常要转换到中心位置，可以通过 shift 变换来实现。
- cv2.dft() 返回的结果是双通道的（实部，虚部），通常还需要转换成图像格式才能展示（0,255）。

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg',0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
# 得到灰度图能表示的形式
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg',0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)     # 中心位置

# 低通滤波
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# IDFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()
```

![](https://www.windilycloud.cn/img/202201132347982.png)

```python
img = cv2.imread('lena.jpg',0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)     # 中心位置

# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0

# IDFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()
```

![](https://www.windilycloud.cn/img/202201132348638.png)
