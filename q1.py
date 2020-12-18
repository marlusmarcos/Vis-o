'''
Aluno: Marlus Marcos
este programa funciona com quadrados sem variação de angulos


'''


import cv2
import numpy as np

filename = 'newimg.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.18)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

#cv2.imshow('dst',img)
cv2.imwrite('imgcantos.png',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()



def local_cantos(dst):
    cantos = []
    lim = 0.01*dst.max()
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] > lim:
                cantos += [[i, j]]
    return cantos

cantos=local_cantos(dst)

tamanho=0
for i in local_cantos(dst):
    tamanho+=1

def paint_square(x,y):
    print("testando")
    #print(cantos.size())
    for i in range(30):
        img[x+i+1,y-1] = [0,255,0]
        img[x+i+1,y] = [0,255,0]
        img[x+i+1,y+1] = [0,255,0]
    for i in range(30):
        img[x+30-1,y+i+1] = [0,255,0]
        img[x+30,y+i+1] = [0,255,0]
        img[x+30+1,y+i+1] = [0,255,0]
    for i in range(30):
        img[x-1,y+i+1] = [0,255,0]
        img[x,y+i+1] = [0,255,0]
        img[x+1,y+i+1] = [0,255,0]
    for i in range(30):
        img[x+i+1,y+30-1] = [0,255,0]
        img[x+i+1,y] = [0,255,0]
        img[x+i+1,y+30+1] = [0,255,0]

def verificar(dst):
    for i in range(0,tamanho,18):
        x1 = cantos[i][0]
        y1 = cantos[i][1]
        x2 = x1+30 
        y2 = y1+30
        if dst[x1,y1] == dst[x1,y2] and dst[x1,y1] == dst[x2,y1]:
            paint_square(x1,y1)

verificar(dst)


#paint_square(122,819)
cv2.imwrite('teste.png',img)












'''

import cv2
import numpy as np

filename = 'foto.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

cv2.imwrite('subpixel5.png',img)'''
