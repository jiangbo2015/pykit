# -*-encoding:utf-8-*-
import pytesseract
from PIL import Image

name = '5.png'

# 四邻域降噪，二值化后，0为黑色，1为白色
def depoint(img):
    pixdata = img.load()
    w,h = img.size
    for y in range(0,h-1):
        for x in range(0,w-1):
            count = 0

            # 白色点直接跳过
            if pixdata[x,y] == 1:
                print('白色')
                continue

            if pixdata[x,y-1] == 1:
                count = count + 1
            if pixdata[x,y+1] == 1:
                count = count + 1
            if pixdata[x-1,y] == 1:
                count = count + 1
            if pixdata[x+1,y] == 1:
                count = count + 1
            # 该点周围白点个数超出阈值，则设为白色
            if count > 2:
                pixdata[x,y] = 1
    return img



def main():
    im = Image.open('source/'+name)

    #转化到灰度图
    imgry = im.convert('L')

    #二值化，可调整threshold
    threshold = 177
    table = []
    for j in range(256):
        if j < threshold:
            table.append(0)
        else:
            table.append(1)

    out = imgry.point(table, '1')
    out.save('temp/two.png')

    out = depoint(out)
    out = depoint(out)

    out.save('temp/'+name)

    text = pytesseract.image_to_string(out)
    print(text)

if __name__ == '__main__':
	main()
