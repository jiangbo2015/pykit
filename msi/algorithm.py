from skimage.measure import compare_ssim
from scipy.misc import imread, imsave
import cv2
import numpy as np
from PIL import Image
from numpy import average, linalg, dot

class Algorithm:
    def __init__(self, im1_path, im2_path):
        self.im1 = cv2.imread(im1_path)
        self.im2 = cv2.imread(im2_path)
        self.im1_path = im1_path
        self.im2_path = im2_path
        
    def use_ssim(self):
        im2 = np.resize(self.im2, (self.im1.shape[0], self.im1.shape[1], self.im1.shape[2]))
        ssim = compare_ssim(self.im1, im2, multichannel=True)
        return ssim

    def use_ssim_gray(self):
        im2 = np.resize(self.im2, (self.im1.shape[0], self.im1.shape[1], self.im1.shape[2]))
        grayA = cv2.cvtColor(self.im1, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, grayB, full=True)
        return score

    def get_thumbnail(self, image, greyscale=False):
        size = size=(self.im1.shape[1], self.im1.shape[0])
        image = image.resize(size, Image.ANTIALIAS)
        if greyscale:
            image = image.convert('L')
        return image

    def use_cosin(self):
        image1 = Image.open(self.im1_path)
        image2 = Image.open(self.im2_path)
        image1 = self.get_thumbnail(image1)
        image2 = self.get_thumbnail(image2)
        images = [image1, image2]
        vectors = []
        norms = []
        for image in images:
            vector = []
            for pixel_tuple in image.getdata():
                vector.append(average(pixel_tuple))
            vectors.append(vector)
            norms.append(linalg.norm(vector, 2))
        a, b = vectors
        a_norm, b_norm = norms
        res = dot(a / a_norm, b / b_norm)
        return res

    def use_hist(self):
        img1_path = self.im1_path
        img2_path = self.im2_path

        def make_regalur_image(img, size=(256, 256)):
            return img.resize(size).convert('RGB')

        def hist_similar(lh, rh):
            assert len(lh) == len(rh)
            return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)


        def calc_similar(li, ri):
            return sum(hist_similar(l.histogram(), r.histogram()) for l, r in zip(split_image(li), split_image(ri))) / 16.0


        def calc_similar_by_path(lf, rf):
            li, ri = make_regalur_image(Image.open(lf)), make_regalur_image(Image.open(rf))
            return calc_similar(li, ri)


        def split_image(img, part_size = (64, 64)):
            w, h = img.size
            pw, ph = part_size
            assert w % pw == h % ph == 0
            return [img.crop((i, j, i+pw, j+ph)).copy() for i in range(0, w, pw) \
                    for j in range(0, h, ph)]
        
        t = calc_similar_by_path(img1_path, img2_path)
        return t

        
    def __aHash(self, img):
        
        img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
        
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        s=0
        hash_str=''
        
        for i in range(8):
            for j in range(8):
                s=s+gray[i,j]
        
        avg=s/64
        
        for i in range(8):
            for j in range(8):
                if  gray[i,j]>avg:
                    hash_str=hash_str+'1'
                else:
                    hash_str=hash_str+'0'            
        return hash_str

    def __compare_hash(self, func):
        h1 = func(self.im1)
        h2 = func(self.im2)
        return self.__hammingDist(h1, h2)

    def __hammingDist(self, s1, s2):
        assert len(s1) == len(s2)
        return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])
        
    
    def __dHash(self, img):
        
        img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
        
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        hash_str=''
        
        for i in range(8):
            for j in range(8):
                if   gray[i,j]>gray[i,j+1]:
                    hash_str=hash_str+'1'
                else:
                    hash_str=hash_str+'0'
        return hash_str

    def __pHash(self, imgfile):
        
        img=cv2.imread(imgfile, 0)
        img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
    
            
        h, w = img.shape[:2]
        vis0 = np.zeros((h,w), np.float32)
        vis0[:h,:w] = img       
    
        
        vis1 = cv2.dct(cv2.dct(vis0))
        
        vis1.resize(32,32)
        
        img_list = vis1.flatten()
    
        
        avg = sum(img_list)*1./len(img_list)
        avg_list = ['0' if i<avg else '1' for i in img_list]
    
        

        return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,32*32,4)])
    
    def __cmpHash(hash1,hash2):
        n=0
        
        if len(hash1)!=len(hash2):
            return -1
        
        for i in range(len(hash1)):
            
            if hash1[i]!=hash2[i]:
                n=n+1
        return n

    def a_hash(self):
        return self.__compare_hash(self.__aHash)
    
    def d_hash(self):
        return self.__compare_hash(self.__dHash)

    def p_hash(self):
        h1= self.__pHash(self.im1_path)
        h2= self.__pHash(self.im2_path)
        t = 1 - self.__hammingDist(h1, h2)*1. / (32*32/4)
        return t

    