from algorithm import Algorithm

algorithm = Algorithm('image/im1.png', 'image/im2.png')
t = algorithm.use_ssim()
t2 = algorithm.use_ssim_gray()
t3 = algorithm.use_cosin()
t4 = algorithm.a_hash()
t5 = algorithm.d_hash()
t6 = algorithm.p_hash()
t7 = algorithm.use_hist()
print(t, t2, t3, t4, t5, t6, t7)