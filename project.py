from __future__ import division
import math
import functools
import time
from PIL import Image
import os
import zipfile

image_mode = "RGB"
grayscale = False

def decorator(d):
    "Make function d a decorator: d wraps a function fn."
    def _d(fn):
        return functools.update_wrapper(d(fn), fn)
    return _d
decorator = decorator(decorator)

@decorator
def disabled(f):
    return f

# for caching
@decorator
def memo(f):
    """Decorator that caches the return value for each call to f(args).
    Then when called again with same args, we can just look it up."""
    cache = {}
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            result = f(*args)
            cache[args] = result
            return result
        except TypeError:
            # some element of args can't be a dict key
            return f(*args)
    _f.cache = cache
    return _f

@decorator
def timer(f):
    def _f(*args):
        t1 = time.time()
        results = f(*args)
        t2 = time.time()
        print f.__name__ , t2-t1
        return results
    return _f

def imread(input_file, gs=False):
    im = Image.open(input_file)
    global image_mode, grayscale
    image_mode = im.mode
    if image_mode == "RGBA" : image_mode = "RGB"
    grayscale = len(image_mode) < 3
    image_array = []
    row, col = im.size
    for x in range(row):
        temp = []
        for y in range(col):
            temp.append(im.getpixel( (x,y)))
        image_array.append(temp)
        
    return image_array

def imsave(output_file, im, format="BMP"):
    im2 = Image.new(image_mode, (len(im), len(im[0])))
    for x in range(len(im)):
        for y in range(len(im[0])):
            if not grayscale:
                element = tuple( im[x][y]) 
            else:
                element = im[x][y]
            im2.putpixel((x,y), element)
    im2.save(output_file, format)
    
## The functions above are not related to compression
##
######################################################

Q = [[16,11,10,16,24,40,51,61],
     [12,12,14,19,16,58,60,55],
     [14,13,16,24,40,57,69,56],
     [14,17,22,29,51,87,80,62],
     [18,22,37,56,68,109,103,77],
     [24,35,55,64,81,104,113,92],
     [49,64,78,87,103,121,120,101],
     [72,92,95,98,112,100,103,99]]


block_size = 8;



# the alpha function used in the dct transform

def alpha(u):
    if u is 0:
        return math.sqrt(1/8.0)
    else :
        return math.sqrt(2/8.0)


# computes a one dimensional dct, or its inverse

def dct(i, block, inverse=False):
    if inverse==False:
        return sum( [block[x] * alpha(i) * math.cos( (math.pi/len(block)) * (x + 1/2.0) * i) 
                     for x in range(len(block))])
    elif inverse==True:
        return sum( [alpha(x) * block[x] * math.cos( (math.pi / 8.0) * ( i + 1/2.0)* x) 
                     for x in range(len(block)) ] )
  
# In this function I go through each row one by one and compute the dct using the dct() function above. 
# I should transpose the image and call this function one more time to obtain the dct coefficients    
def image_dct(image, inverse=False, blocksize=block_size):
    cols = int(len(image[0])/blocksize) # the number of one-dimensional blocks in the each row
    new_image = []
    
    for row in range(len(image)):
        temprow=[]
        for i in range(cols):
            temp = []
            for k in range(blocksize):
                temp.append(
                            dct(k, 
                                image[row][i*blocksize:(i+1)*blocksize],
                                inverse==True)
                            )
            temprow += temp
        new_image.append(temprow)
    return new_image

def center_around_zero( matrix, inverse=False):
    sign = -1
    if inverse is True:
        sign = +1
    balanced_matrix = []
    for x in range(len(matrix)):
        temp = []
        for y in range(len(matrix[x])):
            number = matrix[x][y] + sign*128
            if number > 255: number = 255
            temp.append(int(number))
        balanced_matrix.append(temp)
    return balanced_matrix;

def separate_rgb(image):
    r = []; g= []; b=[];
    for x in range( len(image)):
        r_temp = []; g_temp = []; b_temp = [];
        for y in range(len(image[x])):
            r_temp.append(image[x][y][0])
            g_temp.append(image[x][y][1])
            b_temp.append(image[x][y][2])
        r.append(r_temp); g.append(g_temp); b.append(b_temp)
    return r,g,b

def reunite_rgb(r, g, b):
    block = []
    for x in range(len(r)):
        temp = []
        for y in range(len(r[x])):
            temp.append( [r[x][y], g[x][y],b[x][y]])
        block.append(temp)
    return block

def quantize_dct_row(row, q, inverse=False):
    blocksize = len(q)
    for i in range( int(len(row)/blocksize)):
        for j in range(blocksize):
            if inverse==False:
                row[i*blocksize + j] = round(row[i*blocksize + j] / q[j])
            else:
                row[i*blocksize + j] = round( row[i*blocksize + j] * q[j])
    return row

def quantize_dct_image(image, inverse=False, q=Q):
    for row in range(len(image)):
        image[row] = quantize_dct_row( image[row], q[row % len(q)], inverse==True)
    return image
        
def encode(image, blocksize = block_size):
    new_image = center_around_zero(image)
    new_image = image_dct(new_image, blocksize)
    new_image = map(list, zip(* new_image)) # transpose
    new_image = image_dct(new_image, blocksize)
    new_image = map(list, zip(* new_image)) # transpose back
    new_image = quantize_dct_image(new_image)
    new_image = center_around_zero(new_image, True)
    return new_image

def encode_image(image, grayscale = False, blocksize=block_size):
    if grayscale:
        return encode(image, blocksize)
    else:
        r,g,b = separate_rgb(image)
        r_encoded = encode(r,blocksize)
        g_encoded = encode(g,blocksize)
        b_encoded = encode(b,blocksize)
        
        new_image = reunite_rgb(r_encoded, g_encoded, b_encoded)
        
        return new_image

def decode(image, blocksize = block_size):
    new_image = center_around_zero(image)
    new_image = quantize_dct_image(new_image, True)
    new_image = image_dct(new_image, True, blocksize)
    new_image = map(list, zip(* new_image)) # transpose
    new_image = image_dct(new_image, True, blocksize)
    new_image = map(list, zip(* new_image)) # transpose back
    new_image = center_around_zero(new_image, True)
    return new_image

def decode_image(image, grayscale = False, blocksize=block_size):
    if grayscale:
        return decode(image, blocksize)
    else:
        r,g,b = separate_rgb(image)
        r_decoded = decode(r,blocksize)
        g_decoded = decode(g,blocksize)
        b_decoded = decode(b,blocksize)
        
        new_image = reunite_rgb(r_decoded, g_decoded, b_decoded)
        
        return new_image

def histogram(image):    # finds the histogram of an image
    hist = {}
    for row in image:
        for item in row:
            item = int(item)
            if item in hist.keys():
                hist[item]+=1
            else:
                hist[item] = 1
    return hist

def find_cdf(image):
    hist = histogram(image)
    size = sum(hist.values())
    total = 0
    cdf = {}
    for key, value in hist.items():
        total += value
        cdf[key] = total/size
    return cdf

def match_histograms(reference, target):
    cdf_ref = find_cdf(reference)
    cdf_target = find_cdf(target)
    
    m = {}
    for f1_key, f1_value in cdf_ref.items():
        for f2_key, f2_value in cdf_target.items():
            if f1_value == f2_value:
                m[f1_key] = f2_key
    new_image = target
    for i,row in enumerate(reference):
        for j,item in enumerate(row):
            if item in m:
                new_image[i][j] = m[item]
    return new_image
    
@timer
def match_image_histograms(reference, target):
    if not grayscale:
        l = []
        for g1, g2 in zip(separate_rgb(reference), separate_rgb(target)):
            temp = match_histograms(g1, g2)
            l.append(temp)
            
        new_image = reunite_rgb(*l)
    else:
        new_image = match_histograms(reference, target)
    return new_image
        
def zip_file(filename, output_filename):
    try:
        z = zipfile.ZipFile(output_filename, mode="w", compression=zipfile.ZIP_DEFLATED)
        z.write(filename, filename, zipfile.ZIP_DEFLATED)
        for a in z.infolist():
            return 100 *(1 - a.compress_size/a.file_size)
    except Exception:
        print "zip_file error"
        pass
   
def unzip_file(filename, output_filename):
    try:
        z = zipfile.ZipFile(filename, mode="r")
        
        for name in z.namelist():
            d = z.read(name)
            f = open(output_filename, "w")
            f.write(d)
    except Exception:
        print "unzip_file error"
    

test_block = [[52, 55, 61, 66, 70, 61, 64, 73],
         [63, 59, 55, 90, 109, 85, 69, 72],
         [62, 59, 68, 113, 144, 104, 66, 73],
         [63, 58, 71, 122, 154, 106, 70, 69],
         [67, 61, 68, 104, 126, 88, 68, 70],
         [79, 65, 60, 70, 77, 68, 58, 75],
         [85, 71, 64, 59, 55, 61, 65, 83],
         [87, 79, 69, 68, 65, 76, 78, 94]]

test_block2 = [ [1,2,3,4,5,6,7,8],
                [5,5,5,5,5,5,5,5],
                [3,4,4,4,5,5,5,6],
                [6,5,4,3,2,1,2,2],
                [1,2,3,4,5,6,7,8],
                [5,5,5,5,5,5,5,5],
                [3,4,4,4,5,5,5,6],
                [6,5,4,3,2,1,2,2]]


uncompressed_image = None
compressed_image = None
encoded_image = None

@timer
def test_encode():
    global encoded_image, uncompressed_image, grayscale
    uncompressed_image = imread(input_file_name, grayscale)
    
    encoded_image = encode_image(uncompressed_image, grayscale)
    imsave(encoded_file_name, encoded_image)


@timer
def test_decode():
    global compressed_image, uncompressed_image, encoded_image
    encoded_image = imread(unzipped_file_name)
    uncompressed_image = imread(input_file_name)
    compressed_image = decode_image(encoded_image, grayscale)
#    compressed_image = uncompressed_image
#    compressed_image = encoded_image
#    print "grayscale", grayscale, image_mode
#    compressed_image = match_image_histograms(compressed_image, uncompressed_image)
    imsave(output_file_name,compressed_image)


dirs = ["uncompressed_images\\", "encoded_images\\", "decoded_images\\", "zipped_images\\","unzipped_images\\"]

for d in dirs:
    if not os.path.exists(d):
        os.mkdir(d)
            
root = r''
filename = ''

for filename in os.listdir(root + dirs[0]):
    input_file_name = dirs[0] + filename
    encoded_file_name = dirs[1] + filename[:-4] + "_encoded.bmp"
    output_file_name = dirs[2] + filename[:-4] + r'_decoded.bmp'
    zipped_file_name = dirs[3] + filename[:-4] + r'_zipped.zip'
    unzipped_file_name = dirs[4] + filename[:-4] + '_unzipped.' + filename[-3:]

    test_encode()
    compression_ratio = zip_file(encoded_file_name, zipped_file_name)
    print "compression ratio for " + filename + " : " , round(compression_ratio), "%"
    unzip_file(zipped_file_name, unzipped_file_name)
    test_decode()
