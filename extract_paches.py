import PIL
import numpy as np

# inputs -----------------------------
path_load = '/poze_db/DSC06570.JPG' # requires .jpg
path_save = '/poze_db/bucati/'  # requires / after path
size = 128
stride = 128
# -------------------------------------

def get_patchess(img_arr, size=256, stride=256, filename='PATCH'):
    '''
    Takes single image or array of images and returns
    crops using sliding window method.
    If stride < size it will do overlapping.
    '''
    # check size and stride
    if size % stride != 0:
        raise ValueError('size % stride must be equal 0')

    patches_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1

    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping

        for i in range(i_max):
            for j in range(i_max):
                # print(i*stride, i*stride+size)
                # print(j*stride, j*stride+size)
                patches_list.append(
                    img_arr[i * stride:i * stride + size,
                    j * stride:j * stride + size
                    ])
        j = 0
        for patches in patches_list:
            # print("patch " + str(j))
            j = j + 1
            img = Image.fromarray(patches, 'RGB')
            img.save(path_save + 'extraxt-' + str(j) + '.jpg', 'JPEG')


    elif img_arr.ndim == 4:
        i_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(i_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[i * stride:i * stride + size,
                        j * stride:j * stride + size
                        ])

    else:
        raise ValueError('img_arr.ndim must be equal 3 or 4')

    return np.stack(patches_list)

# filename = 'train/mask/48/train_mask'
x = np.array(Image.open(path_load))
print("x shape: ", str(x.shape))

x_crops = get_patchess(
    img_arr=x, # required - array of images to be cropped
    size=size, # default is 256
    stride=stride,
    filename=path_load) # default is PATCH

print("x_crops shape: ", str(x_crops.shape))
