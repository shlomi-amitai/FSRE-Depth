import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy.matlib


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    """
    def __call__(self, img):
        """Convert a ``numpy.ndarray`` to tensor.

        Args:
            img (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if isinstance(img, np.ndarray):
            # handle numpy array
            if img.ndim == 3:
                img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
            elif img.ndim == 2:
                img = torch.from_numpy(img.copy())
            else:
                raise RuntimeError(
                    'img should be ndarray with 2 or 3 dimensions. Got {}'.
                    format(img.ndim))

            return img

to_tensor = ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def toNumpy(tensorIm):
    im = np.squeeze(tensorIm.cpu().detach().numpy())
    if im.ndim == 3:
        im = im.transpose((1, 2, 0))
    if im.ndim == 4:
        im = im.transpose((0, 2, 3, 1))
    return im


def saveTensorAsNPY(image, filename="tempIm"):
    im = np.squeeze(image.cpu().detach().numpy())
    if im.ndim == 3:
        im = im.transpose((1, 2, 0))
    np.save(filename, im)

def plotTensor(image, crop=False):
    im = np.squeeze(image.cpu().detach().numpy())
    if im.ndim == 3:
        im = im.transpose((1, 2, 0))
    import matplotlib.pyplot as plt
    if crop:
        im = im[cropSize:-cropSize, cropSize:-cropSize]
    plt.imshow(im)
    plt.show()

def plot2Tensors(image1, image2, crop=False):  
    import matplotlib.pyplot as plt
    im = torch.cat((image1, image2), 0)
    im = np.squeeze(im.cpu().detach().numpy())
    imList = list(im)
    show_images(imList, cols=2)

def plot2Numpies(image1, image2, crop=False):
    import matplotlib.pyplot as plt
    imList = list([image1, image2])
    show_images(imList, cols=1)

def plotNumpyMultiple(imList, cols=1):
    import matplotlib.pyplot as plt
    show_images(imList, cols)


def plotNumpy(image):
    im = np.squeeze(image)
    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.show()

def plotTensorScaled(image, crop=True):
    im = np.squeeze(image.cpu().detach().numpy())
    if crop:
        im = im[cropSize:-cropSize, cropSize:-cropSize]
    new_im = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
    import matplotlib.pyplot as plt
    cmap = plt.cm.jet
    coloredIm = cm(new_im)
    plt.imshow(coloredIm)
    plt.show()

def plotTensorMultiple(image, cols=4):
    im = np.squeeze(image.cpu().detach().numpy())
    imList = list(im)
    show_images(imList, cols)


def plotTensorsDiff(image1, image2):
    # image1 = image1.view(1,1,-1,-1)
    image_diff = abs(image1.astype(np.float32) - image2.astype(np.float32))
    images = torch.cat([image1, image2, image_diff])
    plotTensorMultiple(images, 3) 


def plotNumpyDiff(image1, image2):
    image_diff = image1 - image2
    images = np.squeeze([image1, image2, image_diff])
    show_images(images, 3)

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        # if image.ndim == 2:
            # plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def cropIm(image, rect):
    x0, y0, w, h = (np.asarray(rect)).astype(np.int32)
    if image.ndim == 3:
        return image[y0:y0 + h, x0:x0 + w, :]
    elif image.ndim == 2:
        return image[y0:y0 + h, x0:x0 + w]




################ Tensorflow ##########################

def pytorchToTensorflow(torchIm):
    tfImage = torchIm.cpu().detach().numpy()
    if tfImage.ndim == 4:
        tfImage = tfImage.transpose((0, 2, 3, 1))

    return tfImage



## operators
def preProcessDepth(depthMap, edgeTh=0.2, erode_kernel=5):
    kernel1 = np.ones((erode_kernel, erode_kernel), np.uint8)
    kernel2 = np.ones((2*erode_kernel, 2*erode_kernel), np.uint8)
    g = np.gradient(np.squeeze(depthMap))
    gradIm = abs(g[0])+abs(g[1])
    _,gradImBin1 = cv2.threshold(gradIm, edgeTh,255,cv2.THRESH_BINARY_INV); gradImBin1 = gradImBin1.astype(np.uint8)
    _,gradImBin2 = cv2.threshold(gradIm, 4*edgeTh,255,cv2.THRESH_BINARY_INV); gradImBin2 = gradImBin2.astype(np.uint8)
    gradImBin2 = cv2.morphologyEx(gradImBin2, cv2.MORPH_ERODE, kernel2)
    gradImBin1 = cv2.morphologyEx(gradImBin1, cv2.MORPH_ERODE, kernel1)
    gradImBin = gradImBin1 & gradImBin2
    depth_gt = depthMap.copy()
    depthMap[gradImBin==0]=0
    # depthMap = cv2.morphologyEx(depthMap, cv2.MORPH_ERODE, kernel)
    depthMap[gradImBin>0] = depth_gt[gradImBin>0]
    return depthMap





    
if __name__ == "__main__":
    # test depth cleanup
    import PIL.Image as pil
    depth_path = r'\\CPStorage\\RnD\\Users\\Shlomi\\depthCompletionData\\ANSFL\\U_Canyon\\depth\\16233043991687813_abs_depth.tif'
    depth_gt = pil.open(depth_path)
    depth_gt = np.array(depth_gt).astype(np.float32)

    depth = preProcessDepth(depth_gt)
    plotNumpy(depth_gt)
    cv2.waitKey()
