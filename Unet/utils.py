    #
    #	Utilities file 
    #

from keras.callbacks import EarlyStopping
from glob import glob
import os, numpy as np
from keras.preprocessing import image
from PIL import Image

from config import DEBUG_LEVEL

class meetExpectations(EarlyStopping):
    '''
        Applies an early stopping when a metric threshold is reached
        TODO: now the metric applies only to the val_acc but this must be defined though the parameter "monitor"
    '''
    def __init__ (self, threshold, min_epochs, **kwargs):
        super(meetExpectations, self).__init__(**kwargs)
        self.threshold = threshold
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor) # self_monitor is the way to get the monitored metric ... 
        
        if current is None:
            super.warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        
        if (epoch >= self.min_epochs) & (current >= self.threshold):
            self.stopped_epoch = epoch
            print("The model now stops because of reaching the val_acc threshold, in epoch: ", epoch)
            self.model.stop_training = True

class adaptLearningRate(EarlyStopping):
    '''
        Applies updates learning rate to adapt to steep slopes.
    '''
    #TODO: to be done
    pass

def load_image(path, target_size = (1024, 1024)):
    '''
        returns a np.array of the loaded image
    '''
    im = image.load_img(path, target_size=target_size)
    img = image.img_to_array(im)
    if hasattr(im, 'close'):
            im.close()
    return img

def store_image(img, dst):
    '''
        stores image to dst
    '''
    if type(img) is not "numpy.ndarray":
            img = np.asarray(img)
    result = Image.fromarray(img.astype(np.uint8))
    result.save(dst)

def greyscale(x):
    '''
        converts rgb image to greyscale one
    '''
    x = x.astype(int)
    res = (0.21 * x[:,:,:1]) + (0.72 * x[:,:,1:2]) + (0.07 * x[:,:,-1:])
    res = res / 255
    return res.squeeze()

def resize_img(img, min_size=256, fill_color=(0, 0, 0)):
    '''
        Resizes any numpy based image to ta square one ...
        Respects original image ratio 
        Parameters:
        -----------
            im: np.ndarray typed image
            min_size: integer
            file_color: RGBA color to pad image
        Returns:
        -----------
            new_im: np.ndarray typed image
    '''
    im = Image.fromarray(img.astype('uint8'))
    im.thumbnail((min_size, min_size), Image.ANTIALIAS) #Hint: side-effects!
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, ((size - x) // 2, (size - y) // 2))
    new_im = np.array(new_im)
    return new_im

def get_segment_crop(img, tol=0, mask=None, tile_shape = None):   
    '''
        corps any image based on some particular mask
        if mask is not provided segmentation can take 
        place using the tol threshold
        if tile_shape is set to some shape the image shoudl
        feature an exact manifold of that particular tile
        
        Params:
        -------
            img: np.ndarray
            tol: int a threshold to generate an ad hoc mask
            mask: mask as it is laoded
            tile_shape: unit area that we need image are to be manifold of
        
        Returns:
            cropped image: np.ndarray
    '''
    # ================ Debugging routine follows =============
    if DEBUG_LEVEL > 1:
        assert type(img)==np.ndarray
        if mask:
            assert mask.shape[0]==img.shape[0], "Image, mask rows mismatch"
            assert mask.shape[1]==img.shape[1], "Image, mask cols mismatch"
    # ========================================================
    if mask is None:
        img = greyscale(img)
        mask = img > tol
    else:
        mask = greyscale(mask)
        mask = mask > 0

    if tile_shape: # that is for a potential tiling process... 

        # get the only the rows and columns that mask is activated on
        img_rows = mask.any(1)
        img_cols = mask.any(0)

        # Hint img_cols and img_rows are np.arrays of booleans 
        num_of_missing_rows = np.where(img_rows==True)[0].shape[0] % tile_shape[0]
        num_of_missing_cols = np.where(img_cols==True)[0].shape[0] % tile_shape[1]
        if num_of_missing_rows > 0:
            num_of_missing_rows = tile_shape[0] - num_of_missing_rows
        if num_of_missing_cols > 0:
            num_of_missing_cols = tile_shape[0] - num_of_missing_cols

        # get the furthest row and column of True
        f_row = np.where(img_rows==True)[0].max()
        f_col = np.where(img_cols==True)[0].max()
        for i in range(0, num_of_missing_rows+1):
            img_rows[f_row+i] = True
        for i in range(0, num_of_missing_cols+1):
            img_cols[f_col+i] = True
        
        cropped_img = img[np.ix_(img_rows, img_cols)]
        return cropped_img

    else:
        return img[np.ix_(mask.any(1), mask.any(0))]

def generate_paths(input_example, output_example):
    '''
        This is a "smart" function to understand the data organization layout
        Hint: the assumed structure is as indicated bellow
            1. inps: <root_path>/<identifier><suffix>
            2. outs: <root_path>/<identifier><separator><attribute><suffix>

            for outs the <separator><attribute> are optional to exist
    '''
    def lcs (S, T):
        '''
            returns the longest common substring for the given strings s, t
        '''
        m = len(S)
        n = len(T)
        counter = [[0]*(n+1) for x in range(m+1)]
        longest = 0
        lcs_set = set()
        for i in range(m):
            for j in range(n):
                if S[i] == T[j]:
                    c = counter[i][j] + 1
                    counter[i+1][j+1] = c
                    if c > longest:
                        lcs_set = set()
                        longest = c
                        lcs_set.add(S[i-c+1:i+1])
                    elif c == longest:
                        lcs_set.add(S[i-c+1:i+1])

        return "".join(lcs_set)

    # check if input and output paths are relative or absolute
    if input_example[0] == "/":
        input_is_absolute = True
    else:
        input_is_absolute = False
    
    if output_example[0] == "/":
        output_is_absolute = True
    else:
        output_is_absolute = False
    
    # let's od it deflate input example
    input_deflated = input_example.split("/")
    input_root = os.path.join(*input_deflated[:-1])
    if input_is_absolute:
        input_root = "/"+input_root
    input_suffix = "."+input_deflated[-1].split(".")[-1]
    input_name = input_deflated[-1].split('.')[0]

    output_deflated = output_example.split("/")
    output_root = os.path.join(*output_deflated[:-1])
    if output_is_absolute:
        output_root = "/"+output_root
    output_name = output_deflated[-1].split(".")[0]
    output_suffix = "." + output_deflated[-1].split(".")[-1]

    identifier = lcs(input_name, output_name)
    mask_sep = output_name.split(identifier)[-1]

    if mask_sep != "" : # if there is more info in mask add it to the suffix
        output_suffix = mask_sep + output_suffix
        # TODO: now investigate the directory to check for more information
        mask_paths = glob(os.path.join(output_root, identifier + "*"))
        if len(mask_paths) > 1: # we have more classes and attributes
              #TODO:
              mask_paths = list(map(lambda x: x.split("/")[-1].split(".")[0], mask_paths))
              mask_paths = list(map(lambda x: x.split(identifier)[-1], mask_paths))
              attribute_prefix = lcs(mask_paths[0], mask_paths[1])
        else:
            attribute_prefix = ""

    #check if input_root or output_root are relative or absolutes

    return input_root, input_suffix, output_root, output_suffix, attribute_prefix


def pretty_print(message, *kwords):
    '''
        This is a printing wrapper to enable optional
        debugging functionality

        Params:
        -------
        message: string, debigger message
        multiple: other things we need to print along with the message

        Returns:
        -------
            nothing --> just prints on the screen
    '''
    if DEBUG_LEVEL == 2:
        print(message, *kwords)
