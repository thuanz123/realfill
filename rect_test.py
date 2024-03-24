from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import math

def rescale_rects(rects, scaling_factor, image_height, image_width):
    n = len(rects)
    n_width = math.ceil(math.sqrt(n))
    n_height = round(math.sqrt(n))
    predicted_image_width = scaling_factor*np.max(np.array([x['width'] for x in sorted(rects, key = lambda x: x['width'], reverse = True)[:n_width]]))
    predicted_image_height = scaling_factor*np.max(np.array([x['height'] for x in sorted(rects, key = lambda x: x['height'], reverse = True)[:n_height]]))
    width_factor = image_width/predicted_image_width
    height_factor = image_height/predicted_image_height
    
    # print(predicted_image_width)
    print(predicted_image_height)
    # print(width_factor)
    print(height_factor)
    # print(rects)
    # Resize rects
    rects_resize = []
    for idx, rect in enumerate(rects):
        aspect_ratio = rect['width']/rect['height']
        width_new = rect['width']*width_factor*aspect_ratio    
        height_new = rect['height']*height_factor/aspect_ratio
        
        f_aspect_ratio = (width_new/rect['width'])*(rect['height']/height_new)
        # print(f_aspect_ratio)
        # width_final = (f_aspect_ratio*rect['width'])/(rect['height']/height_new)
        width_final = width_new
        height_final = height_new
        # print(width_final)
        # print(height_final)
        rect['width'] = int(round(width_final))
        rect['height'] = int(round(height_final))
        rects_resize.append(rect)
    return(rects_resize)

def pack_rows(rects, scaling_factor = 1, image_height = 700, image_width = 700, order_by_height = False):
    if order_by_height == True:
        rects.sort(key = lambda x: x['height'], reverse = True)
    
    # Calculate resizing factors
    print(rects)
    rects = rescale_rects(rects, scaling_factor, image_height, image_width)
    print(rects)
    xpos = 0
    ypos = 0
    dx = 0
    dy = 0
    largest_h_this_row = 0

    rects_new = []
    # print(rects)
    for rect in rects:
        rect_new = Rect(rect['x'], rect['y'], rect['height'], rect['width'])
        rect_new['x'] = int(round(xpos) + (dx - rect_new['width'])/2)
        rect_new['y'] = int(round(ypos) + (dy - rect_new['height'])/2)
        dx = rect['width']*scaling_factor
        dy = rect['height']*scaling_factor
        # dx = rect['width']
        # dy = rect['height']
        print(dx)
        print(dy)
        if xpos + dx > image_width:
            ypos += largest_h_this_row
            xpos = 0
            largest_h_this_row = 0
        if ypos + dy > image_height:
            break
        xpos += dx

        if rect['height'] >= largest_h_this_row:
            largest_h_this_row = rect['height']*scaling_factor
        rect_new['was_packed'] = True
        rects_new.append(rect_new)
    print(rects_new)
    return(rects_new)

# def pack_two(rects, scaling_factor = 1, image_height = 700, image_width = 700):
#     new_width = 


def pack_circle(rects, phi_0 = 0, scaling_factor = 1, image_height = 700, image_width = 700):
    widths = [z['width'] for z in rects]
    heights = [z['height'] for z in rects]
    rects_new = []
    n = len(rects)
    angle_rad = 2*math.pi/n
    max_dims = np.array([max(widths), max(heights)])
    r = scaling_factor*min(max_dims)
    image_center = (2*r + max_dims/2)/2
    phi = np.arange(n)*angle_rad + phi_0
    x = int(round(r*np.cos(phi) + image_center[0]))
    y = int(round(r*np.sin(phi) + image_center[1]))
    for idx, elem in enumerate(zip(x,y)):
        rects_new.append(Rect(elem[0], elem[1], widths[idx], heights[idx], was_packed=True))
    return(rects_new)



def get_images(image_folder_path):
    ref_data_root = Path(image_folder_path) / "ref"
    train_image_paths = list(Path(ref_data_root).iterdir())
    images = [Image.open(path).convert("RGB") for path in train_image_paths]
    return([images, train_image_paths])

class Rect:
    """
    A class representing a rectangle
    """

    def __init__(self,x,y,height,width,was_packed = False):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.was_packed = was_packed

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self,key)
    
    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, height: {self.height}, width: {self.width}, was_packed: {self.was_packed}" 

def generate_layout(images, method = 'rows', image_height = 1000, image_width = 1000, scaling_factor = 1, order_by_height = False):
    trans = transforms.Compose([transforms.ToTensor()])
    tensors = [trans(im) for im in images]
    sizes = [tuple(x.size()[1:]) for x in tensors]
    rects = [Rect(0,0,sz[0],sz[1]) for sz in sizes]
    # print(rects)
    if method == 'rows':
        rects = pack_rows(rects=rects, scaling_factor = scaling_factor, 
                          image_height = image_height, 
                          image_width = image_width,
                          order_by_height = order_by_height)
    elif method == 'circle':
        rects = pack_circle(rects=rects, scaling_factor = scaling_factor, 
                            image_height = image_height, 
                            image_width = image_width)
    else:
        sys.exit('method must be either "rows" (default) or "circle"')
    return(rects)
    

def layout_and_mask(images_path = '/home/feshap/src/realfill/data/noam_photos/Photos-001', 
                    method = 'rows',
                    scaling_factor = 1.2,
                    image_height = 1000, 
                    image_width = 1000,
                    order_by_height = False):

# Make layout according to constraints of input images and target image size
    images, paths = get_images(images_path)
    
    rects = generate_layout(images, 
                        method = method,
                        image_height = image_height, 
                        image_width = image_width, 
                        scaling_factor = scaling_factor,
                        order_by_height = order_by_height)

    for rect in rects:
        print(rect)
# Make mask and resize images
    mask_tns = torch.ones(1,image_height,image_width)
    for rect, img, path in zip(rects, images, paths):
        mask_tns[0,rect['y']:(rect['y']+rect['height']),rect['x']:(rect['x']+rect['width'])] = 0
        path_edit = path.with_suffix('.rsz.png')
        img_rsz = img.resize((rect['width'], rect['height']), resample = Image.BICUBIC)
        img_rsz.save(path_edit)
    mask_img = transforms.functional.to_pil_image(mask_tns, mode = 'L')
    mask_img.save(Path(images_path) / 'target/mask.png')


def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='none', alpha=0.5):

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='none', ecolor='k')

    return artists

def plot_rects(rects):
    x = np.array([elem['x'] + elem['width']/2 for elem in rects])
    y = np.array([elem['y'] + elem['height']/2 for elem in rects])
    xerr = np.array([elem['width']/2 for elem in rects])
    yerr = np.array([elem['height']/2 for elem in rects])
    xerr = np.vstack(2*[xerr])
    yerr = np.vstack(2*[yerr])
    fig, ax = plt.subplots(1)

    # Call function to create error boxes
    _ = make_error_boxes(ax, x, y, xerr, yerr)

    plt.show()
    # print(xerr.shape)
    # print(yerr.shape)
