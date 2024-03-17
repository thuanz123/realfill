from pathlib import Path
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import math

def pack_rows(rects, scaling_factor = 1, image_height = 700, image_width = 700, order_by_height = False):
    if order_by_height == True:
        rects.sort(key = lambda x: x['height'], reverse = True)
    # Check if rects can be packed in canvas size
    n = len(rects)
    n_width = math.ceil(math.sqrt(n))
    n_height = round(math.sqrt(n))
    predicted_rect_width = scaling_factor*np.sum(np.array([x['width'] for x in rects[:n_width]]))
    predicted_rect_height = scaling_factor*np.sum(np.array([x['height'] for x in rects[:n_height]]))
    width_factor = predicted_rect_width/image_width
    height_factor = predicted_rect_height/image_height
    if width_factor > 1:
        for idx, rect in enumerate(rects):
            rect['width'] /= width_factor
            rects[idx] = rect
    if height_factor > 1:
        for idx, rect in enumerate(rects):
            rect['height'] /= height_factor
            rects[idx] = rect

    xpos = 0
    ypos = 0
    largest_h_this_row = 0
    
    rects_new = []
    for rect in rects:
        rect_new = Rect(rect['x'], rect['y'], rect['height'], rect['width'])
        dx = rect['width']*scaling_factor
        dy = rect['height']*scaling_factor

        if xpos + dx > image_width:
            ypos += largest_h_this_row
            xpos = 0
            largest_h_this_row = 0
        if ypos + dy > image_height:
            break
        rect_new['x'] = xpos
        rect_new['y'] = ypos
        xpos += dx
        if rect['height'] >= largest_h_this_row:
            largest_h_this_row = rect['height']*scaling_factor
        rect_new['was_packed'] = True
        rects_new.append(rect_new)
    return(rects_new)

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
    x = r*np.cos(phi) + image_center[0]
    y = r*np.sin(phi) + image_center[1]
    print(x)
    print(y)
    for idx, elem in enumerate(zip(x,y)):
        rects_new.append(Rect(elem[0], elem[1], widths[idx], heights[idx], was_packed=True))
    return(rects_new)

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


def get_images(image_folder_path):
    ref_data_root = Path(image_folder_path) / "ref"
    train_image_paths = list(Path(ref_data_root).iterdir())
    images = [Image.open(path).convert("RGB") for path in train_image_paths]
    return(images)

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

def generate_layout(images_path, image_height = 1000, image_width = 1000, scaling_factor = 1):
    images = get_images(images_path)
    trans = transforms.Compose([transforms.ToTensor()])
    tensors = [trans(im) for im in images]
    sizes = [tuple(x.size()[1:]) for x in tensors]
    rects = [Rect(0,0,sz[0],sz[1]) for sz in sizes]
    if method == 'rows':
        rects, new_sizes = pack_rows(rects=rects, scaling_factor = scaling_factor, 
                          image_height = image_height, 
                          image_width = image_width,
                          order_by_height = order_by_height)
    elif method == 'circle':
        rects, new_sizes = pack_circle(rects=rects, scaling_factor = scaling_factor, 
                            image_height = image_height, 
                            image_width = image_width)
    else:
        sys.exit('method must be either "rows" (default) or "circle"')
    return([rects, new_sizes])
    

def layout_and_mask(images_path = '/home/feshap/src/realfill/data/noam_photos/Photos-001', 
                    method = 'rows',
                    scaling_factor = 1.2,
                    image_height = 1000, 
                    image_width = 1000,
                    order_by_height = False):

# Make layout according to constraints of input images and target image size
    rects, new_sizes = generate_layout(images_path = images_path, 
                                       image_height = image_height, 
                                       image_width = image_width, 
                                       scaling_factor = scaling_factor)

# Make mask

# Resize input images if necessary
    

    
