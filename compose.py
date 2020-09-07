from PIL import Image, ImageFilter, ImageEnhance
import os
import random
from tqdm import tqdm
from utils import *


class ImageData:
    def __init__(self, img_files, bg_files):
        self.images, self.backgrounds = self.load_images(img_files, bg_files)

    def load_images(self, img_files, bg_files):
        data = []
        for file in (img_files, bg_files):
            if isinstance(file, list) or isinstance(file, tuple) or isinstance(file, set):
                data.append(list(file))
            elif isinstance(file, str):
                if os.path.isfile(file):
                    data.append(self.load_from_file(file))
                elif os.path.isdir(file):
                    data.append(self.load_from_directory(file))
                else:
                    raise FILE_OR_DIRECTORY_ERROR
            else:
                raise IMG_ARG_ERROR
        return data

    def load_from_file(self, file):
        if self.check_if_img(file):
            return file
        with open(file, 'r') as f:
            return f.readlines()

    def load_from_directory(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if self.check_if_img(f)]

    def check_if_img(self, name):
        return name.split('.')[-1].lower() in IMG_EXTENSIONS


class Transformation:
    def __init__(self, options):
        self.reset_tfms()
        self.override_default_tfms(options)

    def reset_tfms(self):
        self.can_flip = True
        self.can_flip_vertical = True
        self.prob_flip = 0.5
        self.max_rotate = 90.0
        self.can_lighten = True
        self.max_lighting = 0.6  # 60%
        self.lighting_prob = 0.4
        self.can_blur = True
        self.prob_blur = 0.05
        self.min_coverage = 0.95
        self.max_coverage = 0.99
        self.can_edge_crop = True
        self.edge_crop_prob = 0.3
        self.max_edge_crop = 0.5
        self.min_edge_crop = 0
        self.can_stretch = True
        self.stretch_prob = 0.4

    def override_default_tfms(self, options):
        pass

    def transform(self, image):
        self.image = image
        self.flip(self.image)
        self.flip_vertical(self.image)
        self.rotate(self.image)
        self.blur(self.image)
        return self.image

    def post_transform(self, image, background):
        self.image = image
        self.background = background
        _, cropped_axis = self.edge_crop(image, True)
        self.resize(self.image, background.size)
        self.superimpose(self.image, self.background, cropped_axis)
        self.brighten(self.image)
        self.contrast(self.image)

        return self.image

    @property
    def random(self):
        return random.random()

    def random_int(self, x, y):
        x, y = int(x), int(y)
        if x > y:
            x, y = y, x
        return random.randint(x, y)

    def flip(self, image):
        if self.can_flip and self.random < self.prob_flip:
            self.image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return self.image

    def flip_vertical(self, image):
        if self.can_flip and self.can_flip_vertical and self.random > self.prob_flip:
            self.image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return self.image

    def rotate(self, image):
        self.image = image.rotate(self.random_int(-self.max_rotate, self.max_rotate), expand=True)
        return self.image

    def blur(self, image):
        if self.can_blur and self.random < self.prob_blur:
            self.image = image.filter(ImageFilter.BLUR)

    def edge_crop(self, image, return_which_cropped=False):
        if self.can_edge_crop:
            w, h = image.size
            left, top, right, bottom = 0, 0, w, h
            if self.random < self.edge_crop_prob:
                amt_crop = self.random_int(self.min_edge_crop * w, self.max_edge_crop * w)
                if self.random > 0.5:
                    left = amt_crop
                else:
                    right = w - amt_crop
            if self.random < self.edge_crop_prob:
                amt_crop = self.random_int(self.min_edge_crop * h, self.max_edge_crop * h)
                if self.random > 0.5:
                    top = amt_crop
                else:
                    bottom = h - amt_crop
            self.image = image.crop((left, top, right, bottom))
        if not return_which_cropped:
            return self.image
        else:
            return self.image, (left != 0, top != 0, right != w, bottom != h)

    def resize(self, image, canvas_size):
        min_width, min_height = [self.min_coverage * x for x in canvas_size]
        max_width, max_height = [self.max_coverage * x for x in canvas_size]
        new_width, new_height = [self.random_int(min_width, max_width), self.random_int(min_height, max_height)]
        if self.can_stretch and self.random < self.stretch_prob:
            self.image = image.resize([new_width, new_height], Image.ANTIALIAS)
        else:
            image.thumbnail([new_width, new_height], Image.ANTIALIAS)
        self.image = image
        return self.image

    def change_lighting(self, image, func):
        if self.random < self.lighting_prob:
            enhancer = func(image)
            pct = self.random * self.max_lighting
            factor = 1 + pct if self.random < 0.5 else 1 - pct
            self.image = enhancer.enhance(factor)
        return self.image

    def brighten(self, image):
        return self.change_lighting(image, ImageEnhance.Brightness)

    def contrast(self, image):
        return self.change_lighting(image, ImageEnhance.Contrast)

    def superimpose(self, image, background, where_info=[0, 0, 0, 0]):
        x_options, y_options = background.size[0] - image.size[0], background.size[1] - image.size[1]
        x, y = self.random_int(0, x_options), self.random_int(0, y_options)
        if where_info[0] or where_info[2]:
            x = where_info[2] * x_options
        if where_info[1] or where_info[3]:
            y = where_info[3] * y_options
        background.paste(image, (x, y), mask=image)
        self.image = background
        return background

class ImageMaker:
    def __init__(self, data, num_samples, **kwargs):
        self.data = data
        self.num_samples = num_samples
        self.tfms = Transformation(kwargs)

    def augment(self, image):
        return self.tfms.transform(image)

    def generate(self):
        for _ in tqdm(range(self.num_samples), ascii=True, desc="Progress"):
            image_location = random.choice(self.data.images)
            image = Image.open(image_location).convert('RGBA')
            image = self.augment(image)

            bg_location = random.choice(self.data.backgrounds)
            background = Image.open(bg_location).convert('RGBA')

            new_img = self.tfms.post_transform(image, background)
            new_img.show()


    def save(self, image, location):
        image.save(location)

ImageMaker(ImageData('images', 'background'), 10).generate()
