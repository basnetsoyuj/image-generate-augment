from PIL import Image, ImageFilter
import os
import random
from tqdm import tqdm

IMG_ARG_ERROR = Exception("Image Argument should be set, tuple, list or string (name of file or directory)")
FILE_OR_DIRECTORY_ERROR = Exception("Please specify an existing file or directory")
IMG_EXTENSIONS = {'jpg', 'jpeg', 'png'}


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
        self.max_lighting = 1
        self.can_blur = True
        self.prob_blur = 0.05
        self.min_coverage = 0.3
        self.max_coverage = 0.8
        self.can_edge_crop = True
        self.edge_crop_prob = 0.3
        self.max_edge_crop = 0.5
        self.min_edge_crop = 0

    def override_default_tfms(self, options):
        pass

    def transform(self, image):
        self.image = image
        self.flip(self.image)
        self.flip_vertical(self.image)
        self.rotate(self.image)
        self.blur(self.image)
        self.edge_crop(self.image)
        return self.image

    def post_transform(self, image, background):
        self.image = image
        self.background = background

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
            return self.image, (left == 0, top == 0, right == w, bottom == h)


class ImageMaker:
    def __init__(self, data, num_samples, **kwargs):
        self.data = data
        self.num_samples = num_samples
        self.tfms = Transformation(kwargs)

    def augment(self, image):
        return self.tfms.transform(image)

    def superimpose(self, image, background, x, y):
        return background.paste(image, (x, y), mask=image)

    def generate(self):
        for _ in tqdm(range(self.num_samples), ascii=True, desc="Progress"):
            image = Image.open(random.choice(self.data.images)).convert('RGBA')
            image = self.augment(image)

            background = Image.open(random.choice(self.data.backgrounds)).convert('RGBA')

            self.superimpose(image, background, 100, 100)
            background.show()


ImageMaker(ImageData('images', 'background'), 5).generate()
