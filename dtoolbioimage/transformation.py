from dtoolbioimage import Image
from dtoolbioimage import ilogging


def create_transformation(func):

    class Transformation(object):

        def __call__(self, *args, **kwargs):
            result = func(*args, **kwargs).view(Image)
            ilogging.info(result, func.__name__)
            return result

    return Transformation()
