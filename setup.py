from setuptools import setup

setup(name='dtoolbioimage',
      version='0.1.2',
      description='dtool bioimaging utilties',
      url='https://github.com/JIC-Image-Analysis/dtoolbioimage',
      author='Matthew Hartley',
      author_email='Matthew.Hartley@jic.ac.uk',
      license='MIT',
      packages=['dtoolbioimage'],
      install_requires=[
	  "click",
	  "parse",
          "imageio",
          "dtoolcore",
          "ipywidgets",
          "scipy"
      ],
      entry_points='''
        [console_scripts]
        convert_image_dataset=dtoolbioimage.scripts.raw_images_to_image_dataset:cli
      ''',
      zip_safe=False)
