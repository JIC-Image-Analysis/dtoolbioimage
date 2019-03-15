from setuptools import setup

setup(name='dtoolbioimage',
      version='0.1.7',
      description='dtool bioimaging utilties',
      packages=['dtoolbioimage', 'dtoolbioimage.util'],
      url='https://github.com/JIC-Image-Analysis/dtoolbioimage',
      author='Matthew Hartley',
      author_email='Matthew.Hartley@jic.ac.uk',
      license='MIT',
      install_requires=[
	  "click",
	  "parse",
      "imageio",
      "dtoolcore",
      "ipywidgets",
      "scipy",
      "ruamel.yaml",
      ],
      entry_points='''
        [console_scripts]
        convert_image_dataset=dtoolbioimage.convert:cli
      ''',
      zip_safe=False)
