from setuptools import setup

setup(name='dtoolbioimage',
      version='0.1.1',
      description='dtool bioimaging utilties',
      url='https://github.com/JIC-Image-Analysis/dtoolbioimage',
      author='Matthew Hartley',
      author_embioimagel='Matthew.Hartley@jic.ac.uk',
      license='MIT',
      packages=['dtoolbioimage'],
      install_requires=[
          "imageio",
          "dtoolcore",
          "ipywidgets",
          "scipy"
      ],
      zip_safe=False)
