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
      zip_safe=False)
