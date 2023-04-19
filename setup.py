from setuptools import setup, find_packages 

# Make utils, Unet, RandomForest be global packages
setup(name = 'utils', packages = find_packages())
setup(name = 'UNet', packages = find_packages())
setup(name = 'RandomForest', packages = find_packages())