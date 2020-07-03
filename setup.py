import setuptools
from setuptools.command.install import install
from io import open
import os

install_requires = ["numpy", "scipy", "pandas", "matplotlib"]
setuptools.setup(
    name='hclust2',
    version='1.0.0',
    author='Francesco Asnicar',
    author_email='f.asnicar@unitn.it',
    url='http://github.com/SegataLab/hclust2/',
    py_modules=['hclust2'],
    scripts = ['hclust2.py'], 
    #entry_points = { "console_scripts" : [ "hclust2 = hclust2:hclust2_main"] },
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    description='Hclust2 is a handy tool for plotting heat-maps with several useful options to produce high quality figures that can be used in publication',
    install_requires=install_requires
)
