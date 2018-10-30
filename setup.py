"""
Install HyQ main libraries
"""

from setuptools import setup
from setuptools import find_packages


setup(name='hyq',
      version='1.0',
      description='Collection of Machine Learning methods for the HyQ Robot ',
      author='Gabriel Urbain',
      author_email='gabriel.urbain@ugent.be',
      url='https://github.com/gurbain/hyq_ml',
      license='MIT',
      packages=find_packages())
