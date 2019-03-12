"""
Install HyQ Gym
"""

from setuptools import setup
from setuptools import find_packages


setup(name='gym_hyq',
      version='1.0',
      description='Gym Environments for the HyQ Robot',
      author='Gabriel Urbain',
      author_email='gabriel.urbain@ugent.be',
      url='https://github.com/gurbain/hyq_ml',
      license='MIT',
      packages=find_packages())
