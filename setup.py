"""
Install HyQ Gym
"""

from setuptools import setup
from setuptools import find_packages


setup(name='hyq_gym',
      version='1.0',
      description='Gym Environments and RL Agents for the HyQ Robot',
      author='Gabriel Urbain',
      author_email='gabriel.urbain@ugent.be',
      url='https://github.com/gurbain/hyq_ml',
      license='MIT',
      install_requires=['gym'],
      packages=find_packages())