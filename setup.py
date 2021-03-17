"""
© 2021, New York University, Tandon School of Engineering, NYU WIRELESS.
"""
from setuptools import setup

setup(name='uav-localization',
      version='1.0',
      packages=['src', 'test', 'src.utils', 'src.positioning'],
      license='© 2021, New York University, Tandon School of Engineering, NYU WIRELESS.',
      description='Module for studying mmWave UAV localization in Dense Urban Scenarios.',
      install_requires=['matplotlib', 'numpy', 'scipy', 'pandas', 'pytest']
      )
