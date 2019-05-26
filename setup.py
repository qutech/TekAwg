from setuptools import setup

classifiers = [
    "Programming Language :: Python :: 3",
    "Development Status :: 4 - Beta"
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Topic :: System :: Hardware :: Hardware Drivers",
]

setup(name='tek_awg',
      version='0.1',
      description='Tektronix AWG interface for python',
      author='Simon Humpohl',
      author_email='simon.humpohl@rwth-aachen.de',
      url='https://github.com/qutech/TekAwg/',
      py_modules=['tek_awg'],
      install_requires=['numpy', 'pyvisa']
)
