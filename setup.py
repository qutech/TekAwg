from setuptools import setup

classifiers = [
    "Programming Language :: Python :: 3",
    "Development Status :: 4 - Beta"
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Topic :: System :: Hardware :: Hardware Drivers",
]

with open("README.md", "r") as fp:
    tek_awg_long_description = fp.read()

setup(name='tek_awg',
      version='0.1',
      author='Simon Humpohl',
      author_email='simon.humpohl@rwth-aachen.de',
      url='https://github.com/qutech/TekAwg/',
      py_modules=['tek_awg'],
      install_requires=['numpy', 'pyvisa'],

      description='Tektronix AWG interface for python',
      long_description=tek_awg_long_description,
      long_description_content_type="text/markdown",
      license="GNU GPLv3",
)
