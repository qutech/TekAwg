from setuptools import setup
import re

classifiers = [
    "Programming Language :: Python :: 3",
    "Development Status :: 4 - Beta"
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Topic :: System :: Hardware :: Hardware Drivers",
]

with open("README.md", "r") as fp:
    tek_awg_long_description = fp.read()

with open("tek_awg.py", "r") as f:
    module_contents = f.read()

def extract_version(version_file):
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(name='tek_awg',
      version=extract_version(module_contents),
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
