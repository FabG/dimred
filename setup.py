import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'dimred'
AUTHOR = 'Fabrice Guillaume'
AUTHOR_EMAIL = 'fabrice.guillaume@email.com'
URL = 'https://github.com/FabG/dimred'

LICENSE = 'MIT'
DESCRIPTION = 'Dimension Reduction and Visualization with PCA SVD, EVD and more'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

PYTHON_REQUIRES = '>=3.6'

INSTALL_REQUIRES = [
      'sklearn',
      'numpy',
      'pandas',
      'matplotlib',
      'scipy',
      'colourmap',
      'pytest',
      'pytest-cov',
      'seaborn',
      'pathlib'
]

CLASSIFIERS = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      classifiers=CLASSIFIERS,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
