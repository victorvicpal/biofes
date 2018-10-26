from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='biofes',
    version='1.0',
    description='Biomedical Feature Selection library',
    license='MIT',
    author='Víctor Vicente Palacios',
    author_email='victorvicpal@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/victorvicpal/biofes',
    packages=['biofes'],
    install_requires=[
        'numpy>=1.14',
        'scipy>=1.0.0',
        'scikit-learn>=0.19.1',
        'pandas>=0.23.4',
        'matplotlib>=3.0.0'
    ],
    extras_require={
        'test': ['unittest']
    },
    python_requires='>=3.5'
)
