from setuptools import setup

setup(
    name='biofes',
    version='1.0',
    description='Biomedical Feature Selection library',
    license='MIT',
    author='Víctor Vicente Palacios',
    author_email='victorvicpal@gmail.com',
    url='https://github.com/victorvicpal/biofes',
    packages=['biofes'],
    install_requires=[
        'numpy>=1.14',
        'scipy>=1.0.0',
        'sklearn>=0.19',
        'pandas>=0.23',
        'matplotlib>=3'
    ],
    extras_require={
        'test': ['unittest']
    },
    python_requires='>=3.6'
)