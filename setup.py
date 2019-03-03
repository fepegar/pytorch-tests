from setuptools import setup, find_packages

# Conda requires: conda install pytorch torchvision -c pytorch

setup(
    name='pytorch_tests',
    version='0.1.0',
    author='Fernando Perez-Garcia',
    author_email='fernando.perezgarcia.17@ucl.ac.uk',
    packages=find_packages(exclude=['*tests']),
    install_requires=[
        # 'namedtensor',
        'tensorflow',
        'tensorboardX',
        'matplotlib',
        'tqdm',
    ],
)
