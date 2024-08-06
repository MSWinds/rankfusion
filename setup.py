from setuptools import setup, find_packages

setup(
    name='rankfusion',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
    
        'pandas>=1.5.3',
        'numpy>=1.24.3'
    ],
    author='Yuri Yu',
    author_email='4kmswinds@gmail.com',
    description='A collection of rank fusion algorithms',
    url='https://github.com/MSWinds/rankfusion',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
