from setuptools import setup, find_packages

setup(
    name="PandaCANDecoder",
    author='Will Martin',
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'numpy',
        'tqdm',
        'scikit-learn',
        'libusb1'
        ]
)
