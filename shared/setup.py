from setuptools import setup, find_packages

setup(
    name="downscalewind-shared",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "zarr>=2.18",
        "numpy>=1.26",
    ],
)
