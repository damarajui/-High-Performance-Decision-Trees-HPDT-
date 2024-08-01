from setuptools import find_packages, setup

setup(
    name="PDTO",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "cupy",
    ],
)