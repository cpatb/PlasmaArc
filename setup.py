from setuptools import setup, find_packages

setup(
    name="plasma-arc",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-image",
        "matplotlib",
        "tqdm",
        "pycine",
    ],
)
