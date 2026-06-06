from setuptools import setup, find_packages

setup(
    name="plasma-arc",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",        # scipy.ndimage / scipy.signal - imported directly across the pipeline
        "scikit-image",
        "tifffile",     # TIFF chunk I/O - imported directly (analyze_blade, build_velocity_map, ...)
        "matplotlib",
        "tqdm",
        "pycine",
    ],
)
