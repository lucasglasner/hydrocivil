'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner, 
 Modified time: 2024-05-13 17:39:50
 Description:
 Dependencies:
'''

from setuptools import setup, find_packages

setup(
    name="hydrocivil",
    version="0.6",
    author="Lucas Glasner",
    author_email="lgvivanco96@gmail.com",
    description="A package for hydrological methods in civil and environmental engineering",
    long_description=open("README.md", encoding='utf8').read(),
    long_description_content_type="text/markdown",
    url="",  # Add the project URL here if available
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
        "xarray",
        "shapely",
        "geopandas",
        "rioxarray",
        "networkx"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Assuming MIT; adjust if needed
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology"
    ],
)
