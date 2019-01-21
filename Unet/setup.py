import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lung_segmentation_pkg",
    version="0.0.1"
    author="Christos Andrikos",
    author_email="mcchran@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https:// github url ...",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
