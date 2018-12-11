import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mip",
    version="1.0.14",
    author="Toffolo, T.A.M. and Santos, H.G.",
    author_email="haroldo@ufop.edu.br",
    description="Python tools for Modeling and Solving Mixed-Integer Programs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuliotoffolo/python-mip",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
