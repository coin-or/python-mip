import setuptools


# read the version number 
version_dict = {}
exec(open('mip/constants.py').read(), version_dict)
VERSION = version_dict['VERSION']

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mip",
    version=VERSION,
    author="Toffolo, T.A.M. and Santos, H.G.",
    author_email="haroldo@ufop.edu.br",
    description="Python tools for Modeling and Solving Mixed-Integer Programs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords = ["Optimization", "Linear Programming", "Integer Programming", "Operations Research"],
    url="https://github.com/coin-or/python-mip",
    packages=['mip', 'mip.libraries'],
	package_data = {'mip.libraries' : ['*', '*.*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
