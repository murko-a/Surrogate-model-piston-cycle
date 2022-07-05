"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(

    name="piston_sim_fun_surrogate_murkoa",  

    version="0.0.3", 

    description="Python package for surrogate modeling of piston simulation function",

    url="https://github.com/murko-a/Surrogate-model-piston-cycle.git",  

    author="Anze Murko", 

    author_email="anze.murko@outlook.com", 

    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        ],

    package_dir={"": "src"}, 

    packages=find_packages(where="src"),  

    python_requires=">=3.10.2",

)