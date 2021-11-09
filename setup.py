from setuptools import setup
from setuptools import find_namespace_packages


documentation = ["*.md", "*.txt", "*.rst"]


with open("copain/VERSION.txt", "r") as f:
    VERSION = f.read().rstrip()


setup(
    name="copain",
    version=VERSION,
    packages=find_namespace_packages(include=["copain", "copain.*"]),
    install_requires=["numpy", "pyvirtualdisplay <3", "skorch <1"],
    include_package_data=True,
    test_suite="copain",
    url="https://github.com/fcharras/copain",
    author="fcharras",
    maintainer="fcharras",
    description="Interface fceux with python and run reinforcement learning.",
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Other Audience",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Other",
        "Topic :: Artistic Software",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "fceux",
        "IA",
        "reinforcement learning",
        "machine learning",
        "deep learning",
        "TAS",
    ],
)
