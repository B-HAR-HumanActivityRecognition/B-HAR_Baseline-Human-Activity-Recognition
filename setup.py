import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="B-HAR baseline framework",
    version="0.0.1",
    author="B-HAR HumanActivityRecognition",
    author_email="cristian.turetta@univr.it",
    description="B-HAR: an open-source baseline framework for in depth study of human activity recognition datasets and workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/B-HAR-HumanActivityRecognition/B-HAR",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)