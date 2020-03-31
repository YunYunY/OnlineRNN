import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="onlinernn",
    version="0.0.1",
    author="Yun Yue",
    author_email="yueylink@gmail.com",
    description="Online RNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=["torch==1.4.0", "torchvision==0.5.0"],
    extras_require={
        "dev": [
            "nose==1.3.7", "matplotlib==3.2.1"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubunton",
    ],
    python_requires='>=3.6',
)