from setuptools import setup

setup(
    name="studiosr",
    version="0.1.3",
    author="veritross",
    description="Python library to accelerate super-resolution research",
    license="MIT",
    long_description_content_type="text/markdown",
    url="https://github.com/veritross/studiosr",
    packages=[
        "studiosr",
        "studiosr.data",
        "studiosr.engine",
        "studiosr.models",
        "studiosr.utils",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "einops>=0.6.0",
        "gdown==4.6.0",
        "opencv-python>=4.7.0",
        "scikit-image>=0.21.0",
        "timm>=0.9.0",
        "torch>=1.12.0",
    ],
    extras_require={
        "linting": [
            "pre-commit>=3.5.0",
        ],
        "testing": [
            "pytest>=7.4.0",
            "pytest-xdist>=3.5.0",
        ],
    },
    python_requires=">=3.8",
)
