from setuptools import setup

setup(
    name="studiosr",
    version="0.0.1",
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
        "einops",
        "gdown",
        "opencv-python",
        "scikit-image",
        "timm",
        "torch",
    ],
    extras_require={
        "linting": [
            "pre-commit",
        ],
        "testing": [
            "pytest",
            "pytest-xdist",
        ],
    },
    python_requires=">=3.6",
)
