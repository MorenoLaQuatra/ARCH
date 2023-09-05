from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='arch_eval',
    version='1.0',
    description='ARCH: Audio Representations benCHmark',
    py_modules=["arch_eval"],
    packages=find_packages(include=['arch_eval', 'arch_eval.*']),
    classifiers={
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    },
    long_description = long_description,
    long_description_content_type = "text/markdown",
    install_requires = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "tqdm",
        "torch",
        "torchaudio",
        "librosa",
        "soundfile",
        "pydub",
        "xmltodict",
        "pyannote.core",
        "pyannote.metrics",
    ],
    extras_require = {
        "dev" : [
            "pytest>=3.7",
        ],
    },
    url="https://github.com/MorenoLaQuatra/ARCH",
    author="Moreno La Quatra",
    author_email="moreno.laquatra@gmail.com",
)