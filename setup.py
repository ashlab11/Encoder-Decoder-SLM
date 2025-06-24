from setuptools import setup, find_packages

setup(
    name="Encoder-Decoder-SLM",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "datasets",
        "accelerate",
    ],
)