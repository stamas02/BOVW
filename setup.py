import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bovw",
    version="0.0.1",
    author="Tamas Suveges",
    author_email="stamas01@gmail.com",
    description="A bag-of-visual-words implementaion using sklearn.cluster.KMemans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stamas02/BOVW/blob/master/bovw/bovw.py",
    packages=["bovw"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["opencv-contrib-python", "tqdm", "scipy", "sklearn"],
)
