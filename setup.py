import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hip_attention",
    version="0.0.1",
    author="Heejun Lee",
    author_email="gmlwns5176@gmail.com",
    description="HiP Attention",
    long_description=long_description,
    long_description_content_type="gmlwns2000/hip-attention",
    url="https://github.com/gmlwns2000",
    project_urls={
        "Bug Tracker": "https://github.com/gmlwns2000",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_dir={"timber": "timber"},
    packages=["timber"],
    python_requires=">=3.6",
)