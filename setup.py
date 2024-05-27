import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hip_attention",
    version="0.0.1",
    author="AUTHOR",
    author_email="EMAIL",
    description="HiP Attention",
    long_description=long_description,
    long_description_content_type="GITHUB LINK",
    url="GITHUB URL",
    project_urls={
        "Bug Tracker": "GITHUB URL",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_dir={"timber": "timber"},
    packages=["timber"],
    python_requires=">=3.6",
)
