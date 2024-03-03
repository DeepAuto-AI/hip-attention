from setuptools import setup, find_packages

setup(
    name='HiPAttention',
    version='0.0.1',
    description='Hierarchically Pruned Attention: HiP-Attention',
    author='gmlwns2000',
    author_email='gmlwns5176@gmail.com',
    url='https://github.com/gmlwns2000',
    install_requires=['torch', 'triton', 'transformers'],
    packages=find_packages(exclude=[]),
    keywords=['hip_attention'],
    python_requires='>=3.9',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
