import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="coreevolution",
    version="0.1",
    author="Marine Lasbleis, Irene Bonati",
    author_email="marine.lasbleis@gmail.com",
    description="Heat and energy budget for core evolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    package_data={}, # to be filled if needed : 'sample': ['package_data.dat'],
    install_requires=[
        'numpy',
        'pandas',
        "yaml", 
        'scipy',
        'matplotlib'
        ],
)
