import setuptools as st

with open("README.md", "r") as fh:
    long_description = fh.read()

st.setup(
    name='ziptie',
    version='1',
    description='An unsupervised feature learner',
    url='https://gitlab.com/brohrer/ziptie',
    download_url='https://gitlab.com/brohrer/ziptie/tags/',
    author='Brandon Rohrer',
    author_email='brohrer@gmail.com',
    license='MIT',
    install_requires=[
        'matplotlib',
        'numba',
        'numpy',
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={
        "": [
            "README.md",
            "LICENSE",
        ],
        "ziptie": [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
