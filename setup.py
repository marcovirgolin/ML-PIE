import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='pyNSGP',
    version='0.1',
    author='Marco Virgolin',
    author_email='marco.virgolin@gmail.com',
    url='https://github.com/marcovirgolin/pyNSGP',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['wheel'],
    install_requires=[
	'tensorflow==2.4.1', 'scikit-learn==0.24.1', 'pytexit==0.3.4',
	'sympy==1.7.1','keras==2.4.3',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',

)
