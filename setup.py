from setuptools import find_packages, setup

install_requires = [
    "bidict>=0.22.0,<1.0",
    "numpy>=1.22.3,<2",
    "sentencepiece>=0.1.96,<0.2.0",
    "tqdm>=4.64.0,<5",
    "tree_sitter_parsers>=0.0.7,<0.1.0",
]

extra_requires = [
    "tensorflow>=2.8.0,<3.0.0",
    "tensorflow-gpu>=2.8.0,<3.0.0",
    "nltk>=3.7,<4.0",
    "sklearn>=1.0.2,<2.0.0",
]

setup(
    name='infercode',
    version="0.0.29",
    py_modules=['infercode'],
    description='Map any code snippet into vector',
    author='Nghi D. Q. Bui and Yijun Yu',
    author_email='bdqnghi@gmail.com',
    license="MIT",
    url='https://github.com/bdqnghi/infercode/',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
    ],
    package_dir={"infercode": "infercode"},
    packages=find_packages(where=".", exclude=["tests", "logo", "datasets"]),
    package_data={'infercode': ['configs/*.ini', 'sentencepiece_vocab/*', 'sentencepiece_vocab/node_types/*',
                                'sentencepiece_vocab/subtrees/*', 'sentencepiece_vocab/tokens/*']},
    install_requires=install_requires,
    include_package_data=True,
    scripts=['./scripts/infercode'],
)
