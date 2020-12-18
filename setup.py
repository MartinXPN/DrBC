import numpy as np
from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
    name='drbcpp',
    version='0.0.1',
    description='Python package for estimating the high Betweenness Centraliy (BC) nodes in a graph',
    author='Martin Mirakyan',
    author_email='mirakyanmartin@gmail.com',
    python_requires='>=3.8.0,<3.9',
    url='https://github.com/MartinXPN/DrBCPP',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'Cython>=0.29.21',
        'networkx>=2.5',
        'scipy>=1.5.4',
        'numpy>=1.19.4',
        'pandas>=1.1.5',
        'tqdm>=4.54.1',
        'tensorflow>=2.4.0',
        'fire>=0.3.1',
    ],
    extras_require={},

    cmdclass={'build_ext': build_ext},
    ext_package='drbcython',
    ext_modules=cythonize([
        Extension('PrepareBatchGraph', sources=['drbcython/PrepareBatchGraph.pyx', 'cpp/PrepareBatchGraph.cpp', 'cpp/utils.cpp', 'cpp/graph.cpp', 'cpp/graph_struct.cpp', 'cpp/graphUtil.cpp'], language='c++', extra_compile_args=['-std=c++11']),
        Extension('graph', sources=['drbcython/graph.pyx', 'cpp/graph.cpp'], language='c++', extra_compile_args=['-std=c++11']),
        Extension('metrics', sources=['drbcython/metrics.pyx', 'cpp/metrics.cpp', 'cpp/graph.cpp'], language='c++', extra_compile_args=['-std=c++11'], include_dirs=[np.get_include()]),
        Extension('utils', sources=['drbcython/utils.pyx', 'cpp/utils.cpp', 'cpp/graph.cpp', 'cpp/graphUtil.cpp'], language='c++', extra_compile_args=['-std=c++11']),
        Extension('graph_struct', sources=['drbcython/graph_struct.pyx', 'cpp/graph_struct.cpp'], language='c++', extra_compile_args=['-std=c++11']),
    ]),

    include_package_data=True,
    license='MIT',
    classifiers=[
        # Full list of Trove classifiers: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
