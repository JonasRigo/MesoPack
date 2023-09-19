from setuptools import setup

setup(
        name='MesoPack',
        version='0.3.0',
        description='a simple library to carry out calculations for mesoscopic quantum systems',
        author='Jonas Rigo',
        author_email='rigojonas@gmail.com',
        license='Apache License 2.0',
        packages=['MesoPack'],
        #install_requires['numpy',
        #    'h5py',
        #    'itertools',
        #    'scipy'],

        classifiers=[
            'Development Status :: 1 - Developement',
            'Intended Audience :: Science/Research', 
            'Operating System :: POSIX :: Linux',  
            'Programming Language :: Python :: 3'
            ]
)
