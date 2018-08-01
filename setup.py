#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages
import gradientboostingencoder


def setup_package():
    metadata = dict()
    metadata['name'] = gradientboostingencoder.__package__
    metadata['version'] = gradientboostingencoder.__version__
    metadata['description'] = gradientboostingencoder.description_
    metadata['author'] = gradientboostingencoder.author_
    metadata['url'] = gradientboostingencoder.url_
    metadata['license'] = 'MIT'
    metadata['packages'] = find_packages()
    metadata['include_package_data'] = False
    metadata['install_requires'] = [
        'scikit-learn',
    ]
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
