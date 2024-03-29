from setuptools import setup


setup(name='mlpro-mpps',
version='1.2.2',
description='MLPro-MPPS - A Customizable Framework for Multi-Purpose Production Systems in Python',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro_mpps'],

# Package dependencies for full installation
extras_require={
    "full": [
        "mlpro[full]>=1.3.1"
    ],
},

zip_safe=False)