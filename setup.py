from setuptools import find_packages, setup

setup(
    name='agwr_library',
    packages=find_packages(include=["agwr_library"]),
    package_data={"agwr_library": ["mgwr/*"]},
    version='0.0.2',
    description='Python AGWR library',
    author='Faizaan',
    install_requires=['access==1.1.9',
 'affine==2.4.0',
 'attrs==23.1.0',
 'bayesian-optimization==1.4.3',
 'beautifulsoup4==4.12.2',
 'certifi==2023.11.17',
 'charset-normalizer==3.3.2',
 'click==8.1.7',
 'click-plugins==1.1.1',
 'cligj==0.7.2',
 'colorama==0.4.6',
 'ConfigSpace==0.7.1',
 'contourpy==1.2.0',
 'cycler==0.12.1',
 'deprecation==2.1.0',
 'esda==2.5.1',
 'fiona==1.9.5',
 'fonttools==4.46.0',
 'geopandas==0.14.1',
 'giddy==2.3.4',
 'idna==3.6',
 'inequality==1.0.1',
 'joblib==1.3.2',
 'kiwisolver==1.4.5',
 'libpysal==4.9.2',
 'llvmlite==0.41.1',
 'mapclassify==2.6.1',
 'matplotlib==3.8.2',
 'mgwr==2.2.0',
 'momepy==0.7.0',
 'more-itertools==10.1.0',
 'mpmath==1.3.0',
 'networkx==3.2.1',
 'numba==0.58.1',
 'numpy==1.26.2',
 'packaging==23.2',
 'pandas==2.1.3',
 'patsy==0.5.4',
 'Pillow==10.1.0',
 'platformdirs==4.1.0',
 'pointpats==2.4.0',
 'PuLP==2.7.0',
 'pyparsing==3.1.1',
 'pyproj==3.6.1',
 'pysal==23.7',
 'python-dateutil==2.8.2',
 'pytz==2023.3.post1',
 'quantecon==0.7.1',
 'rasterio==1.3.9',
 'rasterstats==0.19.0',
 'requests==2.31.0',
 'Rtree==1.1.0',
 'scikit-learn==1.3.2',
 'scipy==1.11.4',
 'seaborn==0.13.0',
 'segregation==2.5',
 'shapely==2.0.2',
 'simplejson==3.19.2',
 'six==1.16.0',
 'snuggs==1.4.7',
 'soupsieve==2.5',
 'spaghetti==1.7.4',
 'spglm==1.1.0',
 'spint==1.0.7',
 'splot==1.1.5.post1',
 'spopt==0.5.0',
 'spreg==1.4.2',
 'spvcm==0.3.0',
 'statsmodels==0.14.0',
 'sympy==1.12',
 'threadpoolctl==3.2.0',
 'tobler==0.11.2',
 'tqdm==4.66.1',
 'typing_extensions==4.8.0',
 'tzdata==2023.3',
 'urllib3==2.1.0',
 'xgboost==2.0.2']
)