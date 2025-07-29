# Install Library

1. (Recommended) Create and activate Python 3 virtual environment

    ```
    # using conda
    $ conda create --name panda-can-decoder python=3.9
    $ conda activate panda-can-decoder
    ```

2. Install Comma ai's [`pandacan`](https://github.com/commaai/panda) library (only need for logging)

    ```
    # from source
    $ git clone https://github.com/commaai/panda.git
    $ cd panda
    $ python setup.py install

    # using pip (avoid this method, the pip version is old and breaks things)
    $ pip install pandacan
    ```

3. Download `/PandaCANDecoder` repository

    ```
    # need to be connected to Argonne network to access
    git clone https://git-in.gss.anl.gov/amtl-mobile-daq/pandacandecoder.git
    ```

4. Install PandaCANDecoder

    ```
    $ cd pandacandecoder

    # for development
    $ python setup.py develop

    # for as-is use
    $ python setup.py install
    ```

5. (Optional) Jupyter Notebook development

    ```
    # install dependancies
    $ conda install ipykernel
    $ conda install jupyter

    # add virtual environment to jupyter
    $ python -m ipykernel install --user --name=panda-can-decoder

    # launch jupyter notebook
    $ jupyter notebook
    ```

    Create new `.ipynb` file using `panda-can-decoder` kernel

_Updated: 08/03/2022_
