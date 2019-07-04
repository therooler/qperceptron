## Code: Implementing perceptron models with qubits

The Python 3 code to reproduce the figures in the paper ARXIV LINK.

### Getting the plots

This documentation was written with Conda 4.6.12. Please use the provided environment.yml

#### Step 1:

Install Conda: [Quickstart](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)

#### Step 2: Clone the git

`git clone https://gitlab.com/rooler/qperceptron.git`

#### Step 3: Create a Virtual Environment

`conda env create -f environment.yml`

It is good to know that you can use this to update the env:

`conda env update -f environment.yml`

And removing it can be done with:

`conda remove --name qperceptron --all`


#### Step 4: Enter the environment

`conda activate qperceptron`

#### Step 5: Install the package

`python setup.py install clean`

#### Step 6: Run the plot scripts

`python generate_plots/<example>.py`
