# msCentipede

*msCentipede* is an algorithm for accurately inferring transcription factor binding sites using chromatin
accessibility data (Dnase-seq, ATAC-seq) and is written in Python2.x. 
The hierarchical multiscale model underlying msCentipede uses patterns of DNA cleavage
by nucleases in open chromatin regions (regions typically bound by transcription factors). msCentipede, 
a generalization of the [CENTIPEDE](http://centipede.uchicago.edu) model, accounts for heterogeneity in the DNA cleavage patterns 
around sites bound by transcription factors.
This repo contains set of scripts to load the data and run the algorithm.

This document summarizes how to setup this software package
and run the algorithm on a test dataset of motif instances
and some publicly available DNase-seq data.

## Parts 

This repo has two components: a library of C and Cython scripts *vars* and
a set of Cython and pure Python scripts to load the data and run the algorithm.

## Dependencies

*msCentipede* depends on 
+ [Numpy](http://www.numpy.org/)
+ [Scipy](http://www.scipy.org/)
+ [Cvxopt](http://www.cvxopt.org/)
+ [Pysam](https://github.com/pysam-developers/pysam)

A number of python distributions already have the first two modules packaged in them. It is also
straightforward to install all these dependencies 
 (1) using package managers for MACOSX and several Linux distributions,
 (2) from platform-specific binary packages, and
 (3) directly from source

## Getting the source code

To obtain the source code from github, let us assume you want to clone this repo into a
directory named `proj`:

    mkdir ~/proj
    cd ~/proj
    git clone https://github.com/rajanil/msCentipede

To retrieve the latest code updates, you can do the following:

    cd ~/proj/msCentipede
    git fetch
    git merge origin/master

## Executing the code

The main script you will need to execute is `call_binding.py`. To see command-line 
options that need to be passed to the script, you can do the following:

    $ python call_binding.py

    Here is how you can use this script

    Usage: python call_binding.py   [-h] [--learn] [--infer]
                                    [--model {msCentipede,msCentipede_flexbg,msCentipede_flexbgmean}]
                                    [--restarts RESTARTS] 
                                    [--mintol MINTOL]
                                    [--model_file MODEL_FILE]
                                    [--posterior_file POSTERIOR_FILE] [--window WINDOW]
                                    [--batch BATCH]
                                    [--bam_file_genomicdna BAM_FILE_GENOMICDNA]
                                    motif_file bam_files [bam_files ...]

### Learning model parameters



### Inferring factor binding



## Running on test data

A test simulated dataset is provided in `test/testdata.bed` with genotypes sampled for
200 individuals at 500 SNP loci. The output files in `test/` were generated as follows:

    $ python structure.py -K 3 --input=test/testdata --output=testoutput_simple --full --seed=100
    $ ls test/testoutput_simple*
    test/testoutput_simple.3.log  test/testoutput_simple.3.meanP  test/testoutput_simple.3.meanQ  
    test/testoutput_simple.3.varP  test/testoutput_simple.3.varQ

    $ python structure.py -K 3 --input=test/testdata --output=testoutput_logistic --full --seed=100 --prior=logistic
    $ ls test/testoutput_logistic*
    test/testoutput_logistic.3.log    test/testoutput_logistic.3.meanQ  test/testoutput_logistic.3.varQ
    test/testoutput_logistic.3.meanP  test/testoutput_logistic.3.varP

Executing the code with the provided test data should generate a log file identical to the ones in `test/`, 
as a final check that the source code has been downloaded and compiled correctly.
