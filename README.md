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

    Usage: python call_binding.py   [-h] 
                                    [--mode {learn,infer} (default: learn)]
                                    [--model {msCentipede,msCentipede_flexbg,msCentipede_flexbgmean} (default: msCentipede)]
                                    [--restarts RESTARTS (default: 1)] 
                                    [--mintol MINTOL (default: 1e-6)]
                                    [--model_file MODEL_FILE (default: None)]
                                    [--posterior_file POSTERIOR_FILE (default: None)]
                                    [--window WINDOW (default: 128)]
                                    [--batch BATCH (default: 10000)]
                                    [--bam_file_genomicdna BAM_FILE_GENOMICDNA (default: None)]
                                    motif_file bam_files [bam_files ...]

The key inputs that need to be passed to this script are a path to the file containing the list of motif instances and the bam files (sorted and indexed) containing sequencing reads from a chromatin accessibility assay (DNase-seq or ATAC-seq). Note that the files must be specified in the correct order (as shown above). When multiple library replicates are available, the bam files for the replicates should be provided as separate files, separated by whitespace. Bam files containing single-end reads and paired-end reads can be mixed since msCentipede currently does not model the fragment size distribution. If the model flag is specified to be `msCentipede-flexbgmean` or `msCentipede-flexbg`, then a path to a bam file containing chromatin accessibility data from genomic DNA must be passed to the flag `bam_file_genomicdna`.

### Learning model parameters

To learn the model parameters, the value of `mode` should be set to `learn` and a custom file name can be provided where optimal model parameters will be stored. For example, given a set of CTCF motif instances in `test/CTCF_motifs.txt.gz` and bam files containing 3 replicate DNase-seq data sets in LCLs, we can learn the model parameters using the command

    python call_binding.py --model learn test/CTCF test/LCL_dnase_seq_Rep1.bam test/LCL_dnase_seq_Rep2.bam test/LCL_dnase_seq_Rep3.bam

This will run msCentipede with all other default values and output a file `test/CTCF_model_parameters.pkl`. Alternatively, you can specify the file name to which the model parameters will be stored, using the flag `model_file`.

### Inferring factor binding

To compute the posterior binding odds for a set of motif instances, the value of `mode` should be set to `infer` and a file path containing the model parameters should be provided (in addition to other key inputs). For example,

    python call_binding.py --model infer --model_file test/CTCF_model_parameters.pkl test/CTCF test/LCL_dnase_seq_Rep1.bam test/LCL_dnase_seq_Rep2.bam test/LCL_dnase_seq_Rep3.bam

This will run msCentipede with all other default values and output a file `test/CTCF_binding_posterior.txt.gz`. Alternatively, you can specify the file name to which the binding posterior odds and other metrics will be written.

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
