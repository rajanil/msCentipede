# msCentipede

*msCentipede* is an algorithm for accurately inferring transcription factor binding sites using chromatin
accessibility data (Dnase-seq, ATAC-seq) and is written in Python2.x. 
The hierarchical multiscale model underlying msCentipede identifies factor-bound genomic sites
by using patterns in DNA cleavage resulting from the action of nucleases in open chromatin regions 
(regions typically bound by transcription factors). msCentipede, 
a generalization of the [CENTIPEDE](http://centipede.uchicago.edu) model, accounts for 
heterogeneity in the DNA cleavage patterns around sites bound by transcription factors.

This repo contains set of scripts to load the data and run the algorithm. This document summarizes 
how to download and setup this software package and provides instructions on how to run the software
on a test dataset of motif instances and some publicly available DNase-seq data.

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

The script you will need to execute is `call_binding.py`. To see command-line 
options that need to be passed to the script, you can do the following:

    $ python call_binding.py

    runs msCentipede, to infer transcription factor binding, given a set of motif
    instances and chromatin accessibility data

    positional arguments:
      motif_file            name of a gzipped text file containing positional
                            information and other attributes for motifs of a
                            transcription factor. Columns of the file should be as
                            follows. Chromosome Start End Strand PWM_Score
                            [Attribute_1 Attribute_2 ...] additional attributes
                            are optional.
      bam_files             comma-separated list of bam files from a chromatin
                            accessibility assay

    optional arguments:
      -h, --help            show this help message and exit
      --task {learn,infer}  specify whether msCentipede is used to learn model
                            parameters or infer factor binding (default: learn)
      --protocol {ATAC_seq,DNase_seq}
                            specifies the chromatin accessibility protocol
                            (default:DNase_seq)
      --model {msCentipede,msCentipede_flexbg,msCentipede_flexbgmean}
                            models differ in how they capture background rate of
                            enzyme cleavage (default:msCentipede)
      --restarts RESTARTS   number of re-runs of the algorithm (default: 1)
      --mintol MINTOL       convergence criterion for change in per-site marginal
                            likelihood (default: 1e-6)
      --model_file MODEL_FILE
                            file name to store the model parameters
      --posterior_file POSTERIOR_FILE
                            file name to store the posterior odds ratio, and
                            likelihood ratios for each model component, at each
                            motif.
      --log_file LOG_FILE   file name to store some statistics of the EM algorithm
                            and a plot of the cleavage profile at bound sites
      --window WINDOW       size of window around the motif, where chromatin
                            accessibility profile is used for inferring
                            transcription factor binding. (default: 128)
      --batch BATCH         number of motifs to use for learning model parameters;
                            also, number of motifs to decode at a time. (default:
                            10000)
      --bam_file_genomicdna BAM_FILE_GENOMICDNA
                            bam file from a chromatin accessibility assay on
                            genomic DNA
      --seed SEED           set seed for random initialization of parameters

We will now describe in detail how to use this software using an example dataset of CTCF motif instances on chromosome 10 in hg19 coordinates is provided in `test/CTCF_chr10_motifs.txt.gz`. DNase-seq data for the GM12878 cell line (bam and bai files) can be downloaded from ENCODE to `test/` (in the following instructions, we assume the data files are named [Gm12878_Rep1.bam]() and [Gm12878_Rep2.bam]())

The key inputs that need to be passed to this script are a path to the file containing the list of motif instances and the bam files (sorted and indexed) containing sequencing reads from a chromatin accessibility assay (DNase-seq or ATAC-seq). Note that the files must be specified in the correct order (as shown above). When multiple library / sample replicates are available, the bam files for the replicates can be provided as separate files, separated by whitespace. 

    python call_binding.py test/CTCF_chr10_motifs.txt.gz test/Gm12878_Rep1.bam test/Gm12878_Rep2.bam

This executes the software to learn the model parametersBam files containing single-end reads and paired-end reads can be mixed since msCentipede currently does not model the fragment size distribution. However, bam files from different protocols or drastically different read lengths are best not mixed since these differences could mask biologically meaningful heterogeneity that is relevant in identifying transcription factor binding sites.

### Learning model parameters

To learn the model parameters, the value of `task` should be set to `learn` and a custom file name can be provided where optimal model parameters will be stored. For example, given a set of CTCF motif instances in `test/CTCF_motifs.txt.gz` and bam files containing 3 replicate DNase-seq data sets in LCLs, we can learn the model parameters using the command

    python call_binding.py --task learn test/CTCF test/LCL_dnase_seq_Rep1.bam test/LCL_dnase_seq_Rep2.bam test/LCL_dnase_seq_Rep3.bam

This will run msCentipede with all other default values and output a file `test/CTCF_model_parameters.pkl`. Alternatively, you can specify the file name to which the model parameters will be stored, using the flag `model_file`.

### Inferring factor binding

To compute the posterior binding odds for a set of motif instances, the value of `task` should be set to `infer` and a file path containing the model parameters should be provided (in addition to other key inputs). For example,

    python call_binding.py --task infer --model_file test/CTCF_model_parameters.pkl test/CTCF test/LCL_dnase_seq_Rep1.bam test/LCL_dnase_seq_Rep2.bam test/LCL_dnase_seq_Rep3.bam

This will run msCentipede with all other default values and output a file `test/CTCF_binding_posterior.txt.gz`. Alternatively, you can specify the file name to which the binding posterior odds and other metrics will be written.

If the model flag is specified to be `msCentipede-flexbgmean` or `msCentipede-flexbg`, then a path to a bam file containing chromatin accessibility data from genomic DNA must be passed to the flag `bam_file_genomicdna`.

## Running on test data

A test dataset of CTCF motif instances in hg19 coordinates is provided in `test/CTCF_motifs.txt.gz`. 
DNase-seq data for the GM12878 cell line, downloaded from ENCODE, are provided in 
`test/Gm12878_Rep1.sort.bam` and `test/Gm12878_Rep2.sort.bam`. The output files in 
`test/` were generated by executing the `msCentipede` model as follows:

    $ python call_binding.py --task learn --model_file test/Ctcf_model_parameters.pkl test/Ctcf_motifs.txt.gz test/Gm12878_Rep1.sort.bam test/Gm12878_Rep2.sort.bam --seed=100
    $ ls test/Ctcf*

    $ python call_binding.py --task infer --model_file test/Ctcf_model_parameters.pkl test/Ctcf_motifs.txt.gz test/Gm12878_Rep1.sort.bam test/Gm12878_Rep2.sort.bam --seed=100
    $ ls test/Ctcf*
