# msCentipede

**msCentipede** is an algorithm for accurately inferring transcription factor binding sites using chromatin
accessibility data (Dnase-seq, ATAC-seq) and is written in Python2.x. 
The [hierarchical multiscale model underlying msCentipede]() identifies factor-bound genomic sites
by using patterns in DNA cleavage resulting from the action of nucleases in open chromatin regions 
(regions typically bound by transcription factors). msCentipede, 
a generalization of the [CENTIPEDE](http://centipede.uchicago.edu) model, accounts for 
heterogeneity in the DNA cleavage patterns around sites bound by transcription factors.

This code repository contains set of scripts to load the data and run the algorithm. The current document summarizes 
how to download and setup this software package and provides instructions on how to run the software
on a test dataset of motif instances and some publicly available DNase-seq data.

## Dependencies

msCentipede depends on 
+ [Numpy](http://www.numpy.org/)
+ [Scipy](http://www.scipy.org/)
+ [Cython](http://cython.org/)
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
                            information and other attributes for motif instances
                            of a transcription factor. columns of the file should
                            be as follows. Chromosome Start End Strand PWM_Score
                            [Attribute_1 Attribute_2 ...]. additional attributes
                            are optional.
      bam_files             whitespace separated list of bam files from a
                            chromatin accessibility assay

    optional arguments:
      -h, --help            show this help message and exit
      --task {learn,infer}  specify whether to learn model parameters or infer
                            factor binding (default: learn)
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
      --window WINDOW       size of window around the motif instance, where
                            chromatin accessibility profile is used for inferring
                            transcription factor binding. (default: 128)
      --batch BATCH         maximum number of motif instances used for learning
                            model parameters. this is also the number of motif
                            instances on which inference is performed at a time.
                            (default: 10000)
      --bam_file_genomicdna BAM_FILE_GENOMICDNA
                            bam file from a chromatin accessibility assay on
                            genomic DNA
      --seed SEED           set seed for random initialization of parameters

We will now describe in detail how to use this software using an example dataset of CTCF motif instances on chromosome 10 in hg19 coordinates is provided in `test/CTCF_chr10_motifs.txt.gz`. DNase-seq data for the GM12878 cell line (bam and bai files) can be downloaded from ENCODE to `test/` . In the following instructions, we assume the data files are named [Gm12878_Rep1.bam](http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwDnase/wgEncodeUwDnaseGm12878AlnRep1.bam) and [Gm12878_Rep2.bam](http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwDnase/wgEncodeUwDnaseGm12878AlnRep2.bam).

The software is designed to run in two separate steps. In the first step, optimal values for the model parameters are estimated using a subset of all motif instances. In the second step, posterior probability of factor binding is inferred for all motif instances. Since accurate estimates of model parameters can be obtained using 5000-10000 motif instances, this enables efficient inference for those transcription factors that have orders of magnitude more motif instances genomewide. If more motif instances are available in the file than the value of the flag `--batch`, then `batch` number of motif instances that have the highest PWM score are used in learning model parameters.

### Key Inputs

The key inputs that need to be passed to this script are 
+   a path to the file containing the list of motif instances
+   the bam files (sorted and indexed) containing sequencing reads from a chromatin accessibility assay (DNase-seq or ATAC-seq). 

    *Note: these inputs are positional arguments and the files must be specified in the correct order (as shown above).* 

The gzipped file of motif instances should have the following format.

    Chr   Start     Stop      Strand  PwmScore
    chr10 3944439   3944456   +       15.21570492
    chr10 15627426  15627443  -       20.39377594

In the above format, positions are 0-based. *Start* corresponds to the first base of the core motif for *+* strand motif instances and the last base of the core motif for *-* strand motif instances.

When multiple library / sample replicates are available, the bam files for the replicates can be provided as separate files, separated by whitespace. Bam files containing single-end reads and paired-end reads can be mixed since msCentipede currently does not model the fragment size distribution. However, bam files from different protocols or drastically different read lengths are best not mixed since protocol or read length differences could mask biologically meaningful heterogeneity that is relevant in identifying transcription factor binding sites. If the data were generated using an ATAC-seq protocol, the location of transpositions can be automatically identified from the read mapping positions by passing the flag `--protocol=ATAC_seq`.

### Learning model parameters

The model parameters can be learned by passing the following flags.

    python call_binding.py --task learn test/CTCF_chr10_motifs.txt.gz test/Gm12878_Rep1.bam test/Gm12878_Rep2.bam

This will run msCentipede with all other default values and output a log file `test/CTCF_chr10_motifs_msCentipede_log.txt` and a file `test/CTCF_chr10_motifs_msCentipede_model_parameters.pkl` in which the model parameter objects are stored. This is a standard Python pickle file that can be viewed using the `cPickle` module.

### Inferring factor binding

The posterior log odds of binding for a set of motif instances can be computed by passing the following flags.

    python call_binding.py --task infer test/CTCF_chr10_motifs.txt.gz test/Gm12878_Rep1.bam test/Gm12878_Rep2.bam

This will run msCentipede with all other default values and output a file `test/CTCF_chr10_motifs_msCentipede_binding_posterior.txt.gz`.

### Optional parameters

Instead of the default file names, you can specify the file name to which the run log, model parameters and binding posterior odds will be written, using the flags `--log_file`, `--model_file` and `--posterior_file`, respectively.

The differences between the three models *msCentipede* , *msCentipede_flexbgmean* , and *msCentipede_flexbg* are specified in detail in the associated [publication](). If the model flag is specified to be *msCentipede_flexbgmean* or *msCentipede_flexbg*, then a path to a bam file containing chromatin accessibility data from genomic DNA must be passed, using the flag `--bam_file_genomicdna`.

