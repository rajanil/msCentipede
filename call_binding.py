import numpy as np
import cPickle
import load_data
import mscentipede
import argparse
import time, pdb
#import genome.db

#gdb = genome.db.GenomeDB(path="/data/internal/genome_db",assembly='hg19')
#chromosomes = gdb.get_chromosomes(get_sex=False)

def infer(options):

    # load motif sites
    motif_handle = load_data.ZipFile(options.motif_file)
    locations = [loc for loc in motif_handle.read() if float(loc[4])>13]
    motif_handle.close()
    scores = np.array([loc[4:] for loc in locations]).astype('float')

    # load read data
    bam_handles = [load_data.BamFile(bam_file) for bam_file in options.bam_files]
    count_data = np.array([bam_handle.get_read_counts(locations) for bam_handle in bam_handles])
    ig = [handle.close() for handle in bam_handles]

    #fwd_track = gdb.open_track("encode_uw_dnase/Gm12878_Rep2_fwd")
    #rev_track = gdb.open_track("encode_uw_dnase/Gm12878_Rep2_rev")
    #fwd_dict = dict()
    #rev_dict = dict()
    #for chrom in chromosomes:
    #    fwd_dict[chrom.name] = fwd_track.get_array(chrom)
    #    rev_dict[chrom.name] = rev_track.get_array(chrom)

    #count_data_check = np.array([np.hstack((fwd_dict[loc[0]][loc[1]-100:loc[1]+100], rev_dict[loc[0]][loc[1]-100:loc[1]+100])) if loc[3]=='+' \
    #    else np.hstack((rev_dict[loc[0]][loc[2]-100:loc[2]+100][::-1], fwd_dict[loc[0]][loc[2]-100:loc[2]+100][::-1])) for loc in locations]) 
    #fwd_track.close()
    #rev_track.close()

    #pdb.set_trace()   

    total_counts = np.sum(count_data, 2).T
    if options.window<200:
        counts = np.array([np.hstack((count[:,100-options.window/2:100+options.window/2], \
            count[:,300-options.window/2:300+options.window/2])).T \
            for count in count_data]).T
    else:
        counts = np.array([count.T for count in count_data]).T

    # specify background
    if options.model=='msCentipede':
        background_counts = np.ones((1,2*options.window,1), dtype=float)
    elif options.model in ['msCentipede_flexbgmean','msCentipede_flexbg']:
        bam_handle = load_data.BamFile(options.bam_file_genomicdna)
        bg_count_data = np.array([bam_handle.get_read_counts(locations)])
        bam_handle.close()

        if options.window<200:
            background_counts = np.array([np.hstack((count[:,100-options.window/2:100+options.window/2], \
                count[:,300-options.window/2:300+options.window/2])).T \
                for count in bg_count_data]).T
        else:
            background_counts = np.array([count.T for count in bg_count_data]).T

    # estimate model parameters
    footprint_model, count_model, prior = mscentipede.infer(counts, total_counts, scores, \
        background_counts, options.model, options.restarts, options.mintol)

    # save model parameter estimates
    model_handle = open(options.model_file, 'w')
    cPickle.Pickler(model_handle,protocol=2).dump(footprint_model)
    cPickle.Pickler(model_handle,protocol=2).dump(count_model)
    cPickle.Pickler(model_handle,protocol=2).dump(prior)
    model_handle.close()


def decode(options):

    # load the model
    handle = open(options.model_file, "r")
    footprint_model = cPickle.load(handle)
    count_model = cPickle.load(handle)
    prior = cPickle.load(handle)
    handle.close()

    # load motifs
    motif_handle = load_data.ZipFile(options.motif_file)
    
    # open read data handles
    bam_handles = [load_data.BamFile(bam_file) for bam_file in options.bam_files]

    # open background data handles
    if options.model=='msCentipede':
        background_counts = np.ones((1,2*options.window,1), dtype=float)
    elif options.model in ['msCentipede_flexbgmean','msCentipede_flexbg']:
        bg_handle = load_data.BamFile(options.bam_file_genomicdna)

    # check number of motif sites
    pipe = load_data.subprocess.Popen("zcat %s | wc -l"%options.motif_file, \
        stdout=load_data.subprocess.PIPE, shell=True)
    Ns = int(pipe.communicate()[0].strip())
    loops = Ns/options.batch+1

    handle = load_data.gzip.open(options.posterior_file, "wb")
    header = ['Chr','Start','Stop','Strand','LogPosOdds','LogPriorOdds','MultLikeRatio','NegBinLikeRatio']
    handle.write('\t'.join(header)+'\n')

    for n in xrange(loops):
        starttime = time.time()
        locations = motif_handle.read(batch=options.batch)

        count_data = np.array([bam_handle.get_read_counts(locations) for bam_handle in bam_handles])
        total_counts = np.sum(count_data, 2).T

        scores = np.array([loc[4:] for loc in locations]).astype('float')

        if options.window<200:
            counts = np.array([np.hstack((count[:,100-options.window/2:100+options.window/2], \
                count[:,300-options.window/2:300+options.window/2])).T \
                for count in count_data]).T
        else:
            counts = np.array([count.T for count in count_data]).T

        # specify background
        if options.model in ['msCentipede_flexbgmean','msCentipede_flexbg']:
            bg_count_data = np.array([bg_handle.get_read_counts(locations)])

            if options.window<200:
                background_counts = np.array([np.hstack((count[:,100-options.window/2:100+options.window/2], \
                    count[:,300-options.window/2:300+options.window/2])).T \
                    for count in bg_count_data]).T
            else:
                background_counts = np.array([count.T for count in bg_count_data]).T

        logodds = mscentipede.decode(counts, total_counts, scores, background_counts, \
            footprint_model, count_model, prior, options.model)

        ignore = [loc.extend(['%.4f'%p for p in pos[:4]])
            for loc,pos in zip(locations,logodds)]
        ignore = [handle.write('\t'.join(map(str,elem))+'\n') for elem in locations]
        print len(locations), time.time()-starttime

    handle.close()
    if options.model in ['msCentipede_flexbgmean','msCentipede_flexbg']:
        bg_handle.close()
    ig = [handle.close() for handle in bam_handles]


def parse_args():

    parser = argparse.ArgumentParser(description="""runs msCentipede, to "
        "infer posterior of transcription factor binding, given a set of motif locations.""")

    parser.add_argument("--infer",
                        default=False,
                        action='store_true',
                        help="call the subroutine to infer msCentipede model parameters")

    parser.add_argument("--decode",
                        default=False,
                        action='store_true',
                        help="call the subroutine to compute binding posteriors, given "
                        "msCentipede model parameters")

    parser.add_argument("--model", 
                        choices=("msCentipede", "msCentipede_flexbg", "msCentipede_flexbgmean"),
                        default="msCentipede",
                        help="model for the background rate of chromatin accessibility")

    parser.add_argument("--restarts", 
                        type=int, 
                        default=1, 
                        help="number of re-runs of the algorithm")

    parser.add_argument("--mintol", 
                        type=float, 
                        default=0.01,
                        help="convergence criterion for change in marginal likelihood")

    parser.add_argument("--model_file", 
                        type=str, 
                        default=None, 
                        help="file name to store the model parameters")

    parser.add_argument("--posterior_file", 
                        type=str, 
                        default=None, 
                        help="file name to store the posterior odds ratio, and "
                        "likelihood ratios for each model component, at each motif. ")

    parser.add_argument("--window", 
                        type=int, 
                        default=64, 
                        help="size of window around the motif, where chromatin "
                        "accessibility profile is used for inferring transcription "
                        "factor binding.")

    parser.add_argument("--batch", 
                        type=int, 
                        default=5000, 
                        help="number of motifs to use for learning model parameters; "
                        " also, number of motifs to decode at a time.")

    parser.add_argument("motif_file",
                        action="store",
                        help="name of a gzipped text file containing "
                        " positional information and other attributes for motifs "
                        " of a transcription factor.")

    parser.add_argument("bam_files",
                        action="store",
                        nargs="+",
                        help="comma-separated list of bam files "
                        " from a chromatin accessibility assay ")

    parser.add_argument("--bam_file_genomicdna",
                        action="store",
                        #nargs="+",
                        default=None,
                        help="comma-separated list of bam files "
                        " from a chromatin accessibility assay on genomic DNA")

    options = parser.parse_args()

    # if no motif file is provided, throw an error
    if options.motif_file is None:
        parser.error("Need to provide a file of motifs for a transcription factor")

    # if no model file is provided, create a `default` model file name
    if options.model_file is None:
        options.model_file = options.motif_file.split('.')[0]+"_model_file.pkl"

    # if no posterior file is provided, create a `default` posterior file name
    if options.posterior_file is None:
        options.posterior_file = options.motif_file.split('.')[0]+"_posterior_file.txt.gz"

    return options


def main():

    options = parse_args()

    if options.infer:
        infer(options)

    elif options.decode:
        decode(options)

if __name__=="__main__":

    main()
