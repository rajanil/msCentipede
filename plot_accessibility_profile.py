import numpy as np
import matplotlib.pyplot as plot
import cPickle
import mscentipede
import argparse
import gzip

def plot_profile(footprint_model, background_model, mlen, protocol):

    foreground = np.array([1])
    for j in xrange(footprint_model.J):
        foreground = np.array([p for val in foreground for p in [val,val]])
        vals = np.array([i for v in footprint_model.value[j] for i in [v,1-v]])
        foreground = vals*foreground

    background = np.array([1])
    for j in xrange(background_model.J):
        background = np.array([p for val in background for p in [val,val]])
        vals = np.array([i for v in background_model.value[j] for i in [v,1-v]])
        background = vals*background

    figure = plot.figure()
    subplot = figure.add_axes([0.1,0.1,0.8,0.8])

    L = foreground.size
    if protocol=='DNase_seq':
        footprint = [foreground[:L/2], -1*foreground[L/2:]]
        footprintbg = [background[:L/2], -1*background[L/2:]]
        xvals = np.arange(-L/4,L/4)

        subplot.plot(xvals, footprint[0], linewidth=1, color='b')
        subplot.plot(xvals, footprint[1], linewidth=1, color='b')

        subplot.plot(xvals, footprintbg[0], linewidth=1, color='#888888')
        subplot.plot(xvals, footprintbg[1], linewidth=1, color='#888888')

        ymin = footprint[1].min()
        ymax = footprint[0].max()

        yticks = [ymin, ymin/2, 0, ymax/2, ymax]

    elif protocol=='ATAC_seq':
        footprint = foreground.copy()
        footprintbg = background.copy()
        xvals = np.arange(-L/2,L/2)

        subplot.plot(xvals, footprint, linewidth=1, color='b')
        subplot.plot(xvals, footprintbg, linewidth=1, color='#888888')

        ymin = 0
        ymax = footprint[0].max()

        yticks = [ymin, ymax/2, ymax]

    subplot.axis([xvals.min()-1, xvals.max()+1, 1.01*ymin, 1.01*ymax])

    xticks_right = np.linspace(0,xvals.max()+1,3).astype('int')
    xticks_left = np.linspace(xvals.min(), 0, 3).astype('int')[:-1]
    xticks = [x for x in xticks_left]
    xticks.extend([x for x in xticks_right])
    xticklabels = ['%d'%i for i in xticks]
    subplot.set_xticks(xticks)
    subplot.set_xticklabels(xticklabels, fontsize=8, color='k')

    yticklabels = ['%.2f'%y for y in yticks]
    subplot.set_yticks(yticks)
    subplot.set_yticklabels(yticklabels, fontsize=8, color='k')

    subplot.axvline(0, linestyle='--', linewidth=0.2, color='k')
    subplot.axvline(mlen, linestyle='--', linewidth=0.2, color='k')
    subplot.axhline(0, linestyle='--', linewidth=0.2, color='k')

    figure.text(0.12, 0.88, 'profile at bound sites', \
        color='b', fontsize=9, horizontalalignment='left', verticalalignment='top')
    figure.text(0.12, 0.85, 'profile at unbound sites', \
        color='#888888', fontsize=9, horizontalalignment='left', verticalalignment='top')

    return figure

def parse_args():

    parser = argparse.ArgumentParser(description="plots the cleavage profile, "
        "constructed from the estimated model parameters")

    parser.add_argument("--protocol",
                        choices=("ATAC_seq","DNase_seq"),
                        default="DNase_seq",
                        help="specifies the chromatin accessibility protocol (default:DNase_seq)")

    parser.add_argument("--model",
                        choices=("msCentipede", "msCentipede_flexbg", "msCentipede_flexbgmean"),
                        default="msCentipede",
                        help="models differ in how they capture background rate of enzyme cleavage (default:msCentipede)")

    parser.add_argument("motif_file",
                        action="store",
                        help="name of a gzipped text file containing "
                        " positional information and other attributes for motif instances "
                        " of a transcription factor. columns of the file should be as follows. "
                        " Chromosome Start End Strand PWM_Score [Attribute_1 Attribute_2 ...]. "
                        " additional attributes are optional.")

    options = parser.parse_args()

    # if no motif file is provided, throw an error
    if options.motif_file is None:
        parser.error("Need to provide a file of motifs for a transcription factor")

    return options

def main():

    options = parse_args()
    model_file = "%s_%s_model_parameters.pkl"%(options.motif_file.split('.')[0], '_'.join(options.model.split('-')))
    figure_file = "%s_%s_footprint_profile.pdf"%(options.motif_file.split('.')[0], '_'.join(options.model.split('-')))

    # load model parameters
    handle = open(model_file, 'r')
    model = cPickle.load(handle)
    handle.close()
    footprint_model = model[0]
    background_model = model[2]

    # get motif length
    handle = gzip.open(options.motif_file, 'rb')
    handle.next()
    row = handle.next().strip().split()
    handle.close()
    mlen = int(row[2])-int(row[1])

    # create figure
    figure = plot_profile(footprint_model, background_model, mlen, options.protocol)

    # save figure
    figure.savefig(figure_file, dpi=450)

if __name__=="__main__":

    main()
