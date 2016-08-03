import numpy as np
import itertools
import subprocess
import pysam
import gzip
import os
import pdb

MIN_MAP_QUAL = 10
MAX_VAL = 65535

class ZipFile():
    """File class to rapidly handle some file operations on 
    a gzip file. To greatly increase speed, methods in this class 
    use the system's `zcat` function to read each line of the 
    file, rather than the `gzip` module.

    Arguments
        filename : string
        name of the gzip file to parse.
    """

    def __init__(self, filename):

        if os.path.isfile(filename):
            pipe = subprocess.Popen(["zcat", filename], stdout=subprocess.PIPE)
            self.handle = pipe.stdout
            # remove header
            header = self.handle.next()
        else:
            raise IOError

    def _readline(self):

        for line in self.handle:
            yield line.strip().split('\t')

    def read(self, batch=None):
        """Reads in the lines of the file, either in batches
        or as a whole.

        Optional arguments
            batch : int
            read in `batch` number of lines at a time.

        """

        if batch is None:
            # read the whole file
            locations = [line for line in self._readline()]
        else:
            # read a chunk of the file
            locations = [loc for loc in itertools.islice(self._readline(), 0, batch)]

        for loc in locations:
            loc[1] = int(loc[1])
            loc[2] = int(loc[2])

        return locations

    def close(self): 
      
        pass

class DataTrack():
    """Class to handle file operations on 
    some sequencing data file.

    Arguments
        filename : string
        name of the bam file to parse.

        protocol : string
        DNase_seq / ATAC_seq

    """

    def __init__(self, filename, protocol):

        self._bam_filename = filename
        self._protocol = protocol

        # if tbi file does not exist,
        # convert bam to tbi
        self._track = self.get_tbi_track()

    def get_tbi_track(self):

        tbi_file_prefix = os.path.splitext(self._bam_filename)[0]
        try:
            if self._protocol=="DNase_seq":
                track = DnaseSeq(tbi_file_prefix)
            elif self._protocol=="ATAC_seq":
                track = AtacSeq(tbi_file_prefix)
        except IOError:
            self.convert_bam_to_tbi()
            if self._protocol=="DNase_seq":
                track = DnaseSeq(tbi_file_prefix)
            elif self._protocol=="ATAC_seq":
                track = AtacSeq(tbi_file_prefix)

        return track

    def convert_bam_to_tbi(self):
        """Convert a bam file containing mapped reads to 
        a tabix file containing counts of mapped reads at
        each genomic position.
        """

        tbi_file_prefix = os.path.splitext(self._bam_filename)[0]
    	sam_handle = pysam.Samfile(filename, "rb")

        if self._protocol=='ATAC_seq':

            tbi_filename = tbi_file_prefix
            tbi_handle = open(tbi_filename, 'w')

            for cname,clen in zip(sam_handle.references,sam_handle.lengths):

                # fetch reads in chromosome
                sam_iter = sam_handle.fetch(reference=cname)

                # initialize count array
                counts = dict()
                for read in sam_iter:

                    # skip read if unmapped
                    if read.is_unmapped:
                        continue

                    # skip read, if mapping quality is low
                    if read.mapq < MIN_MAP_QUAL:
                        continue

                    # site of adapter insertions are 9 bp apart
                    # an offset of +4 / -5 gives approximate site of transposition
                    start = read.pos + 4
                    end = read.pos + read.alen - 1 - 5
                    pdb.set_trace()
            
                    try:
                        counts[site] += 1
                    except KeyError:
                        counts[site] = 1

                # write counts to output file
                indices = np.sort(counts.keys())
                for i in indices:
                    count_handle.write('\t'.join([cname,'%d'%i,'%d'%(i+1),'%d'%counts[i]])+'\n')

            tbi_handle.close()

            # compress count file
            bgzip = which("bgzip")
            pipe = subprocess.Popen("%s -f %s"%(bgzip, tbi_filename), \
                                    stdout=subprocess.PIPE, shell=True)
            stdout = pipe.communicate()[0]

            # index count file
            tabix = which("tabix")
            pipe = subprocess.Popen("%s -f -b 2 -e 3 -0 %s.gz"%(tabix, tbi_filename), \
                                    stdout=subprocess.PIPE, shell=True)
            stdout = pipe.communicate()[0]

        elif self._protocol=="DNase_seq":

            tbi_fwd_filename = tbi_file_prefix+'_fwd'
            tbi_rev_filename = tbi_file_prefix+'_rev'
            tbi_fwd_handle = open(tbi_fwd_filename, 'w')
            tbi_rev_handle = open(tbi_rev_filename, 'w')

            for cname,clen in zip(sam_handle.references,sam_handle.lengths):

                # fetch reads in chromosome
                sam_iter = sam_handle.fetch(reference=cname)

                # initialize count array
                fwd_counts = dict()
                rev_counts = dict()
                for read in sam_iter:

                    # skip read if unmapped
                    if read.is_unmapped:
                        continue

                    # skip read, if mapping quality is low
                    if read.mapq < MIN_MAP_QUAL:
                        continue

                    # site of DNase cut
                    if read.strand=='+':
                        site = read.pos
                        try:
                            fwd_counts[site] += 1
                        except KeyError:
                            fwd_counts[site] = 1
                    elif read.strand=='-':
                        site = read.positions[-1] - 1
                        try:
                            rev_counts[site] += 1
                        except KeyError:
                            rev_counts[site] = 1

                # write forward strand counts to output file
                indices = np.sort(fwd_counts.keys())
                for i in indices:
                    tbi_fwd_handle.write('\t'.join([cname,'%d'%i,'%d'%(i+1),'%d'%fwd_counts[i]])+'\n')

                # write forward strand counts to output file
                indices = np.sort(rev_counts.keys())
                for i in indices:
                    tbi_rev_handle.write('\t'.join([cname,'%d'%i,'%d'%(i+1),'%d'%rev_counts[i]])+'\n')

            tbi_fwd_handle.close()
            tbi_rev_handle.close()

            # compress count file
            bgzip = which("bgzip")
            pipe = subprocess.Popen("%s -f %s"%(bgzip, tbi_fwd_filename), \
                                    stdout=subprocess.PIPE, shell=True)
            stdout = pipe.communicate()[0]
            pipe = subprocess.Popen("%s -f %s"%(bgzip, tbi_rev_filename), \
                                    stdout=subprocess.PIPE, shell=True)
            stdout = pipe.communicate()[0]

            # index count file
            tabix = which("tabix")
            pipe = subprocess.Popen("%s -f -b 2 -e 3 -0 %s.gz"%(tabix, tbi_fwd_filename), \
                                    stdout=subprocess.PIPE, shell=True)
            stdout = pipe.communicate()[0]
            pipe = subprocess.Popen("%s -f -b 2 -e 3 -0 %s.gz"%(tabix, tbi_rev_filename), \
                                    stdout=subprocess.PIPE, shell=True)
            stdout = pipe.communicate()[0]

        sam_handle.close()

    def get_read_counts(self, locations, width):
        """Get the number of sequencing reads mapped to
        each base along a window centered at each of
        several motif instances.

        Arguments:
            locations : list
            each entry of the list is a list that specifies 
            information for a motif instance

            width : int
            length of the genomic window around the motif
            instance.

        """
        return self._track.get_read_counts(locations, width)

    def close(self):

        self._track.close()

class DnaseSeq():

    def __init__(self, file_prefix):

        self._fwd_handle = pysam.TabixFile(file_prefix+'_fwd.gz')
        self._rev_handle = pysam.TabixFile(file_prefix+'_rev.gz')

    def get_read_counts(self, locations, width):
        """Get the number of sequencing reads mapped to
        each base along a window centered at each of
        several motif instances.

        Arguments:
            locations : list
            each entry of the list is a list that specifies
            location information for a motif instance

            width : int
            length of the genomic window around the motif
            instance.

        """

        N = len(locations)
        read_counts = np.zeros((N,2*width), dtype='float')

        for l,location in enumerate(locations):

            chromosome = location[0]
            strand = location[3]
            if strand=='+':
                center = location[1]
            else:
                center = location[2]
            left = center-width/2
            right = center+width/2

            # read counts on fwd strand
            fwd_counts = np.zeros((width,), dtype='int')
            fwd_tbx_iter = self._fwd_handle.fetch(chromosome, left, right)
            for tbx in fwd_tbx_iter:
                row = tbx.split('\t')
                count = int(row[3])
                asite = int(row[1]) - left
                fwd_counts[asite] = count

            # read counts on reverse strand
            rev_counts = np.zeros((width,), dtype='int')
            rev_tbx_iter = self._rev_handle.fetch(chromosome, left, right)
            for tbx in rev_tbx_iter:
                row = tbx.split('\t')
                count = int(row[3])
                asite = int(row[1]) - left
                rev_counts[asite] = count

            # flip fwd and rev strand read counts,
            # if the motif is on the opposite strand.
            if strand=='+':
                read_counts[l] = np.hstack((fwd_counts, rev_counts)).astype(np.float64)
            else:
                read_counts[l] = np.hstack((rev_counts[::-1], fwd_counts[::-1])).astype(np.float64)

        read_counts[read_counts>MAX_VAL] = MAX_VAL
        return read_counts

    def close(self):

        self._fwd_handle.close()
        self._rev_handle.close()

class AtacSeq():

    def __init__(self, file_prefix):

        self._handle = pysam.TabixFile(file_prefix+'.gz')

    def get_read_counts(self, locations, width):
        """Get the number of sequencing reads mapped to
        each base along a window centered at each of
        several motif instances.

        Arguments:
            locations : list
            each entry of the list is a list that specifies
            location information for a motif instance

            width : int
            length of the genomic window around the motif
            instance.

        """

        N = len(locations)
        read_counts = np.zeros((N, width), dtype='float')

        for l,location in enumerate(locations):

            chromosome = location[0]
            strand = location[3]
            if strand=='+':
                center = location[1]
            else:
                center = location[2]
            left = center-width/2
            right = center+width/2

            # read counts on fwd strand
            counts = np.zeros((width,), dtype='int')
            tbx_iter = self._handle.fetch(chromosome, left, right)
            for tbx in tbx_iter:
                row = tbx.split('\t')
                count = int(row[3])
                asite = int(row[1]) - left
                counts[asite] = count

            # flip read counts,
            # if the motif is on the opposite strand.
            if strand=='-':
                read_counts[l] = counts[::-1].astype(np.float64)
            else:
                read_counts[l] = counts

        read_counts[read_counts>MAX_VAL] = MAX_VAL

        return read_counts

    def close(self):

        self._handle.close()

def which(program):

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
