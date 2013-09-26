#!/usr/bin/env python

import sys
import numpy as np
import scipy.spatial.distance as spd 
import scipy.cluster.hierarchy as sph
from scipy import stats
import matplotlib
#matplotlib.use('Agg')
import pylab
import pandas as pd

# samples on rows

class DataMatrix:
    datatype = 'data_matrix'
    
    @staticmethod
    def input_parameters( parser ):
        dm_param = parser.add_argument_group('Input data matrix parameters')
        arg = dm_param.add_argument

        arg( '--sep', type=str, default='\t' )
        arg( '--out_table', type=str, default=None,
             help = 'Write processed data matrix to file' )
        arg( '--fname_row', type=int, default=0,
             help = "row number containing the names of the features "
                    "[default 0, specify -1 if no names are present in the matrix")
        arg( '--sname_row', type=int, default=0,
             help = "column number containing the names of the samples "
                    "[default 0, specify -1 if no names are present in the matrix")
        arg( '--skip_rows', type=str, default=None,
             help = "Row numbers to skip (0-indexed, comma separated) from the input file"
                    "[default None, meaning no rows skipped")
        arg( '--sperc', type=int, default=90, 
             help = "Percentile of sample value distribution for sample selection" )
        arg( '--fperc', type=int, default=90, 
             help = "Percentile of feature value distribution for sample selection" )
        arg( '--stop', type=int, default=None, 
             help = "Number of top samples to select (ordering based on percentile specified by --fperc)" )
        arg( '--ftop', type=int, default=None, 
             help = "Number of top features to select (ordering based on percentile specified by --fperc)" )
        arg( '--def_na', type=float, default=None,
             help = "Set the default value for missing values [default None which means no replacement]")

    def __init__( self, input_file, args ):
        self.args = args
        toskip = [int(l) for l in self.args.skip_rows.split(",")]  if self.args.skip_rows else None
        self.table = pd.read_table( 
                input_file, sep = self.args.sep, # skipinitialspace = True, 
                                  skiprows = toskip,
                                  header = self.args.fname_row if self.args.fname_row > -1 else None,
                                  index_col = self.args.sname_row if self.args.sname_row > -1 else None
                                    )

        def select( perc, top  ): 
            self.table['perc'] = self.table.apply(lambda x: stats.scoreatpercentile(x,perc),axis=1)
            m = sorted(self.table['perc'])[-top]
            self.table = self.table[self.table['perc'] > m ]
            del self.table['perc'] 
        
        if not self.args.def_na is None:
            self.table = self.table.fillna( self.args.def_na )

        if self.args.stop:
            select( self.args.sperc, self.args.stop )
        
        if self.args.ftop:
            self.table = self.table.T 
            select( self.args.fperc, self.args.ftop ) 
            self.table = self.table.T
        

        # add missing values
        
    def get_numpy_matrix( self ): 
        return np.matrix(self.table)
    
    def get_snames( self ):
        return list(self.table.index)
    
    def get_fnames( self ):
        return list(self.table.columns)
   
    def save_matrix( self, output_file ):
        self.table.to_csv( output_file, sep = '\t' )

class DistMatrix:
    datatype = 'distance_matrix'

    @staticmethod
    def input_parameters( parser ):
        dm_param = parser.add_argument_group('Distance parameters')
        arg = dm_param.add_argument

        dist_funcs = [  "euclidean","minkowski","cityblock","seuclidean",
                        "sqeuclidean","cosine","correlation","hamming",
                        "jaccard","chebyshev","canberra","braycurtis",
                        "mahalanobis","yule","matching","dice",
                        "kulsinski","rogerstanimoto","russellrao","sokalmichener",
                        "sokalsneath","wminkowski","ward" ]

        arg( '--f_dist_f', type=str, default="correlation",
             help = "Distance function for features [default correlation]")
        arg( '--s_dist_f', type=str, default="euclidean",
             help = "Distance function for sample [default euclidean]")
    
    def __init__( self, data, feats = None, samples = None, args = None ):
        self.f = feats
        self.s = samples
        
        self.fdf = args.f_dist_f
        self.sdf = args.s_dist_f

        self.f_cdist_matrix = None
        self.s_cdist_matrix = None

        self.numpy_full_matrix = (data if 
                type(data) == np.matrixlib.defmatrix.matrix else None)
    
    def compute_s_dists( self ):
        self.s_cdist_matrix = spd.pdist( self.numpy_full_matrix, self.sdf ) 
    
    def compute_f_dists( self ):
        dt = self.numpy_full_matrix.transpose()
        self.f_cdist_matrix = spd.pdist( dt, self.fdf )

    def get_s_dm( self ):
        return self.s_cdist_matrix

    def get_f_dm( self ):
        return self.f_cdist_matrix

class HClustering:
    datatype = 'hclustering'

    @staticmethod
    def input_parameters( parser ):
        cl_param = parser.add_argument_group('Clustering parameters')
        arg = cl_param.add_argument

        linkage_method = [ "single","complete","average", 
                           "weighted","centroid","median",
                           "ward" ]
        arg( '--no_fclustering', action='store_true',
             help = "avoid clustering features" )
        arg( '--no_sclustering', action='store_true',
             help = "avoid clustering samples" )
        arg( '--flinkage', type=str, default="average",
             help = "Linkage method for feature clustering [default average]")
        arg( '--slinkage', type=str, default="average",
             help = "Linkage method for sample clustering [default average]")

    def get_reordered_matrix( self, matrix, sclustering = True, fclustering = True ):
        idx1 = self.sdendrogram['leaves'][::-1] if sclustering else None
        idx2 = self.fdendrogram['leaves'] if fclustering else None
        #idx1, idx2 = self.sdendrogram['leaves'][::-1], self.fdendrogram['leaves']
        if sclustering and fclustering:
            return matrix[idx1,:][:,idx2]
        if sclustering:
            return matrix[idx1,:][:]
        if sclustering:
            return matrix[:][:,idx2]

    def get_reordered_sample_labels( self, slabels ):
        return [slabels[i] for i in self.sdendrogram['leaves']]

    def get_reordered_feature_labels( self, flabels ):
        return [flabels[i] for i in self.fdendrogram['leaves']]
    
    def __init__( self, s_dm, f_dm, args = None ):
        self.s_dm = s_dm
        self.f_dm = f_dm
        self.args = args
        self.sclusters = None
        self.fclusters = None
        self.sdendrogram = None
        self.fdendrogram = None

    def shcluster( self, dendrogram = True ):
        self.shclusters = sph.linkage( self.s_dm, args.slinkage ) 
        if dendrogram:
            self.sdendrogram = sph.dendrogram( self.shclusters, no_plot=True )

    def fhcluster( self, dendrogram = True ):
        self.fhclusters = sph.linkage( self.f_dm, args.flinkage ) 
        if dendrogram:
            self.fdendrogram = sph.dendrogram( self.fhclusters, no_plot=True )
    
    def get_shclusters( self ):
        return self.shclusters
    
    def get_fhclusters( self ):
        return self.fhclusters
    
    def get_sdendrogram( self ):
        return self.sdendrogram
    
    def get_fdendrogram( self ):
        return self.fdendrogram


class Heatmap:
    datatype = 'heatmap'
   
    bbcyr = {'red':  (  (0.0, 0.0, 0.0),
                        (0.25, 0.0, 0.0),
                        (0.50, 0.0, 0.0),
                        (0.75, 1.0, 1.0),
                        (1.0, 1.0, 1.0)),
             'green': ( (0.0, 0.0, 0.0),
                        (0.25, 0.0, 0.0),
                        (0.50, 1.0, 1.0),
                        (0.75, 1.0, 1.0),
                        (1.0, 0.0, 1.0)),
             'blue': (  (0.0, 0.0, 0.0),
                        (0.25, 1.0, 1.0),
                        (0.50, 1.0, 1.0),
                        (0.75, 0.0, 0.0),
                        (1.0, 0.0, 1.0))}

    bbcry = {'red':  (  (0.0, 0.0, 0.0),
                        (0.25, 0.0, 0.0),
                        (0.50, 0.0, 0.0),
                        (0.75, 1.0, 1.0),
                        (1.0, 1.0, 1.0)),
             'green': ( (0.0, 0.0, 0.0),
                        (0.25, 0.0, 0.0),
                        (0.50, 1.0, 1.0),
                        (0.75, 0.0, 0.0),
                        (1.0, 1.0, 1.0)),
             'blue': (  (0.0, 0.0, 0.0),
                        (0.25, 1.0, 1.0),
                        (0.50, 1.0, 1.0),
                        (0.75, 0.0, 0.0),
                        (1.0, 0.0, 1.0))}

    bcry = {'red':  (   (0.0, 0.0, 0.0),
                        (0.33, 0.0, 0.0),
                        (0.66, 1.0, 1.0),
                        (1.0, 1.0, 1.0)),
             'green': ( (0.0, 0.0, 0.0),
                        (0.33, 1.0, 1.0),
                        (0.66, 0.0, 0.0),
                        (1.0, 1.0, 1.0)),
             'blue': (  (0.0, 1.0, 1.0),
                        (0.33, 1.0, 1.0),
                        (0.66, 0.0, 0.0),
                        (1.0, 0.0, 1.0))}
    

    my_colormaps = [    ('bbcyr',bbcyr),
                        ('bbcry',bbcry),
                        ('bcry',bcry)]
    

    @staticmethod
    def input_parameters( parser ):
        hm_param = parser.add_argument_group('Heatmap options')
        arg = hm_param.add_argument

        arg( '--dpi', type=int, default=150,
             help = "Image resolution in dpi [default 150]")
        arg( '-l', '--log_scale', action='store_true',
             help = "Log scale" )
        arg( '--no_slabels', action='store_true',
             help = "Do not show sample labels" )
        arg( '--minv', type=float, default=None,
             help = "Minimum value to display in the color map [default None meaning automatic]" )
        arg( '--maxv', type=float, default=None,
             help = "Maximum value to display in the color map [default None meaning automatic]" )
        arg( '--no_flabels', action='store_true',
             help = "Do not show feature labels" )
        arg( '--max_slabel_len', type=int, default=25,
             help = "Max number of chars to report for sample labels [default 15]" )
        arg( '--max_flabel_len', type=int, default=25,
             help = "Max number of chars to report for feature labels [default 15]" )
        arg( '--sdend_width', type=float, default=1.0,
             help = "Width of the sample dendrogram [default 1 meaning 100%% of default heatmap width]")
        arg( '--fdend_height', type=float, default=1.0,
             help = "Height of the feature dendrogram [default 1 meaning 100%% of default heatmap width]")
        arg( '--image_size', type=float, default=8,
             help = "Size of the largest between width and eight size for the image in inches [default 8]")
        arg( '--cell_aspect_ratio', type=float, default=1.0,
             help = "Aspect ratio between width and height for the cells of the heatmap [default 1.0]")
        col_maps = ['Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'Dark2', 'GnBu',
                    'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'Paired',
                    'Pastel1', 'Pastel2', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr',
                    'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn',
                    'Reds', 'Set1', 'Set2', 'Set3', 'Spectral', 'YlGn', 'YlGnBu',
                    'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone',
                    'brg', 'bwr', 'cool', 'copper', 'flag', 'gist_earth',
                    'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow',
                    'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray',
                    'hot', 'hsv', 'jet', 'ocean', 'pink', 'prism', 'rainbow',
                    'seismic', 'spectral', 'spring', 'summer', 'terrain', 'winter'] + [n for n,c in Heatmap.my_colormaps]
        for n,c in Heatmap.my_colormaps:
            my_cmap = matplotlib.colors.LinearSegmentedColormap(n,c,256)
            pylab.register_cmap(name=n,cmap=my_cmap)
        arg( '-c','--colormap', type=str, choices = col_maps, default = 'bbcry' )
        arg( '--bottom_c', type=str, default = None,
             help = "Color to use for cells below the minimum value of the scale [default None meaining bottom color of the scale]")

        

        """
        arg( '--', type=str, default="average",
             help = "Linkage method for feature clustering [default average]")
        arg( '--slinkage', type=str, default="average",
             help = "Linkage method for sample clustering [default average]")
        """

    def __init__( self, numpy_matrix, sdendrogram, fdendrogram, snames, fnames, args = None ):
        self.numpy_matrix = numpy_matrix
        self.sdendrogram = sdendrogram
        self.fdendrogram = fdendrogram
        self.snames = snames
        self.fnames = fnames
        self.ns,self.nf = self.numpy_matrix.shape
        self.args = args

    
    
    def draw( self ):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        rat = float(self.ns)/self.nf
        rat *= self.args.cell_aspect_ratio
        x,y = (self.args.image_size,rat*self.args.image_size) if rat < 1 else (self.args.image_size/rat,self.args.image_size)
        fig = plt.figure( figsize=(x,y), facecolor = 'w'  )
        print x,y

        cm = pylab.get_cmap(self.args.colormap)
        bottom_col = [  cm._segmentdata['red'][0][1],
                        cm._segmentdata['green'][0][1],
                        cm._segmentdata['blue'][0][1]   ]
        if self.args.bottom_c:
            bottom_col = self.args.bottom_c
            cm.set_under( bottom_col )

        def make_ticklabels_invisible(ax):
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                 tl.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
      
        def remove_splines( ax ):
            for v in ['right','left','top','bottom']:
                ax.spines[v].set_color('none')

        def shrink_labels( labels, n ):
            shrink = lambda x: x[:n/2]+" [...] "+x[-n/2:]
            return [(shrink(str(l)) if len(str(l)) > n else l) for l in labels]
        

        #gs = gridspec.GridSpec( 4, 2, 
        #                        width_ratios=[1.0-fr_ns,fr_ns], 
        #                        height_ratios=[.03,0.03,1.0-fr_nf,fr_nf], 
        #                        wspace = 0.0, hspace = 0.0 )
        
        fr_ns = float(self.ns)/max([self.ns,self.nf])
        fr_nf = float(self.nf)/max([self.ns,self.nf])
       
        buf_space = 0.05
        minv = min( [buf_space*8, 8*rat*buf_space] )
        print buf_space
        if minv < 0.05:
            buf_space /= minv/0.05
        print buf_space
        
        gs = gridspec.GridSpec( 4, 4, 
                                width_ratios=[ buf_space, buf_space*2, .08*self.args.fdend_height,0.9], 
                                height_ratios=[ buf_space, buf_space*2, .08*self.args.sdend_width,0.9], 
                                wspace = 0.0, hspace = 0.0 )

        ax_hm = plt.subplot(gs[15], axisbg = bottom_col  )
        ax_hm_y2 = ax_hm.twinx() 


        norm_f = matplotlib.colors.LogNorm if self.args.log_scale else matplotlib.colors.Normalize
        minv, maxv = 0.0, None
        im = ax_hm.imshow( self.numpy_matrix, #origin='lower', 
                                interpolation = 'None',  aspect='auto', 
                                extent = [0, self.nf, 0, self.ns], 
                                cmap=cm, 
                                vmin=self.args.minv,
                                vmax=self.args.maxv, 
                                norm = norm_f( vmin=minv if minv > 0.0 else None, vmax=maxv)
                                )
        
        #ax_hm.set_ylim([0,800])
        ax_hm.set_xticks(np.arange(len(fnames))+0.5)
        if not self.args.no_flabels:
            fnames_short = shrink_labels( fnames, self.args.max_flabel_len )
            ax_hm.set_xticklabels(fnames,rotation=90,va='top',ha='center',size=10)
        else:
            ax_hm.set_xticklabels([])
        ax_hm_y2.set_ylim([0,self.ns])
        ax_hm_y2.set_yticks(np.arange(len(snames))+0.5)
        if not self.args.no_slabels:
            snames_short = shrink_labels( snames, self.args.max_slabel_len )
            ax_hm_y2.set_yticklabels(snames_short,va='center',size=10)
        else:
            ax_hm_y2.set_yticklabels( [] )
        ax_hm.set_yticks([])
        remove_splines( ax_hm )
        ax_hm.tick_params(length=0)
        ax_hm_y2.tick_params(length=0)
        #ax_hm.set_xlim([0,self.ns])
        ax_cm = plt.subplot(gs[3], axisbg = 'r', frameon = False)
        fig.colorbar(im, ax_cm, orientation = 'horizontal' )

        if not self.args.no_fclustering:
            ax_den_top = plt.subplot(gs[11], axisbg = 'r', frameon = False)
            sph._plot_dendrogram( self.fdendrogram['icoord'], self.fdendrogram['dcoord'], self.fdendrogram['ivl'],
                                  self.ns + 1, self.nf + 1, 1, 'top', no_labels=True,
                                  color_list=self.fdendrogram['color_list'] )
            ymax = max([max(a) for a in self.fdendrogram['dcoord']])
            ax_den_top.set_ylim([0,ymax])
            make_ticklabels_invisible( ax_den_top )
        if not self.args.no_sclustering:
            ax_den_right = plt.subplot(gs[14], axisbg = 'b', frameon = False)
            sph._plot_dendrogram(   self.sdendrogram['icoord'], self.sdendrogram['dcoord'], self.sdendrogram['ivl'],
                                    self.ns + 1, self.nf + 1, 1, 'right', no_labels=True,
                                    color_list=self.sdendrogram['color_list'] )
            xmax = max([max(a) for a in self.sdendrogram['dcoord']])
            ax_den_right.set_xlim([xmax,0])
            make_ticklabels_invisible( ax_den_right )

        

        #axmatrix.set_ylim(0,100)
        #axmatrix2 = axmatrix.twinx()
        """

        axmatrix.set_xticks([])
        axmatrix2.set_xticks([])
        axmatrix3.set_xticks([])
        axmatrix.set_yticks([])
        axmatrix2.set_yticks([])
        axmatrix3.set_yticks([])
    
        axmatrix.set_xticklabels([])
        axmatrix2.set_xticklabels([])
        axmatrix3.set_xticklabels([])
        axmatrix.set_yticklabels([])
        axmatrix2.set_yticklabels([])
        axmatrix3.set_yticklabels([])
        """
       
        #axmatrix.set_xticks(np.arange(len(fnames)))
        #axmatrix.set_xticklabels(fnames,rotation=90,va='top',ha='center',size=10)
        """
        axmatrix2.set_yticks(np.arange(len(snames)))
        axmatrix2.set_yticklabels(snames,va='center',size=10)
        """

        """
        if not self.args.no_fclustering:
            #axmatrix.set_xticks(np.arange(self.nf)*10+5.0)
            print snames
       
            #axmatrix.set_xticklabels([cols[r] for r in idx2],size=label_font_size,rotation=90,va='top',ha='center')
            ax_fd = divider1.append_axes("top", 0.5, pad=0.0, frameon = False  )
            sph._plot_dendrogram( self.fdendrogram['icoord'], self.fdendrogram['dcoord'], self.fdendrogram['ivl'],
                                  self.ns + 1, self.nf + 1, 1, 'top', no_labels=True,
                                  color_list=self.fdendrogram['color_list'] )
            ax_fd.set_xticks([])
            ax_fd.set_yticks([])
            #ax_fd.set_yticklabels([])
            #ax_fd.set_xticklabels([])
        if not self.args.no_sclustering:
            #axmatrix2.set_ylim([0,self.ns+0.5])
            #axmatrix2 = axmatrix.twinx()
            #axmatrix2.set_yticks(np.arange(self.ns+6)+0.5)
            #axmatrix2.set_yticks(np.arange(self.ns)*0.9+0.5)
            ax_sd = divider1.append_axes("left", 0.5, pad=0.0, frameon = False ) #, sharex = axmatrix )
            sph._plot_dendrogram( self.sdendrogram['icoord'], self.sdendrogram['dcoord'], self.sdendrogram['ivl'],
                                  self.ns + 1, self.nf + 1, 1, 'right', no_labels=True,
                                  color_list=self.sdendrogram['color_list'] )
            ax_sd.set_xticks([])
            ax_sd.set_yticks([])
            #ax_sd.set_xticklabels([])
            #ax_sd.set_yticklabels([])
        """
        
        if not self.args.out:
            plt.show( )
        else:
            fig.savefig( self.args.out, bbox_inches='tight', dpi = self.args.dpi )

class ReadCmd:

    def __init__( self ):
        import argparse as ap
        import textwrap

        p = ap.ArgumentParser( description= "TBA" )
        arg = p.add_argument
        
        arg( '-i', '--inp', '--in', metavar='INPUT_FILE', type=str, nargs='?', default=sys.stdin,
             help= "The input matrix" )
        arg( '-o', '--out', metavar='OUTPUT_FILE', type=str, nargs='?', default=None,
             help= "The output image file [image on screen of not specified]" )

        input_types = [DataMatrix.datatype,DistMatrix.datatype]
        arg( '-t', '--input_type', metavar='INPUT_TYPE', type=str, choices = input_types, 
             default='data_matrix',
             help= "The input type can be a data matrix or distance matrix [default data_matrix]" )

        DataMatrix.input_parameters( p )
        DistMatrix.input_parameters( p )
        HClustering.input_parameters( p )
        Heatmap.input_parameters( p )

        self.args  = p.parse_args()

    def check_consistency( self ):
        pass

    def get_args( self ):
        return self.args

if __name__ == '__main__':
     
    read = ReadCmd( )
    read.check_consistency()
    args = read.get_args()
    
    if args.input_type == DataMatrix.datatype:
        dm = DataMatrix( args.inp, args ) 
        if args.out_table:
            dm.save_matrix( args.out_table )
        
        distm = DistMatrix( dm.get_numpy_matrix(), args = args )
        if not args.no_sclustering:
            distm.compute_s_dists()
        if not args.no_fclustering:
            distm.compute_f_dists()
    elif args.input_type == DataMatrix.datatype:
        # distm = read...
        pass
    else:
        pass

    cl = HClustering( distm.get_s_dm(), distm.get_f_dm(), args = args )
    if not args.no_sclustering:
        cl.shcluster()
    if not args.no_fclustering:
        cl.fhcluster()
    
    hmp = dm.get_numpy_matrix()
    fnames = dm.get_fnames()
    snames = dm.get_snames()
    #if not ( args.no_sclustering or args.no_fclustering ):
    hmp = cl.get_reordered_matrix( hmp, sclustering = not args.no_sclustering, fclustering = not args.no_fclustering  )
    if not args.no_sclustering:
        snames = cl.get_reordered_sample_labels( snames )
    if not args.no_fclustering:
        fnames = cl.get_reordered_feature_labels( fnames )

    hm = Heatmap( hmp, cl.sdendrogram, cl.fdendrogram, snames, fnames, args = args )
    hm.draw()






