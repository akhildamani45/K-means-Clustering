"""
Visualization App to verify that k-means works

The visualize can view any clustering on a set of 2d points. The visualization is
limited to k-values < 20.

Author: Walker M. White (wmw2)
Date: October 26, 2021
"""
import matplotlib
import numpy
import math
import traceback
matplotlib.use('TkAgg')

# Modules to embed matplotlib in a custom Tkinter window
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# File support to load data files
import tkinter as tk
from tkinter import font, filedialog, messagebox
import sys, os, os.path

# The k-means implementation
import a6dataset
import a6cluster
import a6algorithm
import tools


class Visualizer(object):
    """
    A class providing a visualization app.

    This class has many attributes for GUI widgets (buttons and labels). We do not list 
    all of them here.
    """
    # INSTANCE ATTRIBUTES:
    # Attribute _root: The TCL/TK graphics backend 
    # Invariation: _root is a TK object
    #
    # Attribute _canvas: The MatPlotLib canvas 
    # Invariation: _canvas is a FigureCanvas object
    #
    # Attribute _axes: The MatPlotLib axes 
    # Invariation: _axes is a Scatter object
    #
    # Attribute _dset: The data set
    # Invariant: _dset is a Dataset object
    #
    # Attribute _kmean: The clustering algorithm
    # Invariant: _kmean is an Algorithm object
    #
    # Attribute _count: The number of sets computed so far
    # Invariant: _count is an int >= 0
    #
    # Attribute _finish: Whether the computation is done
    # Invariatn: _finish is a boolean
    
    # Maximum allowable k-means
    MAX_KVAL = 20

    # The cluster colors
    COLORS = ((1,0,0),(0,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,0),(1,0.5,0),(0.3,0.5,0.3),(1,0.6,0.7),(0,0,0))

    @classmethod
    def launch(cls,filename,k):
        """
        Launches the visualizer and starts the application loop.

        Parameter filename: The name of the initial dataset
        Precondition: filename is a valid file path OR None.

        Parameter k: The initial number of clusters
        Precondition: k is an int
        """
        cls(filename,k)
        tk.mainloop()

    def __init__(self, filename=None, k=3):
        """
        Initializes a visualization app.

        The initial dataset and k value are optional.  By default, it will
        choose the first dataset from the dataset directory.

        Parameter filename: The name of the initial dataset
        Precondition: filename is a valid file path OR None.

        Parameter k: The initial number of clusters
        Precondition: k is an int
        """
        self._root = tk.Tk()
        self._root.wm_title("Clustering Visualizer")
        self._dset  = None
        self._kmean = None

        # Start the application
        self._config_canvas()
        self._config_control()
        if not k is None:
            self._kval.set(k)

        if filename:
            self._select_data(filename,False)
        else:
            self._select_data()
        self._canvas.draw()

    def _config_canvas(self):
        """
        Loads the MatPlotLib drawing code
        """
        # Create the drawing canvas
        figure = Figure(figsize=(6,6), dpi=100)
        self._canvas = FigureCanvasTkAgg(figure, master=self._root)
        self._canvas._tkcanvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Initialize the scatter plot
        self._axes = figure.gca()
        self._axes.set_xlim((0.0, 1.0))
        self._axes.set_ylim((0.0, 1.0))
        self._axes.set_xlabel('X')
        label = self._axes.set_ylabel('Y')
        label.set_rotation(0)
        self._axes.set_xticks(numpy.arange(0.0,1.0,0.1))
        self._axes.set_yticks(numpy.arange(0.0,1.0,0.1))
        self._axes.tick_params(labelsize=9)

    def _config_control(self):
        """
        Creates the control panel on the right hand side

        This method is WAY too long, but GUI layout code is typically like this. Plus,
        Tkinter makes this even worse than it should be.
        """
        panel = tk.Frame(master=self._root)
        panel.columnconfigure(0,pad=3)
        panel.columnconfigure(1,pad=3,minsize=150)
        panel.rowconfigure(0,pad=3)
        panel.rowconfigure(1,pad=0)
        panel.rowconfigure(2,pad=23)
        panel.rowconfigure(3,pad=3)
        panel.rowconfigure(4,pad=3)
        panel.rowconfigure(5,pad=3)
        panel.rowconfigure(6,pad=3)
        panel.rowconfigure(7,pad=13)
        panel.columnconfigure(2,minsize=20)

        title = tk.Label(master=panel,text='K Means Control',height=3)
        wfont = font.Font(font=title['font'])
        wfont.config(weight='bold',size=20)
        title.grid(row=0,columnspan=2, sticky='we')
        title.config(font=wfont)

        divider = tk.Frame(master=panel,height=2, bd=1, relief=tk.SUNKEN)
        divider.grid(row=1,columnspan=2, sticky='we')

        # Label and button for managing files.
        label = tk.Label(master=panel,text='Data Set: ',height=2)
        wfont = font.Font(font=label['font'])
        wfont.config(weight='bold')
        label.config(font=wfont)
        label.grid(row=2,column=0, sticky='e')

        files = tools.list_csv(os.path.join(os.path.split(__file__)[0],'data'),'-2d')
        files.append('<select file>')
        self._kfile = tk.StringVar(master=self._root)
        self._kfile.set(files[0])
        options = tk.OptionMenu(panel,self._kfile,*files,command=self._select_data)
        options.grid(row=2,column=1,sticky='w')

        # Label and option menu to select k-value
        label = tk.Label(master=panel,text='K Value: ',height=2,font=wfont)
        label.grid(row=3,column=0,sticky='e')

        self._kval = tk.IntVar(master=self._root)
        self._kval.set(3)
        options = tk.OptionMenu(panel,self._kval,*range(1,self.MAX_KVAL+1),command=self._reset)
        options.grid(row=3,column=1,sticky='w')

        # Radius Flag
        label = tk.Label(master=panel,text='Overlay: ',height=2,font=wfont)
        label.grid(row=4,column=0,sticky='e')

        self._kbool = tk.StringVar(master=self._root)
        self._kbool.set('False')
        options = tk.OptionMenu(panel,self._kbool,'False','True',command=self._replot)
        options.grid(row=4,column=1,sticky='w')

        # Label and step indicator
        label = tk.Label(master=panel,text='At Step: ',height=2,font=wfont)
        label.grid(row=5,column=0,sticky='e')

        self._count = 0
        self._countlabel = tk.Label(master=panel,text='0')
        self._countlabel.grid(row=5,column=1,sticky='w')

        # Label and convergence indicator
        label = tk.Label(master=panel,text='Finished: ',height=2,font=wfont)
        label.grid(row=6,column=0,sticky='e')

        self._finished = False
        self._finishlabel = tk.Label(master=panel,text='False')
        self._finishlabel.grid(row=6,column=1,sticky='w')

        # Control buttons
        button = tk.Button(master=panel, text='Reset', width=8, command=self._reset)
        button.grid(row=7,column=0,padx=(10,0))
        button = tk.Button(master=panel, text='Step', width=8, command=self._step)
        button.grid(row=7,column=1)

        panel.pack(side=tk.RIGHT, fill=tk.Y)

    def _plot_clusters(self):
        """
        Plots the clusters in a completed assignment
        """
        for k in range(self._kval.get()):
            c = self.COLORS[k % len(self.COLORS)]
            m = 'x' if (k//10) % 2 == 1 else '+'
            cluster = self._kmean.getClusters()[k]
            rows = numpy.array(cluster.getContents())
            cent = cluster.getCentroid()
            if self._kbool.get().lower() == 'true':
                rads = cluster.getRadius()
                rads = (500*rads)**2 if rads else 10
                opac = 0.5
            else:
                rads = 50
                opac = 1.0
            if (len(rows) > 0):
                self._axes.scatter(rows[:,0], rows[:,1], color=c, marker=m)
            c = list(c)+[opac]
            self._axes.scatter(cent[0],cent[1],color=c,edgecolors=(0,0,0,opac),s=rads,marker='o')

    def _plot_one_cluster(self):
        """
        Plots one cluster in an assignment that has finished Cluster but not Clustering.
        """
        # Try to show everything in one cluster.
        cluster = a6cluster.Cluster(self._dset, self._dset.getPoint(0))
        for i in range(self._dset.getSize()):
            cluster.addIndex(i)
        cluster.update()
        rows = numpy.array(self._dset.getContents())
        cent = cluster.getCentroid()
        if self._kbool.get().lower() == 'true':
            rads = cluster.getRadius()
            rads = (500*rads)**2 if rads else 10
            opac = [0,0,1,0.5]
        else:
            rads = 30
            opac = [0,0,1,1]
        if (len(rows) > 0):
            self._axes.scatter(rows[:,0], rows[:,1], c='b', marker='+')
        self._axes.scatter(cent[0],cent[1],color=opac,edgecolors=(0,0,0),s=rads,marker='o')

    def _plot_points(self):
        """
        Plots the clusters in an assignment that has finished Dataset but not much else.
        """
        rows = numpy.array(self._dset.getContents())
        self._axes.scatter(rows[:,0], rows[:,1], c='k', marker='+')

    def _plot(self):
        """
        Plots the data as it can.
        
        This function replots the data any time that it changes.  It limits what it
        plots to whatever the user has implemented.
        """
        assert not self._dset is None, 'Invariant Violation: Attempted to plot when data set is None'
        
        self._axes.clear()
        if self._kmean is not None:
            try:
                self._plot_clusters()
            except BaseException as e:
                print('FAILED KMEANS VISUALIZATION: ')
                traceback.print_exc()
                print()
                print('Attempting One Cluster Only')
                try:
                    self._plot_one_cluster()
                except BaseException as e:
                    print('FAILED CLUSTER VISUALIZATION ')
                    traceback.print_exc()
                    print()
                    print('Attempting Data Set Only')
                    self._plot_points()
        else:
            self._plot_points()
        
        # Reset axes information
        xb = self._axes.get_xbound()
        xb = (numpy.floor(xb[0]*10)/10.0,numpy.ceil(xb[1]*10)/10.0)
        self._axes.set_xlim(xb)
        self._axes.set_xticks(numpy.arange(xb[0],xb[1],0.1))
        
        yb = self._axes.get_ybound()
        yb = (numpy.floor(yb[0]*10)/10.0,numpy.ceil(yb[1]*10)/10.0)
        self._axes.set_ylim(yb)
        self._axes.set_yticks(numpy.arange(yb[0],yb[1],0.1))
        
        self._axes.set_xlabel('X')
        self._axes.set_ylabel('Y')
        
        self._canvas.draw()

    def _select_data(self,file=None,local=True):
        """
        Selects a data set, either from the data directory or user choice
        
        Parameter file: The (local) file for the data set
        Precondition: file is a string.  It is either '<select file>', the name
        of a file, or a prefix of a 2d data set in the data directory.
        
        Parameter local: Whether to chose the file from the data directory
        Precondition: local is a boolean
        """
        if file is None:
            file = self._kfile.get()
        if file == '<select file>':
            filename = filedialog.askopenfilename(initialdir='.',title='Select a Data File',
                                                    filetypes=[('CSV Data Files', '.csv')])
            self._kfile.set(self._shortname(filename))
        elif local:
            filename = os.path.join(os.path.split(__file__)[0],'data',file+'-2d.csv')
        else:
            filename = file
            self._kfile.set(self._shortname(filename))
        
        try:
            contents = tools.data_for_file(filename)
            message  = None
            if len(contents) == 0 or len(contents[0]) == 0:
                messagebox.showwarning('Load','The dataset is empty')
                return
            elif len(contents[0]) == 1:
                message = 'The data is one-dimensional.\nThe y values will be 0.5.'
                for pos in range(len(contents)):
                    contents[pos].append(0.5)
            elif len(contents[0]) > 2:
                message = 'The data is high dimensional.\nOnly the first two columns are used.'
                for pos in range(len(contents)):
                    contents[pos] = contents[pos][:2]
            self._load_data(contents)
            if message:
                messagebox.showwarning('Load',message)
        except AssertionError as e:
            messagebox.showwarning('Load',str(e))
        except:
            traceback.print_exc()
            messagebox.showwarning('Load','ERROR: An unknown error occurred.')

    def _load_data(self, contents):
        """
        Loads a data set file into a Dataset object.
        
        Parameter contents: The contents of the dataset
        Precondition: contents is a 2d rectangular table
        """
        try:
            self._dset = a6dataset.Dataset(2,contents)
            if not self._dset.getContents():
                raise RuntimeError()
            #self._filebutton.configure(text=shortname)
            self._kmean = None
            self._reset()
            self._plot()
        except RuntimeError:
            messagebox.showwarning('Load','ERROR: You must complete Dataset first.')
        except:
            traceback.print_exc()
            messagebox.showwarning('Load','ERROR: An unknown error occurred.')

    def _reset(self,k=None):
        """
        Resets the k-means calculation with the given k value.
        
        If k is None, it uses the value of self._kval.
        
        Parameter: k the k-means number of clusters
        Precondition: k > 0 is an int, and a dataset with at least k
        points is loaded.  If k is None, the current value of self._kval
        is used.
        """
        if k is None:
            k = self._kval.get()
        if self._dset is None:
            messagebox.showwarning('Reset','ERROR: No data set loaded.')

        self._count = 0
        self._countlabel.configure(text='0')
        self._finished = False
        self._finishlabel.configure(text='False')

        # Student may not have implemented this yet.
        self._kmean = a6algorithm.Algorithm(self._dset, k)
        self._kmean._partition()
        self._plot()

    def _replot(self,overlay=None):
        """
        Refreshes the data plot.
        
        This is necessary when we toggle cluster overlays.
        
        Parameter overlap: Whether to display a transparent overlay in the cluster
        Precondition: overlap is a boolean
        """
        if self._dset is None:
            messagebox.showwarning('Replot','ERROR: No data set loaded.')
        self._plot()

    def _step(self):
        """
        Performs one step in k-means clustering
        """
        if self._dset is None:
            tk.messagebox.showwarning('Step','ERROR: No data set loaded.')
        if self._kmean is None:
            self._reset()
        if self._finished:
            return
        
        self._count = self._count+1
        self._countlabel.configure(text=str(self._count))
        self._finished = self._kmean.step()
        self._finishlabel.configure(text=str(self._finished))
        
        self._plot()

    def _shortname(self,filename):
        """
        Returns the short name of a file.
        
        This is used to display the active file, when possible. It removes any
        parent directories, any file type information, and shortens the name to
        10 characters.
        
        Parameter filename: The name of the file
        Precondition: filename is a string representing a valid file path
        """
        name = os.path.split(filename)[1]
        name = os.path.splitext(name)[0]
        if (len(name) > 10):
            name = name[0:10]+'...'
        return name
