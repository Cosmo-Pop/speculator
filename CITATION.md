# Citing Speculator
If you make use of Speculator, please cite the papers describing the code, its dependencies, and the papers describing any trained models you use. Below is a string of LaTeX code that could be used in, e.g., a software acknowledgements section.
```latex
\texttt{numpy} \citep{harris20};
\texttt{sklearn} \citep{pedregosa11};
\texttt{speculator} \citep{alsing20};
\texttt{torch} \citep{paszke19}.
```
We would also encourage users to acknowledge `fsps`, `prospector`, and `sedpy`, upon which Speculator is based.
```latex
\texttt{fsps} \citep{conroy09, conroy10a, conroy10b};
\texttt{prospector} \citep{johnson21a};
\texttt{python-fsps} \citep{johnson21b};
\texttt{sedpy} \citep{johnson21c}.
```
BibTeX entries for all of these references are included below, based on NASA ADS.
```bibtex
@ARTICLE{alsing20,
       author = {{Alsing}, Justin and {Peiris}, Hiranya and {Leja}, Joel and {Hahn}, ChangHoon and {Tojeiro}, Rita and {Mortlock}, Daniel and {Leistedt}, Boris and {Johnson}, Benjamin D. and {Conroy}, Charlie},
        title = "{SPECULATOR: Emulating Stellar Population Synthesis for Fast and Accurate Galaxy Spectra and Photometry}",
      journal = {\apjs},
     keywords = {Galaxies, Neural networks, Galaxy photometry, 573, 1933, 611, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2020,
        month = jul,
       volume = {249},
       number = {1},
          eid = {5},
        pages = {5},
          doi = {10.3847/1538-4365/ab917f},
archivePrefix = {arXiv},
       eprint = {1911.11778},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020ApJS..249....5A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{conroy09,
       author = {{Conroy}, Charlie and {Gunn}, James E. and {White}, Martin},
        title = "{The Propagation of Uncertainties in Stellar Population Synthesis Modeling. I. The Relevance of Uncertain Aspects of Stellar Evolution and the Initial Mass Function to the Derived Physical Properties of Galaxies}",
      journal = {\apj},
     keywords = {galaxies: evolution, galaxies: stellar content, stars: evolution, Astrophysics},
         year = 2009,
        month = jul,
       volume = {699},
       number = {1},
        pages = {486-506},
          doi = {10.1088/0004-637X/699/1/486},
archivePrefix = {arXiv},
       eprint = {0809.4261},
 primaryClass = {astro-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2009ApJ...699..486C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{conroy10a,
       author = {{Conroy}, Charlie and {White}, Martin and {Gunn}, James E.},
        title = "{The Propagation of Uncertainties in Stellar Population Synthesis Modeling. II. The Challenge of Comparing Galaxy Evolution Models to Observations}",
      journal = {\apj},
     keywords = {galaxies: evolution, galaxies: stellar content, Astrophysics - Cosmology and Extragalactic Astrophysics, Astrophysics - Galaxy Astrophysics},
         year = 2010,
        month = jan,
       volume = {708},
       number = {1},
        pages = {58-70},
          doi = {10.1088/0004-637X/708/1/58},
archivePrefix = {arXiv},
       eprint = {0904.0002},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2010ApJ...708...58C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{conroy10b,
       author = {{Conroy}, Charlie and {Gunn}, James E.},
        title = "{The Propagation of Uncertainties in Stellar Population Synthesis Modeling. III. Model Calibration, Comparison, and Evaluation}",
      journal = {\apj},
     keywords = {galaxies: evolution, galaxies: stellar content, stars: evolution, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2010,
        month = apr,
       volume = {712},
       number = {2},
        pages = {833-857},
          doi = {10.1088/0004-637X/712/2/833},
archivePrefix = {arXiv},
       eprint = {0911.3151},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2010ApJ...712..833C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{harris20,
       author = {{Harris}, Charles R. and {Millman}, K. Jarrod and {van der Walt}, St{\'e}fan J. and {Gommers}, Ralf and {Virtanen}, Pauli and {Cournapeau}, David and {Wieser}, Eric and {Taylor}, Julian and {Berg}, Sebastian and {Smith}, Nathaniel J. and {Kern}, Robert and {Picus}, Matti and {Hoyer}, Stephan and {van Kerkwijk}, Marten H. and {Brett}, Matthew and {Haldane}, Allan and {del R{\'\i}o}, Jaime Fern{\'a}ndez and {Wiebe}, Mark and {Peterson}, Pearu and {G{\'e}rard-Marchant}, Pierre and {Sheppard}, Kevin and {Reddy}, Tyler and {Weckesser}, Warren and {Abbasi}, Hameer and {Gohlke}, Christoph and {Oliphant}, Travis E.},
        title = "{Array programming with NumPy}",
      journal = {\nat},
     keywords = {Computer Science - Mathematical Software, Statistics - Computation},
         year = 2020,
        month = sep,
       volume = {585},
       number = {7825},
        pages = {357-362},
          doi = {10.1038/s41586-020-2649-2},
archivePrefix = {arXiv},
       eprint = {2006.10256},
 primaryClass = {cs.MS},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020Natur.585..357H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{johnson21a,
       author = {{Johnson}, Benjamin D. and {Leja}, Joel and {Conroy}, Charlie and {Speagle}, Joshua S.},
        title = "{Stellar Population Inference with Prospector}",
      journal = {\apjs},
     keywords = {Galaxy evolution, Spectral energy distribution, Astronomy data modeling, 594, 2129, 1859, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2021,
        month = jun,
       volume = {254},
       number = {2},
          eid = {22},
        pages = {22},
          doi = {10.3847/1538-4365/abef67},
archivePrefix = {arXiv},
       eprint = {2012.01426},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@MISC{johnson21b,
       author = {{Johnson}, B.~D. and {Foreman-Mackey}, Dan and {Sick}, Jonathan and {Leja}, Joel and {Byler}, Nell and {Walmsley}, Mike and {Tollerud}, Erik and {Leung}, Henry and {Scott}, Spencer},
        title = "{dfm/python-fsps: python-fsps}",
         year = 2021,
        month = may,
          eid = {10.5281/zenodo.4737461},
          doi = {10.5281/zenodo.4737461},
      version = {v0.4.1rc1},
 howpublished = {Zenodo},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021zndo...4737461J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@MISC{johnson21c,
       author = {{Johnson}, Benjamin D.},
        title = "{bd-j/sedpy: sedpy}",
         year = 2021,
        month = mar,
          eid = {10.5281/zenodo.4582723},
          doi = {10.5281/zenodo.4582723},
      version = {v0.2.0},
 howpublished = {Zenodo},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021zndo...4582723J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@INPROCEEDINGS{paszke19,
       author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
    booktitle = {Advances in Neural Information Processing Systems},
       editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
        pages = {8024--8035},
    publisher = {Curran Associates, Inc.},
        title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
       volume = {32},
         year = {2019},
archivePrefix = {arXiv},
       eprint = {1912.01703},
          url = {https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf}
}

@article{pedregosa11,
  author  = {Fabian Pedregosa and Ga{{\"e}}l Varoquaux and Alexandre Gramfort and Vincent Michel and Bertrand Thirion and Olivier Grisel and Mathieu Blondel and Peter Prettenhofer and Ron Weiss and Vincent Dubourg and Jake Vanderplas and Alexandre Passos and David Cournapeau and Matthieu Brucher and Matthieu Perrot and {{\'E}}douard Duchesnay},
  title   = {Scikit-learn: Machine Learning in Python},
  journal = {Journal of Machine Learning Research},
  year    = {2011},
  volume  = {12},
  number  = {85},
  pages   = {2825--2830},
  url     = {http://jmlr.org/papers/v12/pedregosa11a.html}
}
```
