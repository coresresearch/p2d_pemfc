[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3468679.svg)](https://doi.org/10.5281/zenodo.3468679)


# p2d_pemfc
Two pseudo-2D Newman-type models of a PEM Fuel Cell.

## Objective(s)
This model simulates the cathode of a proton exchange membrane fuel cell (PEMFC). 
Although these types of fuel cells have made their way into production, an expensive 
Pt catalyst has inhibited their growth and manufacturing. Reductions to the amount 
of Pt have been heavily investigated; however, PEMFCs with low Pt loading have shown 
poor performance due to losses that have not been sufficiently explained in literature. 
The objectives of this model were: to incorporate structure-property relationships 
in an attempt to capture and explain performance losses at low Pt loading, to determine 
how the modeled catalyst layer (CL) geometry affects transport parameters, and to 
exercise the model in a search for microstructures that could provide improvements in 
PEMFC performance.

The main difference that sets this model apart from others involves the included 
structure-property relationships. These additions allow the transport of protons 
and oxygen through the thin Nafion electrolyte found within the CL to change based 
on temperature, relative humidity, thickness, and Pt loading. Direct measurements 
of properties such as conductivity and diffusion coefficients have been challenging 
with current techniques for the needed length scale of Nafion (<10nm). Many of the 
few experiments that have been done in a relevant length scale were performed on 
substrates that are not found within PEMFCs. This is important to note since studies 
have shown that in nanothin films, Nafion's structure and resulting properties can 
change due to substrate interactions. Using neutron reflectometry data (to approximate 
the water volume fractions and structure of nanothin Nafion films) combined with 
experiments and empirical formulas for bulk Nafion films, three methods were implemented 
into this model to estimate theoretical transport parameters within the electrolyte 
phase of the CL.

## Modeling Domains
In literature, two common microstructures have been used to represent the CL of 
PEMFCs: core-shell and flooded-agglomerate. Both methods represent attempts to 
simplify the complex geometries observed from transmission electron microscope 
(TEM) images of CLs within PEMFCs. The core-shell model represents a smaller 
length scale in which Pt-covered carbon cores are covered in a shell of Nafion 
electrolyte. Taking multiple core-shell structures and packing them together inside 
an additional Nafion shell represents the structure used for the flooded-agglomerate 
model. Illustrations of these two types of geometries as well as the types of 
transport incorporated into this model are shown in Figure 1.

<p align="center"> <img src="https://user-images.githubusercontent.com/39809042/60464579-84b6f900-9c3e-11e9-95c4-9c6c85ff2c11.PNG"> </p>
<p align="center"> Figure 1: Visualization of two microstructures available to use in CL model. </p>

The model written here allows the user to specify one of these two types of CL 
geometries by adjusting the "model" input found within "pemfc_runner.py". Regardless 
of the CL geometry, this model explicitly simulates the gas diffusion layer (GDL) 
of the cathode side of the PEMFC in addition to the CL. In order to produce 
polarization curves that represent the full cell, a linear resistance term is added 
to account for the membrane and the potentials are shifted by the equilibrium potential 
of the anode.

## Simulation Methods
This model uses a finite volume method in order to conserve mass within the system. 
Although performance at steady state conditions are the output from this model, an 
initial value problem ODE integrator is used along with transient differential 
equations until steady state is reached. Due to this solution method, it is important 
that the user set the "t_sim" variable within "pemfc_runner.py" to a sufficiently 
long time. In order to check that steady state is being reached, the "debug" input 
also within "pemfc_runner.py" can be set to the value of `1` in order to produce plots 
of the variables within the solution vector against time. If steady state is being 
reached, then each of these plots should reach a constant value by the end time set 
by the user.

## Installation Instructions
1. Install [Anaconda](https://www.anaconda.com/distribution/) - make sure to get 
Python 3 syntax.
2. Launch "Anaconda Prompt" once the installation has finished.
3. Type `conda create --name echem --channel cantera/label/dev cantera numpy scipy matplotlib` 
into the terminal of "Anaconda Prompt" to set up an environment named "echem" with the 
needed packages.
4. When prompted, type `y` and press enter to finish setting up the environment. 
Agree to any required pop-up messages.
5. Test the new environment by typing `activate echem` followed by the enter key.
6. Install an editor for Python files. A good option is [Atom](https://atom.io/).
6. Download all of the files from this repository onto your local machine.
7. Follow the operating instructions below to edit and run the model.

## Operating Instructions
1. Open "Anaconda Prompt" and type `activate echem` followed by the enter key.
2. Use `cd` to change into the directory where all of the repository files were 
downloaded to.
3. Once inside the correct directory, run the model by typing in `python pemfc_runner.py` 
and pressing enter.
4. To edit any of the model inputs or options, open the "pemfc_runner.py" file in any 
Python editor (e.g. Atom).
5. After making any desired changes to "pemfc_runner.py", save the file and repeat 
steps 1-3 to rerun the model.

Optional: If you would prefer to use a developer environment (sort of like Matlab) 
instead of the "Anaconda Prompt" terminal, then do the following: open "Anaconda Navigator", 
select "echem" from the dropdown menu labeled "Applications on" near the top of the page, 
and install "spyder" from the tiles on the screen. Once Spyder is installed, the 
"pemfc_runner.py" file can be opened within the program where it can be both edited and 
run without the need for a separate editor and terminal. For more details visit Spyder's 
website [here](https://www.spyder-ide.org/).

## License
This tool is released under the BSD-3 clause license, see LICENSE for details.


## Citing the Model
 This model is versioned using Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3468679.svg)](https://doi.org/10.5281/zenodo.3468679)


If you use this tool as part of a scholarly work, please cite using:

> C.R. Randall and S.C. DeCaluwe. (2019) P2D PEMFC Model v1.1 [software]. Zenodo. https://doi.org/10.5281/zenodo.3468679

A BibTeX entry for LaTeX users is

```TeX
@misc{P2D_PEMFC,
    author = {Corey R. Randall and Steven C. DeCaluwe},
    year = 2019,
    title = {P2D PEMFC Model v1.1},
    doi = {10.5281/zenodo.3468679},
    url = {https://github.com/coresresearch/p2d_pemfc},
}
```

In both cases, please update the entry with the version used. The DOI for the latest version is
given in the badge at the top, or alternately <https://doi.org/10.5281/zenodo.3468679> will
take you to the latest version (and generally represents all versions).
