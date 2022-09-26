# density2attenuation

This repository contains a U-Net structure to determine the 3D attenuation coefficients fo a 3D density cube.

It was trained and tested for turbulent box simulation on ISM scales, but could possibly be adabetd to other scales aswell.


## Motivation

The gas in between the stars constitutes what is called the interstellar medium (ISM). In its coldest regions (several Kelvin) molecules can form and dense filaments are observed which are the sites of star formation. To determine the dynamics of these filaments, their collapse and the formation of prestellar cores, hydrodynamic simulations are performed. For more realistic modeling and comparison to observations, it is necessary to implement a chemical network, tracing the formation and evolution of different molecules. As the chemistry is strongly dependent on external UV irradiation, the surrounding density structure is important to the evolution of the local cell. Accounting for this is computationally very expensive, as for every cell the surrounding structure must be evaluated. To circumvent this costly calculation, we introduce a machine learning approach to predict the attenuation factor for each cell. Once trained, this could lead to a significant speed up of the simulation, as the evaluation ofthe network is much faster than the actual calculation.

## Determination of the attenuation factor

![Ray tracing](https://github.com/ehoemann/density2attenuation/blob/main/images/rayTracing.png)

**Column density**: Summed up denisty along a ray until it reaches on optically thini pixel: $\sigma = \sum_{ray}\ \rho _{x,y}\ \text{d}x$

**Attenuation factor**: The instensity of incoming rays is reduced by $\chi = \text{exp}(-\bar{\sigma})$

The determination of the attenuation factor in simulations is usually done ba a raytracing algorithm which is strongly dependent on the resolution and the number of rays $O(N^4_{Bins}\times N_{rays})$

## U-Net

The 3D U-Net was an adoption of the 2D U-Net provided in: https://github.com/milesial/Pytorch-UNet

![U-Net](https://github.com/ehoemann/density2attenuation/blob/main/images/U-Net.png)

## Validation

### Proof of concept: Overfit with a small and simplified data set

- Standardization of the data was needed $d_{Norm} = (d-\bar{d})/\text{std}(d)$
- Simplification of one ray to caluclate the column density
- Mapping 80$^{3}$ denisty to 40$^{3}$ attenuation factor (calculated by ray tracing)

![Overfitting](https://github.com/ehoemann/density2attenuation/blob/main/images/Overfitting.png)

Overfitting worked, although perfomance varies for different inputs.

However, overal statistics is recovered well, see left plot, and large structures seem to be reproduced, wheras devitaions occur for small structures, see right plot.

![statcitics](https://github.com/ehoemann/density2attenuation/blob/main/images/statictics.png)

### Multi Ray sample

- 42 rays used to determine column density
- Use structure evaluated in proof of concept

![MultiRay](https://github.com/ehoemann/density2attenuation/blob/main/images/MultiRay.png)

## Conclusion

We show a proof of concept: The attenuation factor in simulations can be determined with a U-Network. Training with a bigger sample is necessary, but preliminary results look promising. The next step would be, to implement it in a hydrodynamic simulation to calculate attenuation factors during runtime.
