# SpatialClustering

This repository contains functions for spatial clustering. In particular, it is divided into two modules:
1. **spatial**: for spatial data
2. **spatiotemporal**: for spatio-temporal data

## Description
The final goal of this project is creating a Python library for spatial clustering.

## Content

The included classes perform following clustering algorithms:
- `ClustGeo`: [[1]](#1), [[2]](#2)
- `BootstrapClustGeo`: [[3]](#3)
- `CorClustST`: [[4]](#4)

The classes are constructed to be consistent with `scikit-learn`.

## References

<a id="1">[1]</a> Marie Chavent, Vanessa Kuentz, Amaury Labenne and Jerome Saracco (2017). ClustGeo: Hierarchical Clustering with Spatial Constraints. R package version 2.0. https://CRAN.R-project.org/package=ClustGeo

<a id="2">[2]</a> Chavent, M., Kuentz-Simonet, V., Labenne, A., & Saracco, J. (2018). ClustGeo: an R package for hierarchical clustering with spatial constraints. Computational Statistics, 33(4), 1799-1822.

<a id="3">[3]</a> Distefano, V., Mameli, V., & Poli, I. (2020). Identifying spatial patterns with the Bootstrap ClustGeo technique. Spatial Statistics, 38, 100441.

<a id="4">[4]</a> Hüsch, M., Schyska, B. U., & von Bremen, L. (2018). CorClustST—Correlation-based clustering of big spatio-temporal datasets. Future Generation Computer Systems.
