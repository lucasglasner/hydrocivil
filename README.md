<img src="image/logo.png" width=500 align='right'>

## hydrocivil: a package for hydrological methods in civil and enviromental engineering

Typical tasks related to water resources and engineering require fast calculations of hydrological properties and phenomena such as: flood hydrographs, flood routing along channels and reservoirs, evapotranspiration, infiltration, among other processes. With this purpose in mind, hydrocivil is presented as an alternative package to perform calculations that are usually done in tedious spreadsheets in a flexible and adjustable way to the user's needs. The purpose is to give tools to the engineer to calculate hydrologic processes/parameters with the methods and techniques he deems convenient, such as different varieties of unit hydrographs, synthetic storms or basin geomorphometric parameters. The package is not intended to be a replacement for larger hydrological models (e.g. HEC-HMS), but rather a fast, customizable and automatic alternative for simple multi-basin calculations.

The package is largely oriented to Chilean standards, however many methods and formulas originally come from the USA NCRS National Engineering Handbook. By default, most of the routines use formulas widely used in the hydrological community and only optionally you can choose methods and alternatives from national (Chilean) standards. The package is 100% written in English in order to maintain consistency with the syntax and basic classes/functions of the Python language.

## References

```bib
@article{NCRS_NEH630,
  title={National Engineering Handbook Part 630 - Hydrology},
  author={Natural Resources Conservation Service, United States Department of Agriculture (USDA)},
  year={}
}

@article{mcarreteras,
  title={Manual de Carreteras},
  author={Dirección de vialidad, Ministerio de Obras Públicas (MOP), Chile},
  year={2022}
}

@article{DGA_modificacioncauces,
  title={Guías metodológicas para presentación y revisión técnica de proyectos de modificación de cauces naturales y artificiales.},
  author={Dirección General de Aguas (DGA), Ministerio de Obras Públicas (MOP), Chile},
  year={2016}
}

@article{DGA_manualcrecidas,
  title={Manual de cálculo de crecidas y caudales mínimos en cuencas sin información fluviométrica},
  author={Dirección general de Aguas (DGA), Ministerio de Obras Públicas (MOP), Chile},
  year={1995},
}

```
