# Eg9PP_Ganoderma

This repository contains the source R code to implement survival models presented in the paper "Identification of Ganoderma disease resistance loci using natural field infection of an oil palm multi-parent population"
by [Sébastien Tisné] (https://www.researchgate.net/profile/Sebastien_Tisne) ([CIRAD](https://en.wikipedia.org/wiki/Centre_de_coop%C3%A9ration_internationale_en_recherche_agronomique_pour_le_d%C3%A9veloppement)), Virginie Pomiès, Virginie Riou, Indra Syahputra, Benoît Cochard and Marie Denis for predicting spatial effects and mapping resistance QTL.
R code was created by [Marie Denis](https://www.researchgate.net/profile/Marie_Denis2) (CIRAD).

The project hence is currently funded, and the copyright owned, by [PalmElit] (http://www.palmelit.com/en/) and CIRAD.
Document is under the [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/) license, and all R code are under the [GNU AGPL v3](https://www.gnu.org/licenses/agpl.html).

R code:
  1. call.R -- script to run R functions for predicting spatial effects and mapping resistance QTL on a real dataset
  2. UtilFunctions.R -- functions called in Spatial and MappingQTL functions
  3. Spatial.R -- function to predict spatial effects
  4. MappingQTL.R -- function to map resistance QTL
  
Data:
  1. KIN_Eg9PP_10.Rdata -- list of IBD matrices for the 10 first positions
  2. Eg9PP_Phenotypes -- matrix with survival times, censoring indicators, and additional covariates for all palm trees
  3. Eg9PP_Phenotypes_Mapping -- matrix with survival times, censoring indicators, and additional covariates for fully genotyped palm trees
  4. Eg9PP_Pedigree: matrix with palm, father, and mother identifiers


