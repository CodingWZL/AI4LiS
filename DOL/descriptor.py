from pymatgen.core import Structure
from pymatgen.core import Element
import numpy as np
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import Slab, ReconstructionGenerator

# VN: number of valence electrons
# EN: electronegativity_allen
# DP: dipole polarizability
# AR: atomic radius
# AN: atomic number
#feat_Ni = np.array([10, 11.13, 49.0, 13.5, 28])
feat_Ni = np.array([11.13])
#feat_S = np.array([6, 15.31, 19.4, 10.0, 16])
feat_S = np.array([15.31])
#feat_O = np.array([6, 21.36, 5.3, 6.0, 8])
feat_O = np.array([21.26])

# component feature of total structure
def component_feature(frac_Ni, frac_S):
    feature = frac_Ni * feat_Ni
    feature = np.hstack((feature, frac_S * feat_S))
    # print(feature)
    return feature

# structure feature of total structure
def structure_feature(ratio, flag):
    feature = np.array(())
    if flag == 0:
        for i in range(len(ratio)):
            if ratio[i] == 0:
                ratio[i] = 1
            feature = np.hstack((feature, ratio[i] * feat_S))
    else:
        for i in range(len(ratio)):
            if ratio[i] == 0:
                ratio[i] = 1
            feature = np.hstack((feature, ratio[i] * feat_Ni))
    # print(feature)
    return feature

# calculate the adsorption feature by taking feature Ni / feature O
def adsorption_feature(ratio_Ni):
    feature = np.array(())
    for i in range(len(ratio_Ni)):
        if ratio_Ni[i] == 0:
            ratio_Ni[i] = 1
        feature =np.hstack((feature, feat_O / (feat_Ni * ratio_Ni[i])))
    # print(feature)
    return feature


def calculate_feature(file_name):
    structure = Structure.from_file(file_name)
    frac_Ni = structure.composition.get_atomic_fraction(Element("Ni"))
    frac_S = structure.composition.get_atomic_fraction(Element("S"))
    #a = structure.composition.get_reduced_composition_and_factor(Element("Ni"))
    #print(a)

    # calculate the component feature from the fraction of elements in total structure
    feature_com = component_feature(frac_Ni, frac_S)

    cutoff = [3.0, 4.0, 5.0, 6.0]  # different cutoff
    # obtain the index of Ni in structure
    sites_Ni = structure.indices_from_symbol("Ni")
    sites_S = structure.indices_from_symbol("S")


    ratio_Ni = []
    ratio_S = []
    for r in cutoff: # for each cutoff
        count_Ni = 0
        count_S = 0
        for site in sites_Ni: # for each Ni atom, take Ni as center
            neighbors = structure.get_neighbors(structure.sites[site], r=r) # get the neighbors(sites) of Ni in a cutoff
            for neighbor in neighbors: # for each neighbor
                specie = neighbor.specie # obtain the specie of neighbor
                if specie.symbol == "Ni":
                    count_Ni += 1 # count the number of neighbors of Ni
                else:
                    count_S += 1

        ratio_Ni.append((count_Ni-count_S) / len(sites_Ni)) # abs
        #print(count_Ni / (count_Ni + count_S))
    #print(ratio_Ni)
    # calculate the structure feature for a cutoff (take Ni atom as center)
    feature_str_Ni = structure_feature(ratio_Ni, flag=1)



    ratio_S = []
    for r in cutoff:  # for each cutoff
        count_Ni = 0
        count_S = 0
        for site in sites_S:  # for each Ni atom, take S as center
            # the structure is in the top of crystal cell
            neighbors = structure.get_neighbors(structure.sites[site],
                                                r=r)  # get the neighbors(sites) of S in a cutoff
            for neighbor in neighbors:  # for each neighbor
                specie = neighbor.specie  # obtain the specie of neighbor
                if specie.symbol == "S":
                    count_S += 1  # count the number of neighbors of S
                else:
                    count_Ni += 1

        ratio_S.append((count_Ni-count_S) / len(sites_S)) # abs
    feature_str_S = structure_feature(ratio_S, flag=0)
    # calculate the adsorption feature by taking feature Ni / feature O
    feature_ads = adsorption_feature(ratio_Ni)

    # print(feature_com.shape, feature_str.shape, feature_ads.shape)
    # print(feature_com, feature_str, feature_ads)
    feature = np.concatenate((feature_com, feature_str_Ni, feature_str_S, feature_ads), axis=0)
    return feature

def main():
    feature = np.array(())
    data_path = "Your path"
    feature = calculate_feature(data_path+str(i)+".cif")
    #print(a.shape)
    feature = feature.reshape((-1, 14))
    #np.savetxt("DOL-feature.txt", feature)

if __name__ == '__main__':
    main()

