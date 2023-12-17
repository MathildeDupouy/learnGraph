import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

DEPARTEMENT_CODE = 'DEPARTEMENT_CODE'
DEPARTEMENT_LIBELLE ='DEPARTEMENT_LIBELLE'
REGION_CODE = 'REGION_CODE'
REGION_LIBELLE = 'REGION_LIBELLE'
ANNEE = 'ANNEE'
SUPER_PLOMBE = 'SUPER_PLOMBE'
SUPER_SANS_PLOMB = 'SUPER_SANS_PLOMB'
GAZOLE = 'GAZOLE'
FOD = 'FOD'
FOL = 'FOL'
GNR = 'GNR'
GPL = 'GPL'
CARBUREACTEUR = 'CARBUREACTEUR'
SUPER_SANS_PLOMB_95 = 'SUPER_SANS_PLOMB_95'
SUPER_SANS_PLOMB_95_E10 = 'SUPER_SANS_PLOMB_95_E10'
SUPER_SANS_PLOMB_98 = 'SUPER_SANS_PLOMB_98'
SUPER_ETH_E85 = 'SUPER_ETH_E85'
geometry = 'geometry'

CENTROID = "CENTROID"

def get_departement_centroids() :
    """See french_fuel script"""
    # Download and read the GeoDataFrame from the provided URL
    url = "https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb"
    geo = gpd.read_file(url)
    print(geo.head())
    # Extract necessary columns
    codes = [_ for _ in set(geo.code) if len(_) < 3]
    metropole = geo[geo.code.isin(codes)]

    # Reproject to a projected CRS
    metropole = metropole.to_crs(epsg=3395)  # Change the EPSG code to an appropriate projected CRS

    # Calculate centroids of department polygons
    metropole['centroid'] = metropole['geometry'].centroid
    #Remove all the rows where the department code is not a number
    metropole = metropole[metropole["code"].str.isnumeric()]
    #Remove all the rows where the department code is greater than 95
    metropole = metropole[metropole["code"].astype(int) <= 96]
    #Convert the code column to numeric values
    metropole["code"] = pd.to_numeric(metropole["code"])
    return metropole[["code", "centroid"]].rename(columns={"code": DEPARTEMENT_CODE, "centroid": CENTROID})


class Fuel_data() :
    def __init__(self, data_path : str) -> None:
        self.data_path = data_path

        self.dataframe_fuel = gpd.read_file('../__data/Donnees-annuelles-de-consommation-de-produits-petroliers-par-departement-France-metropol.2022-09.csv', sep=';')
        self.keys = self.dataframe_fuel.columns.tolist()

        # Data first processing
        #Remove all the rows where the department code is not a number
        self.dataframe_fuel = self.dataframe_fuel[self.dataframe_fuel[DEPARTEMENT_CODE].str.isnumeric()]
        #Remove all the rows where the department code is greater than 95
        self.dataframe_fuel = self.dataframe_fuel[self.dataframe_fuel[DEPARTEMENT_CODE].astype(int) <= 96]
        #Convert the columns department_code, year and fuel consumption to numeric values
        self.dataframe_fuel[DEPARTEMENT_CODE] = pd.to_numeric(self.dataframe_fuel[DEPARTEMENT_CODE])
        self.dataframe_fuel[ANNEE] = pd.to_numeric(self.dataframe_fuel[ANNEE])
        self.dataframe_fuel[GAZOLE] = pd.to_numeric(self.dataframe_fuel[GAZOLE])

        # Department centroids and fusion
        dataframe_metropole = get_departement_centroids()
        self.dataframe_fuel = self.dataframe_fuel.merge(dataframe_metropole, left_on="DEPARTEMENT_CODE", right_on="DEPARTEMENT_CODE", how="inner")

    def truncate(self, keys_to_keep = list) :
        self.dataframe_fuel = self.dataframe_fuel[keys_to_keep]
    
    



