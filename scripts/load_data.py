import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import networkx as nx
import numpy as np

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
DPT_GEOMETRY = "dpt_geometry"
CENTROID = "CENTROID"

FUELS_BEFORE_2009 = [SUPER_PLOMBE, SUPER_SANS_PLOMB, GAZOLE, FOD, FOL]
FUELS_AFTER_2009 = [SUPER_SANS_PLOMB_95, SUPER_SANS_PLOMB_95_E10, SUPER_SANS_PLOMB_98, SUPER_SANS_PLOMB, GAZOLE, FOD, FOL]


def get_metropole() :
    # Download and read the GeoDataFrame from the provided URL
    url = "https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb"
    geo = gpd.read_file(url)
    # Extract necessary columns
    codes = [_ for _ in set(geo.code) if len(_) < 3]
    metropole = geo[geo.code.isin(codes)]

    # Reproject to a projected CRS
    metropole = metropole.to_crs(epsg=3395)  # Change the EPSG code to an appropriate projected CRS
    #Remove all the rows where the department code is not a number
    metropole = metropole[metropole["code"].str.isnumeric()]
    #Remove all the rows where the department code is greater than 95
    metropole = metropole[metropole["code"].astype(int) <= 96]
    #Convert the code column to numeric values
    metropole["code"] = pd.to_numeric(metropole["code"])
    return metropole

def get_departement_centroids() :
    """See french_fuel script"""
    metropole= get_metropole()
    # Calculate centroids of department polygons
    metropole['centroid'] = metropole['geometry'].centroid
    metropole = metropole[["code", "centroid", "geometry"]].rename(columns={"code": DEPARTEMENT_CODE, "centroid": CENTROID, "geometry" : DPT_GEOMETRY})
    return metropole


class Fuel_data() :
    def __init__(self, data_path : str, normalize = True) -> None:
        self.data_path = data_path

        self.dataframe_fuel = gpd.read_file('../__data/Donnees-annuelles-de-consommation-de-produits-petroliers-par-departement-France-metropol.2022-09.csv', sep=';')

        # Data first processing
        #Remove all the rows where the department code is not a number
        self.dataframe_fuel = self.dataframe_fuel[self.dataframe_fuel[DEPARTEMENT_CODE].str.isnumeric()]
        #Remove all the rows where the department code is greater than 95
        self.dataframe_fuel = self.dataframe_fuel[self.dataframe_fuel[DEPARTEMENT_CODE].astype(int) <= 96]
        #Convert the columns department_code, year to numeric values
        self.dataframe_fuel[DEPARTEMENT_CODE] = pd.to_numeric(self.dataframe_fuel[DEPARTEMENT_CODE])
        self.dataframe_fuel[ANNEE] = pd.to_numeric(self.dataframe_fuel[ANNEE])

        # Department centroids and fusion
        dataframe_metropole = get_departement_centroids()
        self.dataframe_fuel = self.dataframe_fuel.merge(dataframe_metropole, left_on="DEPARTEMENT_CODE", right_on="DEPARTEMENT_CODE", how="inner")
        self.num_rows = len(self.dataframe_fuel[DEPARTEMENT_CODE])

        # Keys and fuel columns to numeric
        self.keys = self.dataframe_fuel.columns.tolist()
        self.identifying_keys = [DEPARTEMENT_CODE, DEPARTEMENT_LIBELLE, REGION_CODE, REGION_LIBELLE, ANNEE, CENTROID, DPT_GEOMETRY, 'geometry']
        self.fuel_keys = [key for key in self.keys if key not in self.identifying_keys]
        for key in self.fuel_keys :
            self.dataframe_fuel[key] = pd.to_numeric(self.dataframe_fuel[key])
            self.dataframe_fuel[key].fillna(value=0)

        # Normalisation
        # self.normalize_by_year_by_fuel()

        # Compute sizes information
        self.num_dpt = len(set(self.dataframe_fuel[DEPARTEMENT_CODE]))
        self.num_fuel = len(self.fuel_keys)
        self.num_years = len(set(self.dataframe_fuel[ANNEE]))

    def normalize_by_year_by_fuel(self) :
        grouped = self.dataframe_fuel.groupby([ANNEE], dropna=False)
        mean_values = grouped[self.fuel_keys].transform('mean')
        std_values = grouped[self.fuel_keys].transform('std')
        self.dataframe_fuel[self.fuel_keys] = (self.dataframe_fuel[self.fuel_keys] - mean_values) / std_values

    def truncate(self, keys_to_keep : list = None, dpt_to_keep : list = None) :
        """Truncate the dataframe to keep only the desired variables 
        (the identifying keys are automatically kept)
        """
        # Filter columns
        if keys_to_keep is not None :
            self.num_fuel = len(keys_to_keep)
            for key in keys_to_keep :
                self.dataframe_fuel[key] = pd.to_numeric(self.dataframe_fuel[key])
                self.dataframe_fuel[key].fillna(value=0)
            self.fuel_keys = [key for key in keys_to_keep if key not in self.identifying_keys] #deep copy
            keys_to_keep += self.identifying_keys
            self.dataframe_fuel = self.dataframe_fuel[keys_to_keep]

        # Filter lines
        if dpt_to_keep is not None :
            self.dataframe_fuel = self.dataframe_fuel[self.dataframe_fuel[DEPARTEMENT_CODE].isin(dpt_to_keep)]
            self.num_dpt = len(dpt_to_keep)
        
    def generate_graph(self) :
        """Generate a graph with a node for each department 
        and a 'pos' associated to the department centroid"""
        G = nx.Graph()
        for code in set(self.dataframe_fuel[DEPARTEMENT_CODE]) :
            G.add_node(code)
            # Get the associated 'pos' with a Point object
            pos_point = self.dataframe_fuel[self.dataframe_fuel[DEPARTEMENT_CODE] == code][CENTROID].unique()[0]
            G.nodes[code]['pos'] = [pos_point.x, pos_point.y]
        self.graph = G

    def samples_by_year(self, var_name) -> np.array :
        """Returns an array of size num_dpt x num years with the value of var_name variable\
            and a list of years """
        samples = self.dataframe_fuel.pivot_table(index='DEPARTEMENT_CODE', columns='ANNEE', values=var_name)
        years = samples.columns.tolist()
        return samples.values, years
    
    def samples_one_year(self, year) -> np.array :
        """Returns an array of size num_dpt x num_var with the value of a given_year"""
        filtered_df = self.dataframe_fuel[self.dataframe_fuel["ANNEE"] == year]
        filtered_df = filtered_df[self.fuel_keys]
        samples = filtered_df.values
        return samples, self.fuel_keys
    
def plot_graph_department(g : nx.Graph, node_values : np.array = None, title = "", plot_labels = False) :
    metropole = get_metropole()
    metropole = metropole[metropole["code"].isin(g.nodes)]

    # Create figure and axis for the graph
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    metropole.plot(ax=ax, color='white', edgecolor='grey')
    pos = nx.get_node_attributes(g, 'pos')
    node_size = 5000 / 96
    if node_values is not None :
        # Define a colormap and normalize the node values to it
        cmap = cm.coolwarm
        vmin = np.min(node_values)
        vmax = np.max(node_values)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        nodes = nx.draw_networkx_nodes(g, pos, node_color=node_values, cmap=cmap, node_size=node_size, vmin=vmin, vmax=vmax)
        nodes.set_norm(norm)
    else :
        nodes = nx.draw_networkx_nodes(g, pos, node_size=node_size)
    edges_width = [g.edges[u, v]['weight'] for u, v in g.edges()]
    nx.draw_networkx_edges(g, pos, width=edges_width)

    if plot_labels:
        labels = {node: f"{node}\n\n{metropole[metropole['code'] == node]['nom'].values[0]}" for node in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels=labels, font_color='grey')

    # Create colorbar axis
    if node_values is not None :
        cbar_ax = fig.add_axes([0.95, 0.2, 0.05, 0.6])  # Adjust position and size as needed
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Signal amplitude')
    ax.set_title(title)
    plt.show()




