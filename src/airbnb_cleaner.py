import pandas as pd
import numpy as np

# Separadores
def div(sep = 1, verbose=False):
    """
    Adiciona separadores para isolar resultados.

    Parametros
    ----------
    sep : 1, 2, 3
        1 : Separador de traço simples longo.
        2 : Separador de traço simples curto.
        3 : Separador de traço duplo longo.
    verbose : bool, opcional.
        Se True, imprime separadores.
    """
    logs = []
    if verbose:
        if sep == 1:
            print("-"*130)
        if sep == 2:
            print("="*100)
        if sep == 3:
            print("-"*70)

def format_name_col(df, verbose=False):
    """
    Padroniza os nomes das colunas do Dataset

    Parâmetros
    ----------
    df: pandas.DataFrame
        DataFrame original.
    verbose: bool, opcional
        Se True, imprime mensagens informando que os nomes das colunas foram padronizados.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com as colunas renomeadas.
    logs: list
        lista contendo verbose atual.

    """
    logs = []
    df.columns = df.columns.str.replace("_", " ").str.strip().str.title()
    if verbose:
        logs.append("Nomes das colunas padronizados.")
    return df, logs

def drop_unused_cols(df, verbose=False):
    # Remover colunas não utilizadas
    """
    Remove colunas não utilizadas.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando quais colunas foram removidas.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame sem as colunas desnecessárias.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    if ("Name" in df.columns):
        df.drop(columns=["Name"], inplace=True)
        if verbose:
            logs.append("Variável 'Name' removida!")
    if ("Host Name" in df.columns):
        df.drop(columns=["Host Name"], inplace=True)
        if verbose:
            logs.append("Variável 'Host Name' removida!")
    if ("Country" in df.columns):
        df.drop(columns=["Country"], inplace=True)
        if verbose:
            logs.append("Variável 'Country' removida!")
    if ("Country Code" in df.columns):
        df.drop(columns=["Country Code"], inplace=True)
        if verbose:
            logs.append("Variável 'Country Code' removida!")
    if ("Last Review" in df.columns):
        df.drop(columns=["Last Review"], inplace=True) 
        if verbose:
            logs.append("Variável 'Last Review' removida!")
    if ("House Rules" in df.columns):
        df.drop(columns=["House Rules"], inplace=True)
        if verbose:
            logs.append("Variável 'House Rules' removida!")
    if ("License" in df.columns):
        df.drop(columns=["License"], inplace=True) 
        if verbose:
            logs.append("Variável 'License' removida!")
    return df, logs

def host_id_verify_clean(df, verbose=False):
    # Tratamento da variável 'Host Identity Verified'
    """
    Faz o tratamento da coluna Host Identity Verified.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df.fillna({"Host Identity Verified": "unconfirmed"}, inplace=True)
    df["Host Identity Verified"] = df["Host Identity Verified"].str.strip().str.capitalize()
    if verbose:
        logs.append("Valores NaN da variável 'Host Identity Verified' substituídos por 'unconfirmed'!")
    return df, logs

def neigh_group_clean(df, verbose=False):
    # Tratamento da variável 'Neighbourhood Group'
    """
    Faz o tratamento da coluna Neighbourhood Group.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df["Neighbourhood Group"] = df["Neighbourhood Group"].str.strip().str.title().replace("Brookln", "Brooklyn").replace("Manhatan", "Manhattan")
    ngnan = df.loc[df["Neighbourhood Group"].isna(), "Neighbourhood"]
    ngnan_n = df["Neighbourhood"].isin(ngnan)
    ng_filter = df.loc[df["Neighbourhood Group"].notna(), ["Neighbourhood", "Neighbourhood Group"]].drop_duplicates(subset="Neighbourhood").set_index("Neighbourhood")["Neighbourhood Group"]
    df.loc[ngnan_n, "Neighbourhood Group"] = df.loc[ngnan_n, "Neighbourhood"].map(ng_filter)
    if verbose:
        logs.append("Valores NaN da variável 'Neighbourhood Group' substituídos!")
    return df, logs

def neigh_clean(df, verbose=False):
    # Tratamento da variável 'Neighbourhood'
    """
    Faz o tratamento da coluna Neighbourhood.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df.dropna(subset="Neighbourhood", inplace=True)
    if verbose:
        logs.append("Valores NaN da variável 'Neighbourhood' removidos!")
    return df, logs

def lat_long_clean(df, verbose=False):
    # Tratamento da variável 'Lat e Long'
    """
    Faz o tratamento das colunas Lat e Long.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como as colunas foram tratadas.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com as colunas tratadas.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df.dropna(subset=["Lat", "Long"], inplace=True)
    if verbose:
        logs.append("Valores NaN da variável 'Lat' e 'Long' removidos!")
    return df, logs

def inst_bookable_clean(df, verbose=False):
    # Tratamento da variável 'Instant Bookable'
    """
    Faz o tratamento da coluna Instant Bookable.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df["Instant Bookable"] = df["Instant Bookable"].astype(bool)
    df.fillna({"Instant Bookable": False}, inplace=True)
    if verbose:
        logs.append("Valores NaN da variável 'Instant Bookable' substituídos por False!")
    return df, logs

def cancel_policy_clean(df, verbose=False):
    # Tratamento da variável 'Cancellation Policy'
    """
    Faz o tratamento da coluna Cancellation Policy.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    cancel_mode = df["Cancellation Policy"].mode()[0]
    df.fillna({"Cancellation Policy": cancel_mode}, inplace=True)
    df["Cancellation Policy"] = df["Cancellation Policy"].str.strip().str.capitalize()
    if verbose:
        logs.append(f"Valores NaN da variável 'Cancellation Policy' substituídos por '{str(cancel_mode)}'!")
    return df, logs

def const_year_clean(df, verbose=False):
    # Tratamento da variável 'Construction Year'
    """
    Faz o tratamento da coluna Construction Year.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    median_year = int(df["Construction Year"].median())
    df.fillna({"Construction Year": median_year}, inplace=True)
    df["Construction Year"] = df["Construction Year"].astype(int)
    if verbose:
        logs.append(f"Valores NaN da variável 'Construction Year' substituídos por '{str(median_year)}'!")
    return df, logs

def price_clean(df, verbose=False):
    # Tratamento da variável 'Price'
    """
    Faz o tratamento da coluna Price.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    if ("Price" in df.columns):
        df["Price"] = df["Price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        df.rename(columns={"Price": "Price In $"}, inplace=True)
        df.dropna(subset="Price In $", inplace=True)
        df["Price In $"] = df["Price In $"].astype(int)
        if verbose:
            logs.append("Valores NaN da variável 'Price' removidos!\nVariável 'Price' renomeada para 'Price In $'!\nValor '$' removido!\nValor ',' renomeada para '.'!")
    return df, logs

def service_fee_clean(df, verbose=False):
    # Tratamento da variável 'Service Fee'
    """
    Faz o tratamento da coluna Service Fee.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    if ("Service Fee" in df.columns):
        df["Service Fee"] = df["Service Fee"].str.replace("$", "", regex=False)
        df.rename(columns={"Service Fee": "Service Fee In $"}, inplace=True)
        df.dropna(subset="Service Fee In $", inplace=True)
        df["Service Fee In $"] = df["Service Fee In $"].astype(int)
        if verbose:
            logs.append("Valores NaN da variável 'Service Fee' removidos!\nVariável 'Service Fee' renomeada para 'Service Fee In $'!")
    return df, logs

def min_nights_clean(df, verbose=False):
    # Tratamento da variável 'Minimum Nights'
    """
    Faz o tratamento da coluna Minimum Nights.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df.loc[(df["Minimum Nights"] <= 0) | (df["Minimum Nights"].isna()), "Minimum Nights"] = 1
    df["Minimum Nights"] = df["Minimum Nights"].astype(int)
    if verbose:
        logs.append("Valores NaN e valores menores que 0 da variável 'Minimum Nights' substituídos por 1!")
    return df, logs

def num_reviews_clean(df, verbose=False):
    # Tratamento da variável 'Number Of Reviews'
    """
    Faz o tratamento da coluna Number Of Reviews.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df.fillna({"Number Of Reviews": 0}, inplace=True)
    df["Number Of Reviews"] = df["Number Of Reviews"].astype(int)
    if verbose:
        logs.append("Valores NaN da variável 'Number of Reviews' substituídos por 0!")
    return df, logs

def reviews_per_month_clean(df, verbose=False):
    # Tratamento da variável 'Reviews Per Month'
    """
    Faz o tratamento da coluna Reviews Per Month.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df.fillna({"Reviews Per Month": 0}, inplace=True)
    df.loc[df['Number Of Reviews'] == 0, ["Reviews Per Month"]] = 0
    if verbose:
        logs.append("Valores NaN e de 'Number Of Reviews' igual a 0 da variável 'Reviews Per Month' substituídos por 0!")
    return df, logs
 
def review_rate_clean(df, verbose=False):
    # Tratamento da variável 'Review Rate Number'
    """
    Faz o tratamento da coluna Review Rate Number.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df.dropna(subset="Review Rate Number", inplace=True)
    df.loc[df['Number Of Reviews'] == 0, ["Review Rate Number"]] = 0
    if verbose:
        logs.append("Valores NaN e de 'Number Of Reviews' igual a 0 da variável 'Review Rate Number' substituídos por 0!")
    return df, logs

def calc_host_list_clean(df, verbose=False):
    # Tratamento da variável 'Calculated Host Listings Count'
    """
    Faz o tratamento da coluna Calculated host Listings Count.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df.fillna({"Calculated Host Listings Count": 1}, inplace=True)
    df["Calculated Host Listings Count"] = df["Calculated Host Listings Count"].astype(int)
    if verbose:
        logs.append("Valores NaN da variável 'Calculated Host Listings Count' substituídos por 1!")
    return df, logs

def availability_clean(df, verbose=False):
    # Tratamento da variável 'Availability 365'
    """
    Faz o tratamento da coluna Availability 365.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando como a coluna foi tratada.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com a coluna tratada.
    logs: list
        lista contendo verbose atual.
    """
    logs = []
    df.loc[df["Availability 365"] < 0, "Availability 365"] = 0
    df.fillna({"Availability 365": 0}, inplace=True)
    df.loc[df["Availability 365"] > 365, "Availability 365"] = 365
    df["Availability 365"] = df["Availability 365"].astype(int)
    if verbose:
        logs.append("Valores NaN e valores menores que 0 da variável 'Availability 365' substituídos por 0!\nValores maiores que 365 da variável 'Availability 365' substituídos por 365!")
    return df, logs

def df_cleaner(df, verbose):
    """
    Realiza o tratamento completo do DataFrame.

    Parametros
    ----------
    df : pandas.DataFrame
        DataFrame original
    verbose : bool, opcional.
        Se True, imprime mensagens informando o tratamento de todas as colunas.

    Retorno
    -------
    df: pandas.DataFrame
        DataFrame com as colunas tratadas e status do tratamento concluído.
    all_logs: list
        lista contendo verbose de todas ações executadas.
    """
    all_logs = []
    df = df.copy()
    step_clean = [format_name_col, drop_unused_cols, host_id_verify_clean, neigh_group_clean, neigh_clean, lat_long_clean, inst_bookable_clean,cancel_policy_clean, const_year_clean, price_clean, service_fee_clean, min_nights_clean, num_reviews_clean, reviews_per_month_clean, review_rate_clean, calc_host_list_clean, availability_clean]
    for step in step_clean:
        df, logs = step(df, verbose)
        all_logs.extend(logs)
    return df, all_logs