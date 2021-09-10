import pandas as pd
from ggdata.scripts import download


def download_from_WB():

    to_download = [('SH.DYN.MORT', 'health_mortality'),
                   ('SH.STA.MALN.ZS', 'nourishment_underweight'),
                   ('SH.IMM.IDPT', 'health_immunization'),
                   ('SH.STA.STNT.ZS', 'nourishment_stunting'),
                   ('SL.TLF.0714.ZS', 'labor_employment_total'),
                   ('SL.TLF.0714.SW.TM', 'labor_hour_total'),
                   ('SE.LPV.PRIM', 'education_learning_poverty'),
                   ('SE.LPV.PRIM.OOS', 'education_school_out'),
                   ('SE.LPV.PRIM.BMP', 'education_literacy')
                  ]
    
    dfs = []

    for wb_indic, ggi_code in to_download:
        config = {'GGI_code': ggi_code, 'params': {'indicator': wb_indic}}
        
        df = download(API_name='WB', config=config, raw=False)
        
        dfs.append(df)

    df = pd.concat(dfs)

    return df
    

def process(df):

    df = df[~df.ISO.isin([''])] # Remove empty ISOs

    # Add units
    df.loc[df.Variable == 'health_mortality', 'Unit'] = 'under 5 rate per 1,000 live birth'
    df.loc[df.Variable == 'nourishment_underweight', 'Unit'] = '% of children under 5'
    df.loc[df.Variable == 'health_immunization', 'Unit'] = '% of children ages 15-23 months'
    df.loc[df.Variable == 'nourishment_stunting', 'Unit'] = '% of children under 5'
    df.loc[df.Variable == 'labor_employment_total', 'Unit'] = '% of children ages 7-14'
    df.loc[df.Variable == 'labor_hour_total', 'Unit'] = 'hours per week'
    df.loc[df.Variable == 'education_learning_poverty', 'Unit'] = '%'
    df.loc[df.Variable == 'education_school_out', 'Unit'] = '%'
    df.loc[df.Variable == 'education_literacy', 'Unit'] = '%'
    
    
    df = df.drop(columns = ['URL', 'DownloadDate']).astype({'Year':int})

    #Modify Values for mortality and immunization
    df.loc[df.Variable == 'health_mortality', 'Value'] = df.loc[df.Variable == 'health_mortality', 'Value']/10
    df.loc[df.Variable == 'health_immunization', 'Value'] = 100 - df.loc[df.Variable == 'health_immunization', 'Value']
    
    #Interpolate
    df = (
        df.pivot(index = ['Country', 'ISO', 'Year'], columns = 'Variable', values = 'Value')
          .groupby(['ISO']).apply(lambda x : x.interpolate(limit_direction = 'both', axis = 0))
            .drop(columns = "education_literacy")
            .dropna()
    )
    
    return df


def main():
    df = download_from_WB()
    df = process(df)
    df.to_csv('data/data.csv')


if __name__ == '__main__':
    main()
