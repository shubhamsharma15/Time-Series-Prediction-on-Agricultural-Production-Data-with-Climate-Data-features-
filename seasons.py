import pandas as pd


def getYear(st):
  return st[:4]

def getSeason(st):
  mn = int(st[5:7])
  if mn in range(7,11):
    return 'kharif'
  else:
    return 'rabi'


def getData(df, season='whole'):
    assert season in ['rabi', 'kharif', 'whole'], 'Not a season';
    df['year'] = df['date_time'].apply(getYear)
    df['season'] = df['date_time'].apply(getSeason)
    df = df[['year','season', 'tempC', 'humidity', 'precipMM']]
    dfnew = df[['year','season', 'humidity', 'tempC']].groupby(['year', 'season']).mean()
    dfnew['precipMM'] = df[['year','season', 'precipMM']].groupby(['year', 'season']).sum()['precipMM']
    dfnew.reset_index(inplace=True)
    if season == 'whole':
        dfnew = dfnew[['year','humidity', 'tempC']].groupby(['year']).mean()
        dfnew['precipMM'] = df[['year', 'precipMM']].groupby(['year']).sum()['precipMM']
        return dfnew
    else:
        return dfnew[dfnew['season'] == season]
    

if __name__ == '__main__':
    df = pd.read_csv('bengaluru.csv')
    print(getData(df, 'kharif'))