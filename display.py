import streamlit as st
import numpy as np
import pandas as pd
import json
from fbprophet import Prophet
import seasons
from fbprophet.plot import plot_plotly, plot_components_plotly


@st.cache
def import_data():
    return pd.read_csv('apy.csv')

def display_df(df):
    st.dataframe(df)

df = import_data()
# display_df(df)

st.title('Crop Production Prediction')

def options(df, col):
    return df[col].value_counts().index
st.sidebar.title("Welcome")
state = st.sidebar.selectbox("select the state", ['--'] + list(options(df, 'State_Name')))

if state != None:
    dfstate = df[df['State_Name']==state]
    district = st.sidebar.selectbox("Select the district",  ['--']+ list(options(dfstate, 'District_Name')))

if district != None:
    dfseason = dfstate[dfstate['District_Name']==district]
    season = st.sidebar.selectbox('Select season', ['--']+ list(options(dfseason, 'Season')))
    dic = {'MUMBAI':'bombay.csv', 'PUNE':'pune.csv', 'NAGPUR':'nagpur.csv', 
            'KANPUR NAGAR':'kanpur.csv', 'KANPUR DEHAT':'kanpur.csv', 'HYDERABAD':'hyderabad.csv', 
            "BANGALORE RURAL":'bengaluru.csv', 'BENGALURU URBAN':'bengaluru.csv', 
            'JAIPUR':'jaipur.csv'}
    
if season != None:
    dfcrop = dfseason[dfseason['Season']== season]
    crop = st.sidebar.selectbox('Select crop', ['--']+list(options(dfseason, 'Crop')))
if crop != None:
    st.text(f'state: {state} district:{district} season:{season} crop:{crop}')
    dfnew = pd.DataFrame(dfcrop[dfcrop['Crop'] == crop][['Crop_Year', 'Area', 'Production']])
    st.dataframe(dfnew)

season = season.strip()
seas = {'Kharif':'kharif', 'Rabi':'rabi', 'Whole Year':'whole'}  
district = district.strip()
ben = seasons.getData(pd.read_csv(dic[district]), seas[season])
ben['year'] = ben['year'].apply(int)
ben.set_index('year', inplace=True)
dfnew['Crop_Year'] = dfnew['Crop_Year'].apply(str)


def addTemp(year):
    if year in ['2009', '2010', '2011', '2012', '2013', '2014']:
        return ben.loc[[int(year)]]['tempC'].get(key=int(year))
    else:
        return np.nan

def addHum(year):
    if year in ['2009', '2010', '2011', '2012', '2013', '2014']:
        return ben.loc[[int(year)]]['humidity'].get(key=int(year))
    else:
        return np.nan

def addPrecip(year):
    if year in ['2009', '2010', '2011', '2012', '2013', '2014']:
        return ben.loc[[int(year)]]['precipMM'].get(key=int(year))
    else:
        return np.nan
sel = st.multiselect("Add regressors", ['Temp', 'humidity', 'Precip'])
dic = {'Temp':addTemp, 'humidity':addHum, 'Precip':addPrecip}
#Area is default regressor
for reg in sel:
    dfnew[reg] = dfnew['Crop_Year'].apply(dic[reg])



button1 = st.button('Add climate features')
if button1:
    st.dataframe(dfnew)

button = st.button('Predict')
if button:
    m = Prophet()
    st.write(f'Regressors: {sel}')
    m.add_regressor('Area')
    for reg in sel:
        m.add_regressor(reg)
    dfnew = dfnew[dfnew['Crop_Year'].isin(['2009', '2010', '2011', '2012', '2013','2014'])]
    dfnew.rename(columns={'Crop_Year':'ds', 'Production':'y'}, inplace=True)
    dftrain = dfnew[dfnew['ds'].isin(['2009', '2010', '2011', '2012', '2013'])]
    m.fit(dftrain)
    from fbprophet.plot import plot_yearly
    # st.plot(plot_yearly(m))
    future = m.make_future_dataframe(periods=1, freq='Y')
    future.iloc[[5]] = '2014-01-01 00:00:00'
    future['Area'] = list(dfnew['Area'])
    for reg in sel:
        future[reg] = list(dfnew[reg])
    forecast = m.predict(future)
    st.write(dfnew)
    fig = plot_plotly(m, forecast)
    fig.update_layout(
    title="Crop Yield Predicition",
    xaxis_title="Year",
    yaxis_title="Production",
    legend_title="labels"
    )
    fig.update_xaxes(
    rangeslider_visible=False)
    
    st.write(fig)
    st.write(f'MAE for the prediction is', abs(forecast.iloc[5]['yhat'] - dfnew.iloc[5]['y']))
    if forecast.iloc[5]['yhat'] > dfnew.iloc[4]['y']:
        if dfnew.iloc[5]['y'] > dfnew.iloc[4]['y']:
            dirag = 'positive'
        else:
            dirag = 'negative'
    else:
        if dfnew.iloc[5]['y'] > dfnew.iloc[4]['y']:
            dirag = 'negative'
        else:
            dirag = 'positive'
    st.write(f'Direction aggrement is ', dirag)










