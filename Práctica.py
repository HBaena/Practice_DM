#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr as correlation 
from re import sub


# In[ ]:


df = pd.read_csv("survey_results_public.csv")
df.describe(include='all')
df = df.replace('NaN', np.nan)
df = df.replace('nan', np.nan)
df = df.replace('JavaScript', 'javascript')
def edit_value(item, replace, new_value):
    try:
        return item.replace(replace, new_value)
    except:
        return item
def remove_c(item):
    try: 
        item = sub('(;|^)C(;|$)', ';C Language;', item)
        return sub('^;|;$', '', item)
    except:
        return item
def remove_noise(item):
    try:
        return sub('\s*\(.*\)\s*', '', item).strip()
    except:
        return item
# Change JavaScript to javascript to not count into java (sub_str implementation) 
df['LanguageWorkedWith'] = df['LanguageWorkedWith'].apply(edit_value, args=('JavaScript', 'javascript'))
# Change C to C Language to not count into C++,C##, typescript, etc (sub_str implementation) 
df['LanguageWorkedWith'] = df['LanguageWorkedWith'].apply(remove_c)
# 
df['EdLevel'] = df['EdLevel'].apply(edit_value, 
                                    args=('Professional degree (JD, MD, etc.)', 
                                          'Bachelor’s degree (BA, BS, B.Eng., etc.)'))
df['EdLevel'] = df['EdLevel'].apply(remove_noise)


# In[ ]:


df.head()


# In[ ]:


def is_iterable(obj:object) -> bool:
    '''
        Determine if a object is iterable or not
        return
            
    '''
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True
def new_dict(keys:iter, values:iter) -> dict:
    '''
        Merge two iterables (keys and values) into a dict
        return 
            dictionary
    '''
    return {k:v for k, v in zip(keys, values)}
def sort_dict(d:dict, reverse:bool=False, 
              func:callable=lambda item: item[1]) -> dict:
    '''
        Sort a dict by a given fucntion or according to the value
        d       -> dict to sort
        reverse -> direction to sort
        func    -> sort criteria
        
        return 
            sorted dict
    '''
    return {k: v for k, v in sorted(d.items(), key=func, reverse=reverse)}        
def not_nulls(df:pd.DataFrame, *columns) -> pd.DataFrame:
    '''
        Remove rows with null values
        if no columns are passed so dropna is applied to the whole data frame
        if columns are passed
        are filtered with notnull() func and & operators
        return:
            Pandas DataFrame
    '''
    if not columns:
        return df[df.notnull()]
    filter_ = df[columns[0]].notnull()
    for column in columns[1:]:
        filter_ &= df[column].notnull() # Avoinding null values
    return df[filter_]
def describe_to(df:pd.DataFrame, column:str=None, by:str=None, 
                slice_:slice=slice(0, 8), as_numpy:bool=True) -> np.array:
    '''
        Convert decribe() DataFrame method to numpy array or pd.serie
        Functiion can take only a DataFrame and even 2 DataFrame's columns name
        df       -> Pandas DataFrame
        column   -> DataFrame's column name
        column and by   -> Describe() of column for each unique value of by
        slice    -> Slice to return. All values by defaul
        as_numpy -> numpy or pd.Series. numpy by defaul
        
        return sliced Decribe()
    '''
    if by and not column:
        return None
    if not column and not by:
        return df.describe()[slice_].to_numpy() if as_numpy else df.describe()[slice_]
    new_df = df[column]
    if not by:
        return new_df.describe()[slice_].to_numpy() if as_numpy else new_df.describe()[slice_]
    else:
        l = []
        new_df = not_nulls(df, column, by)
        uniques = uniques_values(new_df, by)
        for unique in uniques:
            filter_ = new_df[by].apply(contain_substr, args=(unique,)) # get certain value even on multimple values rows
            f_df = new_df[filter_][column] # filtered dataframe
            l.append((unique, f_df.describe()[slice_].to_numpy() if as_numpy else f_df.describe()[slice_]))
        return np.array(l)  
def five_number_summary(df:pd.DataFrame, column:str, by:str, 
                        as_numpy:bool=True) -> np.array:
    '''
        Five numbers summary
        df     -> Pandas DataFrame
        column          -> DataFrame's column name
        column and by   -> Five numbers summary of column for each unique value of by
        return:
            min, q1, q2 (median), q3, max or a list
    '''
    return describe_to(df, column, by, slice(3, 8), as_numpy)
def contain_substrs(str_:str, *sub_strs) -> bool:
    condition = True
    length = len(sub_strs)
    i = 0
    while condition and (i < length):
        condition = condition and contain_substr(str_, sub_strs[i])
#         print(sub_strs[i], condition)
        i += 1
    return condition
def contain_substr(str_:str, *sub_str) -> bool:
    '''
        Functions that find a substring on a string with certain characteristics
        This function will find "substr" inside str
        return 
            True or False
    '''
    for s in sub_str:
        if str_.find(s) != -1:
            return True
    return False
def uniques_values(df:pd.DataFrame, column:str=None) -> list:
    '''
        This functions take as considerations that columns values could contain multiples 
        values separated by semicols.
        
        df     -> Pandas DataFrame
        column -> columns to find uniques
        
        return list of uniques values
    '''
#     print(df.head(2))
    if column:
        uniques = df[column].unique().tolist()
    else:
        uniques = df.unique().tolist()
    uniques = ';'.join(str(g) for g in uniques)
    return list(set(list(uniques.split(';'))))
def hist_box_by(df:pd.DataFrame, column:str, by:str, alias_column:str=None, 
                alias_by:str=None, boxplottype:str='outliers',
                colors:list=None, nbins:int=10,
                title:str=None) -> make_subplots:
    '''
        This function excludes null values on "column" and "by" doing a simple & functions
        df      -> Pandas DataFrame
        column  -> Column to analyze
        by      -> Column respect to analyze "column"
        alias_column -> String to show in texts (legend, titles, axis)
        alias_by     -> String to show in texts (legend, titles, axis)
        boxplottype  -> The way of outliers are handle and showed. 
                        'all'        -> all points are considerated
                        False (bool) -> Only wishkers are considerated
                        'suspectedoutliers' -> Only suspected outliers are considerated
                        'outliers'   -> Only outliers are considerated (default value)
        colors       -> list of colors wich the plots will take. This colors should be representated
                        in format:
                        hexa   -> '#F33DF5'
                        rgb    -> 'rgb(200, 56, 13)'
                        rgba   -> 'rgba(20, 189, 12, 90)'
                        string -> 'green'
                        Thes list can contain mix format.
                        colors=['lightgreen', 'rgba(1,200,3,10)', '#FF0034']
                        colors=['green', 'lightgreen', 'darkgreen']
                        This list must be at least teh same lenght of unique values to plot
        nibins       -> Number of bins for the histograms
        return plotly figure or None
        
        exaple 1:
        hist_box_by(df, column='StudentsCalifications', by='Gender',
                    column_alias='Califications').show()
        example 2:
        fig = hist_box_by(df, 'dogs_weights', 'kind', boxplottype=False)
        fig.show()
        example 3:
        hist_box_by(df, 'salary', 'Gender', boxplottype='all',
                    colors=['#073632', '#7ED388', '#2F668C']).show()
    '''
    if not alias_column:
        alias_column = column
    if not alias_by:
        alias_by = by
    new_df = not_nulls(df, column, by)
    if new_df[by].dtype == 'object':
        uniques = uniques_values(new_df, by)
        func = contain_substr
    else:
        uniques = new_df[by].unique()
        uniques.sort()
        func = lambda item, arg: item == arg
    fig = make_subplots(
        rows=int((len(uniques)*2/4 + 1)), cols=4,
        subplot_titles=['Histogram', 'Boxplot'])
    if not colors:
        colors = [ "#5E3A6D", "#526A2B", "#855B96", "#938534", "#C696DA", "#F0B9F4", "#F2D2FF", "#F4E3B9", "#293375", "#175565", "#717275",
        "#4E98AC", "#EA928D", "#A8C6CE", "#F5CBC8", "#AAE4E6", "#4C7CAF", "#C6CD51", "#BA7847", "#E7E884", "#F27754", 
        "#92EBE1", "#F1CCCC", "#D5F1EE", "#3930B4", "#8B1B1B", "#75BCB9", "#BF5B97", "#B0AEA8", "#A07CC9", "#F1DBDB", "#9ED5F1"][::-1]    
    j = 1
    
    for i, unique_ in enumerate(uniques):
        filter_ = new_df[by].apply(func, args=(unique_,)) # get certain value even on multimple values rows

        filtered_df = new_df[filter_] # filtered dataframe
#         print(j, (i*2)%2 + 1 + (i*2)%4)
#         print(j, (i*2)%2 + 2 + (i*2)%4)
        fig.add_trace(
            go.Histogram(y=filtered_df[column], name='Histogram' + str(unique_), nbinsy=nbins,
                         marker_color=colors[i]),
                        row=j, col=(i*2)%2 + (i*2)%4 + 1) # add histogram plot
        fig.add_trace(
            go.Box(y=filtered_df[column], boxpoints='outliers', name=str(unique_), 
                        marker_color='#dedbd9',
                        line_color=colors[i]
                  ),
                        row=j, col=(i*2)%2 + (i*2)%4 + 2) # add boxplot
        # adding axis labels
        fig.update_xaxes(title_text='Count', row=j, col=(i*2)%2 + (i*2)%4 + 1)
        fig.update_xaxes(title_text='', row=j, col=(i*2)%2 + (i*2)%4 + 2)
        fig.update_yaxes(title_text=alias_column, row=j, col=(i*2)%2 + (i*2)%4 + 1)
        j += (0,1)[i%2]
    # configurating size, titles and legend
    if not title:
        title = alias_column + ' by ' + alias_by
    fig.update_layout(height=350*int((len(uniques)*2/4 + 1)), width=1400,
                      title_text=title,
                    showlegend=False)
    return fig
def hist_by(df:pd.DataFrame, column:str, by:str, alias_column:str=None, 
                alias_by:str=None, colors:list=None, 
                nbins:int=10, func:callable=np.sum) -> go.Figure:
    '''
        This function excludes null values on "column" and "by" doing a simple & functions
        df      -> Pandas DataFrame
        column  -> Column to analyze
        by      -> Column respect to analyze "column"
        alias_column -> String to show in texts (legend, titles, axis)
        alias_by     -> String to show in texts (legend, titles, axis)
        func         -> Criteria for the histogram. Frequency by defaul
        colors       -> list of colors wich the plots will take. This colors should be representated
                        in format:
                        hexa   -> '#F33DF5'
                        rgb    -> 'rgb(200, 56, 13)'
                        rgba   -> 'rgba(20, 189, 12, 90)'
                        string -> 'green'
                        Thes list can contain mix format.
                        colors=['lightgreen', 'rgba(1,200,3,10)', '#FF0034']
                        colors=['green', 'lightgreen', 'darkgreen']
                        This list must be at least teh same lenght of unique values to plot
        nibins       -> Number of bins for the histograms
        return plotly figure or None
    '''
    if not alias_column:
        alias_column = column
    if not alias_by:
        alias_by = by
    new_df = not_nulls(df, column, by)
    uniques = uniques_values(new_df, by)
    if not colors:
        colors = ['#7ED388', '#2F668C', '#ABB392', '#EA7251', '#8501BA', '#09993E', '#BB60C4', '#65F587', '#13CEB7', '#C17B62', '#73A12E', '#344620', '#9B355F', '#B02D36', '#22B407', '#B3391C', '#408C09', '#7846C3', '#548FE2', '#290AFC', '#7C01A8', '#0D8911', '#DEC5CB', '#741BE5', '#5E11F9', '#1C4D07', '#7827A1', '#BFA409', '#F6CB1C', '#DC7AE4', '#97F7ED', '#492CDE', '#CEE30D', '#FAD149', '#BE98C9', '#04D364', '#3D4FF6', '#BB326A', '#4F50A5', '#D02CE0', '#11C68F', '#1863DD', '#0BC224', '#4F2000', '#A38BBC', '#06C5B2']*5
    fig = make_subplots(
        rows=len(uniques),
        subplot_titles=list(uniques))
    for i, unique_ in enumerate(uniques):
        filter_ = new_df[by].apply(contain_substr, args=(unique_,)) # get certain value even on multimple values rows
        filtered_df = new_df[filter_] # filtered dataframe
        fig.add_trace(
            go.Histogram(x=filtered_df[column], name=unique_, 
                        marker_color=colors[i], nbinsx=nbins
                  ), row=i+1, col=1
        )
        fig.update_xaxes(title_text=alias_column, row=i + 1, col=1)
        fig.update_yaxes(title_text=alias_by, row=i + 1, col=1)

    # configurating size, titles and legend
    fig.update_layout(height=350*len(uniques), width=800,
                      title_text=alias_column + ' by ' + alias_by,
                    showlegend=False)
    return fig
def mean_std_median(df:pd.DataFrame, column:str=None, by:str=None, 
                    as_numpy:bool=True) -> np.array:
    '''
        mean, standar deviation and median
        df     -> Pandas DataFrame
        column   -> DataFrame's column name
        column and by   -> values of column for each unique value of by
        return 
            maean, std and meadin or a list
    '''
    return describe_to(df, column, by, [1,2,5], as_numpy)
def barplot(df:pd.DataFrame, column:str, alias_column:str=None, 
            alias_count:str=None, sort:bool=False, 
            min_:int=0,color:str=None) -> go.Figure:
    '''
        df     -> Pandas DataFrame
        column -> Column name
        alias_column -> Text sto show on the plot
        sort   -> If value will be showed sorted or not
        return 
            Plotly figure
    '''
    if not alias_column:
        alias_column = column
    if not alias_count:
        alias_count = 'Count'
    new_df = not_nulls(df, column)
    uniques = uniques_values(new_df, column)
    lengths = {u : new_df[column].apply(contain_substr, args=(u,)).sum() for u in uniques}
    if sort:
        lengths = sort_dict(lengths, reverse=True if sort=='reverse' else False)
    others = sum(lengths.values())
    print('Others', others)
    lengths = dict(filter(lambda item: item[1] > min_, lengths.items()))
    others = others - sum(lengths.values())
    if others:
        lengths['Others'] = others
    if not color or len(color) != 2:
        color = ('rgb(158,202,225)', 'rgb(8,48,107)')
    fig = go.Figure(data=[go.Bar(x=list(lengths.keys()), y=list(lengths.values()), 
                                text=list(lengths.values()), textposition='auto',
                                textfont=dict(size=18, color="black"))])
    fig.update_traces(marker_color=color[0], marker_line_color=color[1],
                  marker_line_width=1.5, opacity=0.6)
    fig.update_layout(
        height=1000,
        title='Barchar for ' + alias_column,
        xaxis_tickfont_size=14,
        yaxis=dict(
            title=alias_count,
            titlefont_size=16,
            tickfont_size=14,
        ),
        xaxis=dict(
            title=alias_column,
            titlefont_size=14,
            tickfont_size=12,
            tickangle=55
        ),
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
#     fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig
def str_to_float(value:object) -> bool:
    '''
        Filter for DataFrames that exclude not numeric values
        return 
            True if could e converted to numeric
            Fals if don't
    '''
    try:
        float(value)
        return True
    except:
        return False
def remove_not_numeric(df:pd.DataFrame, *columns) -> pd.DataFrame:
        
        new_df = not_nulls(df, *columns)
        if not columns:
            return new_df[new_df.apply(str_to_float)]
        for column in columns:
            new_df = new_df[new_df[column].apply(str_to_float)]            
        return new_df
def scatter_correlation(df:pd.DataFrame=None, x:str=None, y:str=None,
                        aliasx:str=None,  aliasy:str=None,
                        text:object=None, color:object='#7ED388') -> (float, go.Figure) :
    '''
        Pearson correlation calculation and scatter plot unsing x and y
        if a dataframe is given, clean not numeric values and null values
        df       -> Pandas DataFrame
        x and y  -> If a dataframe is given, x and y are columns' names of this
                    If not dataframe is given, x and y are arrays
        aliasy and alias x -> text for the plot
        
        return
            correlation value and a Plotly Figure with the scatter plot
    '''
    if df is not None:
        if not aliasx:
            aliasx = x
        if not aliasy:
            aliasy = y
#         new_df = not_nulls(df, x, y)
#         new_df = new_df[new_df[x].apply(str_to_float)]
#         new_df = new_df[new_df[y].apply(str_to_float)]
        new_df = remove_not_numeric(df, x,y)
        x, y = new_df[x].astype('float64'), new_df[y].astype('float64')
    if df is None:
        if not aliasx:
            aliasx = 'x'
        if not aliasy:
            aliasy = 'y'
    if text is None:
        text=''
    figure = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers', 
                                        text=text, marker_color=color)]
                      )
    figure.update_layout(
        title="Correlation of " + aliasx + ' and ' + aliasy,
        xaxis_title=aliasx,
        yaxis_title=aliasy,
    )
    corr, _ = correlation(x,y)
    return corr, figure
def ordinal_to_numeric(df:pd.DataFrame, ranks:list, column:str=None, 
                       as_numpy:bool=True) -> (np.array, dict): 
    if column:
        new_df = df[column]
    else:
        new_df = df
    new_df = not_nulls(new_df)
    for k, v in ranks.items():
        new_df = new_df.replace(k, v)        
    return new_df.astype('float64').to_numpy() if as_numpy else new_df.astype('float64')


# In[ ]:





# In[ ]:


# Exericise 1: Compute the five-number summary, the boxplot, the mean, and the standard deviation
# for the annual salary per gender.
def p_01():
    print(five_number_summary(df, column='ConvertedComp', by='Gender', as_numpy=False))
    fig = hist_box_by(df, 'ConvertedComp', 'Gender', alias_column='Salary in USD')
    # fig.write_image('salary_gender_hist.pdf')
    fig.show()
    # describe_to(df, 'ConvertedComp', 'Gender', as_numpy=False)
p_01()


# In[ ]:


# Exercise 2: Compute the five-number summary, the boxplot, the mean, and the standard deviation
# for the annual salary per ethnicity.
def p_02():
    print(five_number_summary(df, column='ConvertedComp', by='Ethnicity', as_numpy=False))
    fig = hist_box_by(df, 'ConvertedComp', 'Ethnicity', alias_column='Salary in USD')
#     fig.write_image("salary_ethnicity_hist.pdf")
    fig.show()
p_02()


# In[ ]:


# Exercise 3: Compute the five-number summary, the boxplot, the mean, and the standard deviation
# for the annual salary per developer type.
def p_03():
    print(five_number_summary(df, column='ConvertedComp', by='DevType', as_numpy=False))
    fig = hist_box_by(df, 'ConvertedComp', 'DevType', alias_column='Salary',
                alias_by='developer type')
    fig.show()
    # fig.write_image("salary_devtype_hist.pdf")
p_03()


# In[ ]:


# Exercise 4: Compute the median, mean and standard deviation of 
# the annual salary per country.
def p_04():
    print(mean_std_median(df, column='ConvertedComp', by='Country', as_numpy=False))
p_04()


# In[ ]:


# Exercise 5: Obtain a bar plot with the frequencies of responses for each developer type.
def p_05():
    barplot(df, 'DevType', alias_column='Developer types').show()
p_05()


# In[ ]:


# Exercise 6: Plot histograms with 10 bins for the years of 
# experience with coding per gender.
def p_06():
    hist_by(df, 'YearsCode', by='Gender', nbins=10, 
            alias_column='Years coding', 
            alias_by='Programmers', func=np.sum).show()
p_06()


# In[ ]:


# Exercise 7: Plot histograms with 10 bins for the average number of working hours per week, per
# developer type.
def p_07():
    print(five_number_summary(df, 'WorkWeekHrs', 'Gender', as_numpy=False))
    hist_box_by(df, 'WorkWeekHrs', by='Gender', nbins=10, 
                alias_column='Hour worked by week', alias_by='Programmers').show()
#     # Seeing that some values makes charts skew, I decided to limit the values to possible week hours 
#     hist_box_by(df[(df['WorkWeekHrs'] < 24*7)], 'WorkWeekHrs', 
#                 by='Gender', nbins=10, alias_column='Hour worked by week', 
#                 alias_by='Programmers').show()
#     # Seeing that some values makes charts skew, I decided to limit the values to 12 hour per day
#     hist_box_by(df[(df['WorkWeekHrs'] < 12*7)], 'WorkWeekHrs', 
#                 by='Gender', nbins=10, alias_column='Hour worked by week', 
#                 alias_by='Programmers').show()
p_07()


# In[ ]:


# Exercise 8: Plot histograms with 10 bins for the age per gender.
def p_08():
    print(five_number_summary(df, 'Age', 'Gender', as_numpy=False))
    hist_box_by(df, 'Age', by='Gender', nbins=10, 
                alias_column='Age', alias_by='Programmers').show()
#     # Limitting age to greather than 10 years old
#     hist_box_by(df[df['Age'] > 10], 'Age', by='Gender', 
#                 nbins=10, alias_column='Hour worked by week', alias_by='Programmers').show()
p_08()


# In[ ]:


# Exercice 9: Compute the median, mean and standard deviation of the age per programming
# language.
def p_09():
    print(mean_std_median(df, 'Age', 'LanguageWorkedWith', as_numpy=False))
#     hist_box_by(df[(df['Age'] > 10) & (df['Age'] < 60)], 'Age', 'LanguageWorkedWith').show()
p_09()


# In[ ]:


# Exercise 10: Compute the correlation between years of experience and annual salary.
def p_10():
    corr, fig = scatter_correlation(df, 'YearsCode', 'ConvertedComp', 
                                    aliasx='Years coding', aliasy='Salary')
    print(corr)
    fig.show()
p_10()


# In[ ]:


# Exercise 11: Compute the correlation between the age and the annual salary.
# (df['Age'] > 20) & (df['Age'] < 70)
def p_11():
    corr, fig = scatter_correlation(df,
                                    'Age', 'ConvertedComp', 
                                    aliasx='Age', aliasy='Salary')
    print(corr)
    fig.show()
p_11()


# In[ ]:


# Exercise 12: Compute the correlation between educational level and annual salary. In this case,
# replace the string of the educational level by an ordinal index (e.g. Primary/elementary
# school = 1, Secondary school = 2, and so on).
def p_12():
    new_df = not_nulls(df, 'EdLevel', 'ConvertedComp')
    (uniques_values(new_df, 'EdLevel'))

    ranks = {
        'I never completed any formal education': 0.1, 
        'Primary/elementary school': 0.2, 
        'Secondary school': 0.3, 
        'Associate degree': 0.4,
        'Bachelor’s degree': 0.5, 
        'Some college/university study without earning a degree': 0.6, 
        'Master’s degree': 0.7, 
        'Other doctoral degree': 0.8
    }
    # sort_dict(ranks)
    y = ordinal_to_numeric(new_df['EdLevel'], ranks)
    x = new_df['ConvertedComp']
    corr, fig = scatter_correlation(x=x, y=y, text=new_df['EdLevel'], color=y,

                                   aliasx='Education level', aliasy='Salary')
    labels = list(sort_dict(ranks).keys())
    fig.update_yaxes(tickangle=25,


                     tickvals=[.1,.2,.3,.4,.5,.6,.7,.8],
                     ticktext=list(map(lambda s: s[:15] + '...', labels)))

    fig.update_layout(title='Salary according to education')
    print(corr)
    fig.show()
#     fig.write_image('salary_EdLevel.pdf')
#     barplot(df, 'EdLevel', sort=True).show()
p_12()


# In[ ]:


# Exercise 13: Obtain a bar plot with the frequencies of the different programming languages.
def p_13():
    barplot(df, column='LanguageWorkedWith', 
            alias_column='Programming languages', sort='reverse').show()
p_13()


# In[581]:


# Some code used to create plot ofr the report
def others():
    # Worked hours respect to education level
    new_df = not_nulls(df, 'WorkWeekHrs', 'EdLevel')
    new_df = new_df[new_df['WorkWeekHrs'] < 16*7]
    new_df = new_df[new_df['WorkWeekHrs'] > 5*7]

    ranks = {
        'I never completed any formal education': 0.1, 
        'Primary/elementary school': 0.2, 

        'Secondary school': 0.3, 
        'Associate degree': 0.4,
        'Bachelor’s degree': 0.5, 
        'Some college/university study without earning a degree': 0.6, 
        'Master’s degree': 0.7, 
        'Other doctoral degree': 0.8
    }
    # sort_dict(ranks)
    y = ordinal_to_numeric(new_df['EdLevel'], ranks)
    x = new_df['WorkWeekHrs']
    corr, fig = scatter_correlation(x=x, y=y, text=new_df['EdLevel'], color=y,         
                                   aliasx='Education level', aliasy='Worked hours per week')
    fig.update_yaxes(tickangle=25,                 
                     tickvals=[.1,.2,.3,.4,.5,.6,.7,.8],
                     ticktext=list(map(lambda s: s[:15] + '...', labels)))

    fig.update_layout(title='Worked hour per week according to education')
    # fig.show()
    fig.write_image('WorkedHours_EdLevel.pdf')
    print(corr)
#     -----------------------------------------------
    new_df = not_nulls(df, 'LanguageWorkedWith', 'ConvertedComp')
    amounts = new_df['LanguageWorkedWith'].apply(
        lambda item: item.count(';') + 1)
    # Histograms
    fig = hist_box_by(new_df, 'ConvertedComp', 'LanguageWorkedWith',
                alias_column='Salary in USD', 
                alias_by='Amount of programming language that they worked with',
                nbins=100, title='Salaries in USD by amount of programming languages')
    fig.show()
    # Sactter plot
    corr, fig = scatter_correlation(x=new_df['ConvertedComp'], y=amounts, 
                                    color=amounts, aliasx='Salaries', 
                                    aliasy='Amount of programming languages')
    print('Correlation')
    fig.show()
    fig.write_image('salary_amount.pdf')
    #     --------------------------------------------------
    new_df = not_nulls(df, 'LanguageWorkedWith')
    fig = barplot(new_df, 'LanguageWorkedWith', 'Programming languages', 
                  sort='reverse', color=('#588da8', '#001a33'))
    fig.write_image('programminglanguages.pdf')
    fig.show()
    # --------------------------------------------------------
    new_df = not_nulls(df, 'LanguageWorkedWith', 'ConvertedComp')
    d = describe_to(new_df[(new_df['ConvertedComp'] < 1000000) & (new_df['ConvertedComp'] > 1000)], 'ConvertedComp', 'LanguageWorkedWith', as_numpy=False)
    d = {k: v for k, v in d}
    d = sort_dict(d, func=lambda item : item[1]['count'], reverse=True)
    figure = go.Figure()
    figure.add_scatter(
                        x=list(d.keys()), 
                        y=list(map(lambda value: value['mean'],d.values())),
                        mode='lines+markers', name='Mean'
                      )
    figure.add_scatter(
                        x=list(d.keys()), 
                        y=list(map(lambda value: value['25%'],d.values())),
                        mode='lines+markers', name='Firts quartile'
                      )
    figure.add_scatter(
                        x=list(d.keys()), 
                        y=list(map(lambda value: value['50%'],d.values())),
                        mode='lines+markers', name='Median'
                      )
    figure.add_scatter(
                        x=list(d.keys()), 
                        y=list(map(lambda value: value['75%'],d.values())),
                        mode='lines+markers', name='Third quartile'
                      )
    figure.add_scatter(
                        x=list(d.keys()), 
                        y=list(map(lambda value: value['std'],d.values())),
                        mode='lines+markers', name='Standar deviation'
                      )
    figure.update_layout(
        title='Statistics for programming language by amount of programmers',
        xaxis=dict(title='Language programming in decendent order according to amount of users'),
        yaxis=dict(title='Salary in USD')
    )
    figure.show()
    figure.write_image('statistics_programm.pdf')
# ---------------------------------------------------------
    ranks = {
        'I never completed any formal education': 0.1, 
        'Primary/elementary school': 0.2,    
        'Secondary school': 0.3, 
        'Associate degree': 0.4,
        'Bachelor’s degree': 0.5, 
        'Some college/university study without earning a degree': 0.6, 
        'Master’s degree': 0.7, 
        'Other doctoral degree': 0.8
    }
    new_df = not_nulls(df, 'WorkWeekHrs', 'EdLevel')
    d = describe_to(new_df[(new_df['WorkWeekHrs'] < 16*7) & (new_df['WorkWeekHrs'] > 6*7)], 
                    'WorkWeekHrs', 'EdLevel', as_numpy=False)
    d = {k: v for k, v in d}
    d = {k[:15] + '...': d[k] for k in ranks.keys()}
    # d = sort_dict(d, func=lambda item : item[1]['count'], reverse=True)
    figure = go.Figure()
    figure.add_scatter(
                        x=list(d.keys()), 
                        y=list(map(lambda value: value['mean'],d.values())),
                        mode='lines+markers', name='Mean'
                      )
    figure.add_scatter(
                        x=list(d.keys()), 
                        y=list(map(lambda value: value['25%'],d.values())),
                        mode='lines+markers', name='Firts quartile'
                      )
    figure.add_scatter(
                        x=list(d.keys()), 
                        y=list(map(lambda value: value['50%'],d.values())),
                        mode='lines+markers', name='Median'
                      )
    figure.add_scatter(
                        x=list(d.keys()), 
                        y=list(map(lambda value: value['75%'],d.values())),
                        mode='lines+markers', name='Third quartile'
                      )
    # figure.add_scatter(
    #                     x=list(d.keys()), 
    #                     y=list(map(lambda value: value['count'],d.values())),
    #                     mode='lines+markers', name='Count'
    #                   )
    figure.update_layout(
        title='Statistics for worked hours per week by education',
        xaxis=dict(title='Worked hours'),
        yaxis=dict(title='Education Level')
    )
    figure.show()
    figure_1 = figure
    # figure.write_image('statistics_hours.pdf')
    # ----------------------------------------------------------------
    fig = barplot(df, 'Gender')
    fig.update_yaxes(title='')
    fig.update_xaxes(tickangle=25,                 
                     tickvals=[0,1,2],
                     ticktext=['Non-binary', 'Man', 'Woman'])
    fig.update_layout(height=300)
    fig.write_image('gender_count.pdf')
    fig.show()
    # -----------------------------------------------------------------
    # df['Gender'] = df['Gender'].apply(edit_value, args=('Non-binary, genderqueer, or gender non-conforminng', 'Non-binary'))
    new_df = not_nulls(df, 'Gender', 'DevType')
    new_df = new_df[new_df['Gender'].apply(contain_substr, 
                                  args=('Woman',))]

    fig = barplot(new_df, 'DevType', color=('#588da8', '#001a33'), 
                  alias_column='Developer type')
    # fig.data[0].x = list(map(lambda item: item[:20] + '...', fig.data[0].x))
    fig.update_layout(height=450, title='Developer type for women')
    fig.update_yaxes(title='')

    # fig.update_xaxes(tickangle=25,                 
    #                  tickvals=[0,1,2],
    #                  ticktext=['Non-binary', 'Man', 'Woman'])
    fig.write_image('women_devtype.pdf')
    fig.show()

