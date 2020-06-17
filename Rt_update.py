
# coding: utf-8

# In[37]:

# Growth rate function for  Rt histogram plots
import numpy as np
from lmfit import minimize, Parameters, report_fit
def get_gr_lm(DFSeries):
    
    '''
    This function will calculate the value of Rt for a series.
    In order to calculate R0 we can adapt this 
    https://en.wikipedia.org/wiki/Basic_reproduction_number#Estimation_methods

    Reminder: Rt = R0 when no intervention has been applied and there is mixing
    
    usage get_Rt(pandas_series)
    
    Input:
        Take in a pandas series with time and value
    Output:
        Return the value of R0
    In addition we use a different fitting approach than above
    '''
    import numpy as np
    from lmfit import minimize, Parameters, report_fit # added by [NO]
    
    params = Parameters()
    params.add('a', value=0.)
    params.add('b', value=0.)
    def get_residuals(params, x, data):
        a=params['a'].value
        b=params['b'].value
        model = a * np.exp(b*x)
        return data - model
    x = np.arange(len(DFSeries))
    data = list(DFSeries.values)
    out = minimize(get_residuals,params, args = (x,data))
    a = out.params['a'].value
    b = out.params['b'].value
    sigma_b = out.params['b'].stderr
    
    #a = LogisticFit[0][0]
    #b = LogisticFit[0][1] # This is also known as K from description in Wikipedia
    
    # The dooubling rate is given by T_d
    
    T_d = np.log(2)/b
        
    return (b) 

def Update_Rt(DF,Days_of_int=14):
    '''Take as input cumulative cases or deaths dataframe and 
     specify the number of days prior (default is 14 days) for which you want to 
     estimate R0 '''
    import numpy as np
    import pandas as pd
    np.random.seed(10)
    itr = 1000
    Rt_dict = dict()
    for col in DF.columns:
        b = get_gr_lm(DF.tail(Days_of_int)[col])
        R_t = np.zeros(itr)
        for i in range(itr):
            tau = np.random.normal(10.5,1.75)
            #tau = 4.4 * np.random.weibull(1.4) # Sampling from a Weibul distribution with shape parameter 1.04 and scale  parameter 4.4
            R_t[i] = np.exp(b*tau)
        Rt_dict.update({col:[np.percentile(R_t, 50),(np.percentile(R_t, 2.5),np.percentile(R_t, 97.5))]})
    Rt_est_df = pd.DataFrame.from_dict(Rt_dict, orient='index', columns=['Median', '95%CI'])
    return(Rt_est_df)


# In[ ]:



