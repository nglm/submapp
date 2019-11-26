from math import exp,log, ceil
import numpy as np
from numpy import linalg as LA
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import time
import datetime
import matplotlib.dates as dates
from statistics import mean, variance, stdev
import os



def data_extraction() :
        
    def unix_time_seconds(year,month,day):
    #  Ref:
    #    https://stackoverflow.com/questions/6999726/how-can-i-convert-a-datetime-object-to-milliseconds-since-epoch-unix-time-in-p
        import datetime
        dt = datetime.datetime(year,month,day)
        epoch = datetime.datetime.utcfromtimestamp(0) # total seconds since UTC 1970.1.1:00:00:00
        return int((dt - epoch).total_seconds())

    def moving_average(x, w):
    #   Refs:
    #     https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
        import numpy as np
        return np.convolve(x,np.ones(w),'same')/w

    #read the datafile 
    cwd = os.getcwd()
    #print("Current working directory: ", cwd)
    nc_f   = cwd + '/GotmFabmErsem-BATS.nc'
    nc_fid = Dataset(nc_f, mode='r')

    #extract data
    time = nc_fid.variables['time'][:]
    depth = nc_fid.variables['z'][1,:,:,:].squeeze()
    p_data = depth.size
    depth_min = -100
    depth = depth[depth > depth_min]
    p = depth.size
    temp  = nc_fid.variables['temp'][:,:,:,:].squeeze()

    u10  = nc_fid.variables['u10'][:,:,:].squeeze()  # 10m wind in x [m/s]
    v10  = nc_fid.variables['v10'][:,:,:].squeeze()  # 10m wind in y [m/s]
    dswr = nc_fid.variables['I_0'][:,:,:].squeeze()  # incoming short wave radiation [J/m2]
    #hflx = nc_fid.variables['heat'][:,:,:].squeeze() # net surface heat fluxes [J/m2]
    airt = nc_fid.variables['airt'][:,:,:].squeeze() # 2m air temperature [C]
    sst  = nc_fid.variables['sst'][:,:,:].squeeze()  # sea surface temperature [C]

    nc_fid.close()

    # apply moving average
    #   Refs:
    #     https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy

    #data_ma = moving_average(sst,4)
    #sst = data_ma

    data_ma = moving_average(dswr,15)
    dswr = data_ma

    #data_ma = moving_average(airt,30)
    #airt = data_ma

    # apply moving average

    ws10 = np.sqrt(np.square(u10)+np.square(v10))
    ws10_ma = moving_average(ws10,15)
    ws10 = ws10_ma

    sse0 = unix_time_seconds(1989,1,1)       # gotm time origin
    time_epoch = time + sse0                                  # numpy array of epoch time
    date_axis  = dates.epoch2num(time_epoch)                  # Matplotlib dates

    yini = 1992
    yend = 2008

    nb_years = 16
    temp_y_tmp = [np.empty([p,]) for i in range(nb_years)]
    temp_y = [np.empty([]) for i in range(nb_years)]
    dswr_y = [np.empty([]) for i in range(nb_years)]
    airt_y = [np.empty([]) for i in range(nb_years)]
    ws10_y = [np.empty([]) for i in range(nb_years)]
    sst_y = [np.empty([]) for i in range(nb_years)]

    for y in range(nb_years):

        #--- extract date axis and data for the specified date range
        yini= 1992+y
        yend=yini+1

        mpdate1 = dates.epoch2num(unix_time_seconds(yini,1,1)) # Matplotlib dates
        if yend == 2008:
            mpdate2 = dates.epoch2num(unix_time_seconds(2007,12,26)) # Matplotlib dates
        else:
            mpdate2 = dates.epoch2num(unix_time_seconds(yend,1,1)) # Matplotlib dates

        idx_range = np.where((date_axis >= mpdate1) & (date_axis < mpdate2)) # index search

        sst_y[y] = sst[idx_range].squeeze()
        dswr_y[y] = dswr[idx_range].squeeze()
        airt_y[y] = airt[idx_range].squeeze()
        ws10_y[y] = ws10[idx_range].squeeze()
        temp_y[y] = temp[idx_range].squeeze()
        temp_y_tmp[y] = temp_y[y][:,p_data-p:]
        (temp_y[y], depth_final) = mean_depth(temp_y_tmp[y], depth, step_size=5)
        #temp_y[y], depth_final = temp_y_tmp[y], depth


    return([temp_y, sst_y, dswr_y, airt_y, ws10_y, depth_final])
    

def mean_steps(data, step_size):
    if len(data.shape)==1:
        (T,) = data.shape
        data = np.reshape(data,(T,1))
    (T,p) = data.shape
    T_new = ceil(T/step_size)

    new_data = np.zeros((T_new,p))
    for t in range(0, T_new-1):
        new_data[t] = np.mean(data[t*step_size:(t+1)*step_size], axis=0)
    new_data[-1] = np.mean(data[(T_new-1)*step_size:], axis=0)

    if p == 1 :
        data.squeeze()
    return(new_data)

def mean_depth(data, depth=None, step_size=1):
    (T, p) = data.shape
    p_new = ceil(p/step_size)
    new_data = np.zeros((T,p_new))
    new_depth = np.zeros(p_new) 
    for p in range(0,p_new-1):
        start = p*step_size
        end = start + step_size
        new_data[:,p] = np.mean(data[:, start:end], axis=1)
        if depth is not None:
            new_depth[p] = np.mean(depth[start:end])
    start = (p_new-1)*step_size
    new_data[:,-1] = np.mean(data[:,start:])
    if depth is not None:
        new_depth[-1] = np.mean(depth[start:])
        return new_data, new_depth
    else:
        return new_data

def create_mean_year(years):
    # the mean year will have the minimum duration
    T_y = []
    for year in years:
        T_y.append(len(year))
    T = min(T_y)
    mean_year = np.zeros_like(years[0][:T])
    for year in years:
        mean_year += year[:T]
    mean_year = mean_year/len(years)
    return mean_year

def remove_mean_year(years, mean_year=None):
    if mean_year is None:
        mean_year = create_mean_year(years)
    new_years = []
    T = len(mean_year)
    for year in years:
        new_years.append(year[:T]-mean_year)
    return new_years

def subtract_year(y1, y2):
    T = min(len(y1),len(y2))
    diff = y1[:T] - y2[:T]
    return diff

def print_data(data, y_axis=None, legend=" ",save=False, path=None,
filename=None,zmin=None,zmax=None, cmap=None, figsize=None) :
    if len(data.shape)==1:
        (T_year,) = data.shape
        data = np.reshape(data,(T_year,1))
    (T_year,n) = data.shape
    data = np.transpose(data)
    if zmin is None:
        zmin = np.nanmin(data)
    if zmax is None:
        zmax = np.nanmax(data)

    plt.figure(figsize=figsize)
    if (n>1) :
        #axis initialization
        x = np.transpose(range(T_year))
        #y = depth
        if y_axis is None:
            y = np.linspace(0,n,n)
        else:
            y = np.transpose(y_axis)
        X, Y = np.meshgrid(x, y)
        #plot
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.pcolor(X, Y, data, cmap=cmap, vmin=zmin, vmax=zmax, label="Temperature [C]")
        plt.colorbar(label="Temperature [C]")
        plt.ylabel('Depth [m]')
        
    else :
        x = np.linspace(0,T_year,T_year)
        plt.plot(x, data[0,:])
        plt.axis([0, T_year-1, zmin, zmax])
    plt.xlabel('Time [arbitrary indices]') 
    plt.title(legend)

    if save:
        if path is None:
            path = "../figs/Map/"
        os.makedirs(path, exist_ok=True)
        if filename is None:
            filename = "myPlot"
        plt.savefig(path + filename +'.png',format='png', dpi=1000)

    plt.show()
   

def print_error(
        relerr,
        legend=None, 
        zmin=None,
        zmax=None,
        save=False,
        filename=None,
        path=None,
        ) :   
    T_year = len(relerr)
    if legend is None:
        legend = "Error"
    if zmin is None:
        zmin = np.nanmin(relerr)
    if zmax is None:
        zmax = np.nanmax(relerr)
    #plot error
    plt.figure()
    x = np.linspace(0,T_year,T_year)
    plt.xlabel('Time [arbitrary indices]')
    plt.ylabel('Relative error')
    plt.title(legend)
    plt.axis([0, T_year-1, zmin, zmax])
    plt.plot(x, relerr)
    if save:
        if path is None:
            path = "../figs/Map/"
        os.makedirs(path, exist_ok=True)
        if filename is None:
            filename = "err"
        plt.savefig(path + filename +'.png',format='png', dpi=1000)
    plt.show()
    plt.close()
