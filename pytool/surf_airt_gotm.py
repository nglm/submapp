from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates  as dates
import matplotlib.ticker as ticker
import matplotlib.font_manager
import pylab as pl

def unix_time_seconds(year,month,day):
#  Ref:
#    https://stackoverflow.com/questions/6999726/how-can-i-convert-a-datetime-object-to-milliseconds-since-epoch-unix-time-in-p
    import datetime
    dt = datetime.datetime(year,month,day)
    epoch = datetime.datetime.utcfromtimestamp(0) # total seconds since UTC 1970.1.1:00:00:00
    return int((dt - epoch).total_seconds())

# read GOTM variables

nc_f   = '../GotmFabmErsem-BATS.nc'

nc_fid = Dataset(nc_f, mode='r')

lons = nc_fid.variables['lon'][:]                # longitude []
lats = nc_fid.variables['lat'][:]                # latitude []
time = nc_fid.variables['time'][:]               # time [seconds since 1989.01.01.00:00]
taux = nc_fid.variables['tx'][:,:,:].squeeze()   # wind stress (! taux/rho0) in x [m2/s2]
tauy = nc_fid.variables['ty'][:,:,:].squeeze()   # wind stress (! tauy/rho0) in y [m2/s2]
u10  = nc_fid.variables['u10'][:,:,:].squeeze()  # 10m wind in x [m/s]
v10  = nc_fid.variables['v10'][:,:,:].squeeze()  # 10m wind in y [m/s]
airt = nc_fid.variables['airt'][:,:,:].squeeze() # 2m air temperature [C]
dswr = nc_fid.variables['I_0'][:,:,:].squeeze()  # incoming short wave radiation [J/m2]
hflx = nc_fid.variables['heat'][:,:,:].squeeze() # net heat flux [J/m2]
sst  = nc_fid.variables['sst'][:,:,:].squeeze()  # sea surface temperature [C]

nc_fid.close()

data = airt

# convert surface flux 

#--- convert time (seconds since 1989.01.01.00:00) to time_epoch (seconds since epoch) in Python 2.7

sse0 = unix_time_seconds(1989,1,1)       # gotm time origin [epoch]
time_epoch = time + sse0                 # time [epoch]
date_axis  = dates.epoch2num(time_epoch) # numpy array of Matplotlib dates

#--- check range of dates in original data

mpdate_ini = dates.epoch2num(time_epoch[0])           # Matplotlib dates
mpdate_end = dates.epoch2num(time_epoch[time.size-1]) # Matplotlib dates

print dates.num2date(mpdate_ini).strftime('Data Start: %Y-%m-%d')
print dates.num2date(mpdate_end).strftime('Data End  : %Y-%m-%d')

#--- plotting
# Refs:
#  

year_ini = 1992
year_end = 2007
yinc = 4

plt.rcParams["font.family"] = "Sans Serif"
fig = plt.figure(figsize=(9,10))

year_range = range(year_ini,year_end,yinc)

nrow = 0
for yini in year_range:

    nrow = nrow + 1
    ax = fig.add_subplot(len(year_range),1,nrow)

    #--- extract date axis and data for the specified date range
    yend=yini-1+yinc

    mpdate1 = dates.epoch2num(unix_time_seconds(yini,1,1))   # Matplotlib dates
    mpdate2 = dates.epoch2num(unix_time_seconds(yend+1,1,1)) # Matplotlib dates

    idx_range = np.where((date_axis >= mpdate1) & (date_axis < mpdate2)) # index search

    date_axis_range=date_axis[idx_range]
    data_range=data[idx_range].squeeze()

    print dates.num2date(mpdate1).strftime('Plot Start: %Y-%m-%d')
    print dates.num2date(mpdate2).strftime('Plot End  : %Y-%m-%d')

    #- define axis limit

    years  = dates.YearLocator(1,month=1,day=1)  # every year
    months = dates.YearLocator(1,month=6,day=16) # every June 16th
    yfmt   = dates.DateFormatter('%Y')           # YYYY
    #hfmt   = dates.DateFormatter('%Y/%m')       # YYYY/MM

    plt.plot(date_axis_range, data_range, color='royalblue')

    #- tick control
    #  Refs:
    #    https://stackoverflow.com/questions/43187398/center-x-axis-labels-in-line-plot
    #    https://matplotlib.org/gallery/ticks_and_spines/tick-formatters.html

    plt.gca().xaxis.set_minor_locator(months)
    plt.gca().xaxis.set_minor_formatter(yfmt)
    plt.gca().xaxis.set_major_locator(years)
    plt.gca().xaxis.set_major_formatter(ticker.NullFormatter())
    plt.gcf().autofmt_xdate()

    ax.tick_params(direction='out')
    ax.tick_params(axis='both', which='major', direction='out', top=False, right=False)
    ax.tick_params(axis='both', which='minor', length=0)

    #- add vertical guide line at mejor tick on x-axis

    ax.xaxis.grid(which="major", color='k', linestyle='-', linewidth=1)
    ax.axhline(0.0, color='k', linestyle='-', linewidth=0.5)
    
    #- axis range control

    xmin = date_axis_range.min()
    xmax = date_axis_range.max()
    plt.gca().set_xlim(xmin, xmax)

    ymax = 30.0
    ymin = 10.0
    plt.gca().set_ylim(ymin, ymax)

    #- axis labels

    plt.ylabel('2m AirT [C]')
    plt.xlabel('Time')

plt.tight_layout()
fig.suptitle("GOTM 2m Air Temperature at BATS (12 hourly)", fontsize=12)
plt.subplots_adjust(top=0.95)

outfile='../figs/plot_surf_airt_'+str(year_ini)+'-'+str(year_end)+'.png'
pl.savefig(outfile,dpi=300)
plt.show()
