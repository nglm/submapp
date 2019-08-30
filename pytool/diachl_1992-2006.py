from netCDF4 import Dataset
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates

nc_file='../GotmFabmErsem-BATS.nc'
fh = Dataset(nc_file, mode='r')

lons = fh.variables['lon'][:]
lats = fh.variables['lat'][:]
time = fh.variables['time'][:]
depth = fh.variables['z'][1,:,:,:].squeeze()
diachl = fh.variables['ECO_diachl'][:,:,:,:].squeeze()
flachl = fh.variables['ECO_flachl'][:,:,:,:].squeeze()
bgchl  = fh.variables['ECO_bgchl'][:,:,:,:].squeeze()

chla = diachl

# define time range

yini = '1992'
yend = '2006'

# chl-a data

data = np.transpose(chla)

# convert time (seconds since 1989.01.01.00:00) to time_epoch (seconds since epoch)
# in Python 2

epoch = datetime.datetime(1970,1,1)

sse0 = (datetime.datetime(1989,1,1) - epoch).total_seconds() 
time_epoch = time + sse0                                  # numpy array of epoch time
date_axis  = dates.epoch2num(time_epoch)                  # Matplotlib dates

sse1 = (datetime.datetime(int(yini),1,1) - epoch).total_seconds() 
sse2 = (datetime.datetime(int(yend),12,31) - epoch).total_seconds()

mpdate1 = dates.epoch2num(sse1)
mpdate2 = dates.epoch2num(sse2)

idx = (date_axis>mpdate1)*(date_axis<mpdate2)
idx = np.where((date_axis > mpdate1) & (date_axis < mpdate2))

date_axis_range=date_axis[idx]
data_range=data[:,idx].squeeze()

#--- setting up figures

fig = plt.figure()
ax = fig.add_subplot(111)

# define axis limit

years  = dates.YearLocator(1,month=1,day=1)               # every year
months = dates.MonthLocator()                             # every month
yfmt   = dates.DateFormatter('%Y')                        # YYYY
hfmt   = dates.DateFormatter('%Y/%m')                     # YYYY/MM

#plt.imshow(data_range, aspect='auto', origin='lower', vmin=0.0, vmax=0.8, cmap='viridis')
plt.pcolormesh(date_axis_range, depth, data_range, vmin=0.0, vmax=0.5, cmap='viridis')

# axis control

plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(yfmt)
#plt.gca().xaxis.set_minor_locator(months)
plt.gcf().autofmt_xdate()

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(direction='out')

xmin = date_axis_range.min()
xmax = date_axis_range.max()
plt.gca().set_xlim(xmin, xmax)

ymax = depth.max()
ymin = -200.
plt.gca().set_ylim(ymin, ymax)

plt.ylabel('Depth [m]')
plt.xlabel('Time')

cbar = plt.colorbar()
#cbar.ax.get_yaxis().set_ticks([0.0 0.1 0.2 0.3 0.4 0.5])
cbar.ax.set_ylabel('DIACHL [mg/m3]', rotation=270, va='bottom')

plt.savefig('../figs/diachl_'+yini+'-'+yend+'.png',format='png',dpi=1000)
plt.show()

