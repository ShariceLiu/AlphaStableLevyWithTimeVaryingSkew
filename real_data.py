import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

def extract_track(filepath):
    with open(filepath,'r') as f:
        lines= f.readlines()
    
    length = len(lines[0].split(','))
    array_data = np.zeros((len(lines),length))
    for i,l in enumerate(lines):
        list_data = l.strip().split(',')
        array_data[i,:] = np.array(list_data, dtype = 'f')

    num_fish = len(set(array_data[:,1]))
    num_frame = len(set(array_data[:,0]))

    track_data=np.zeros((num_fish, num_frame, 3))

    for i in range(num_fish):
        fish_idx = np.linspace(0,num_frame,num=num_frame, endpoint=False, dtype='i')*num_fish+i
        ith_track = array_data[fish_idx,2:5]
        track_data[i,:,:] = ith_track

    return track_data

def extract_mat_data(filename = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\dataEurUS.mat"):
    mat_contents = sio.loadmat(filename)['last_traded']
    ts = mat_contents['t_l'][0,0].flatten()
    xs = mat_contents['z_l'][0,0].flatten()

    # return xs, ts

    # ts -= ts[0]

    t0= ts[0]
    xs_non_rep = [xs[0]]
    ts_non_rep = [ts[0]]
    for x, t in zip(xs[1:], ts[1:]):
        if t==t0: # repetitive time
            continue
        xs_non_rep.append(x)
        ts_non_rep.append(t)

        t0 = t

    return np.array(xs_non_rep), np.array(ts_non_rep)

def extract_csv(filename = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVIDIA CORPORATION (01-23-2025 09.30 _ 01-29-2025 16.00).csv"):
    # extract from 1.27, every 5 min?
    nvda_data_reloaded= pd.read_csv(filename, index_col=0, parse_dates=True)

    plt.figure()
    plt.plot(nvda_data_reloaded['Price'])
    plt.xlabel('Minutes')
    plt.ylabel('Price')
    plt.savefig(r'experiments\figure\simplified\nvdia\price.png')

    plt.figure()
    plt.plot(nvda_data_reloaded['Price']['2025-01-27'])
    plt.xlabel('Minutes')
    plt.ylabel('Price')
    plt.savefig(r'experiments\figure\simplified\nvdia\price01_27.png')
    plt.figure()
    plt.plot(nvda_data_reloaded['Price']['2025-01-28'])
    plt.xlabel('Minutes')
    plt.ylabel('Price')
    plt.savefig(r'experiments\figure\simplified\nvdia\price01_28.png')
    plt.figure()
    plt.plot(nvda_data_reloaded['Price']['2025-01-29'])
    plt.xlabel('Minutes')
    plt.ylabel('Price')
    plt.savefig(r'experiments\figure\simplified\nvdia\price01_29.png')
    return

def plot3d(track_data, save=False):
    # plot track data
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(track_data[0,:,0], track_data[0,:,1], track_data[0,:,2])

    if save:
        plt.savefig(save)

def plot_fish_data():
    file_path=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt'

    track_data = extract_track(file_path)

    plot3d(track_data, save=r'experiments\data\real_data\fish2')

    N= track_data.shape[1]-2
    
    track_x = track_data[1,:,0]
    track_v = track_x[1:]-track_x[:-1]
    track_a = track_v[1:]-track_v[:-1]
    # y_ns_noisy = track_x+np.random.normal(0, sigma_w*k_v, N)

    # hist of track a: so the increments of v
    # plt.hist(track_a[:500], bins=50)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(track_x)
    plt.ylabel('displacement')
    plt.subplot(2,1,2)
    plt.plot(track_v)
    plt.ylabel('velocity')
    plt.xlabel('Time (n)')
    plt.savefig(r'experiments\data\real_data\fish2_x')

def nvidia_tl():
    csvfile = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVDA.USUSD_Ticks_20.06.2025-20.06.2025.csv"
    nvda_data_reloaded= pd.read_csv(csvfile)
    x_ns = (np.array(nvda_data_reloaded['Ask'])+np.array(nvda_data_reloaded['Bid']))/2

    nvda_data_reloaded['Local time'] = pd.to_datetime(nvda_data_reloaded['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
    nvda_data_reloaded['Local time'] = (nvda_data_reloaded['Local time'] - nvda_data_reloaded['Local time'].iloc[0]).dt.total_seconds()

    t_ns = np.array(nvda_data_reloaded['Local time'])

    start = 20500
    end = 21000
    plt.plot(x_ns[start:end])
    plt.show()

if __name__=='__main__':
    csvfile = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVDA.USUSD_Ticks_20.06.2025-20.06.2025.csv"
    nvda_data_reloaded= pd.read_csv(csvfile)
    x_ns = (np.array(nvda_data_reloaded['Ask'])+np.array(nvda_data_reloaded['Bid']))/2

    nvda_data_reloaded['Local time'] = pd.to_datetime(nvda_data_reloaded['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
    nvda_data_reloaded['Local time'] = (nvda_data_reloaded['Local time'] - nvda_data_reloaded['Local time'].iloc[0]).dt.total_seconds()

    t_ns = np.array(nvda_data_reloaded['Local time'])

    start = 20500
    end = 21000
    plt.plot(x_ns[start:end])
    plt.show()
    # plt.savefig(r'experiments\figure\simplified\nvdia_tl\data.png')
    
