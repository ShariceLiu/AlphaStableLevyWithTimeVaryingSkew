import numpy as np
import matplotlib.pyplot as plt


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

def plot3d(data, save=False):
    # plot track data
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(track_data[0,:,0], track_data[0,:,1], track_data[0,:,2])

    if save:
        plt.savefig(save)


if __name__=='__main__':
    l = -0.05
    c = 10
    delta_t = 1
    sigma_w = 0.05
    sigma_mus = np.array([0.015, 0.015])
    mu0s = [0.1, 0.05]
    alpha = 1.2
    k_v = 1

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

    