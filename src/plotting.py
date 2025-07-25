import matplotlib.pyplot as plt
import numpy as np
from src.traffic_sim import *

def plot_vsls(vsl_matrix, time_steps, num_segm, v_free, T=10/3600, l=0.5):
    '''
    Given a matrix of VSL speeds, plot the VSL speeds as a heatmap. 
    (0,0) is the top left corner of the matrix. Rows represent time steps and columns represent segments.
    '''
    # Plot the optimal vsl matrix in a heatmap
    plt.figure(figsize=(15, 5))
    plt.imshow(vsl_matrix.T, cmap='RdBu', interpolation='nearest', aspect='auto', vmax=v_free, vmin=0)
    # Scale y axis labels by 3/10
    plt.yticks(np.arange(0, num_segm, 2), np.round(np.arange(0, num_segm*l, 2*l), 2), fontsize=14)
    # Scale x axis labels by 1/6
    plt.xticks(np.arange(0, time_steps, 60), (np.arange(0, time_steps * T * 60, 60 * T * 60)).astype(int), fontsize=14)

    # Label x and y axes
    plt.xlabel('Time (min)', fontsize=16)
    plt.ylabel('Distance (km)', fontsize=16)
    # Label the colorbar
    cbar = plt.colorbar(label='VSL speed (km/hr)')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('VSL speed (km/hr)', fontsize=16)
    # Reorient so that (0,0) in the matrix is coordinate (0,0) in the plot
    # Make the y axis longer so that the aspect ratio is correct
    plt.gca().invert_yaxis()
    plt.show()

def plot_outflow(traffic_demand, downstream_density, vsl_control, lane_map, v_free, T=10/3600, l=500/1000, seg=0):
    time_steps, num_segm = vsl_control.shape

    start_state =  (np.full(num_segm, 0), np.full(num_segm, 0), traffic_demand[0], 0)
    p_nocontrol, v_nocontrol, tts_nocontrol, queue_nocontrol= metanet_sim(T, l, start_state, np.full((time_steps, num_segm), v_free), traffic_demand, downstream_density, plotting=True, lanes=lane_map)
    p_optimal, v_optimal, tts_optimal, queue_optimal = metanet_sim(T, l, start_state, vsl_control, traffic_demand, downstream_density, plotting=True, lanes=lane_map)
    improvement_tts = round((tts_nocontrol - tts_optimal)/tts_nocontrol * 100, 1)

    lane_list = np.array(list(lane_map.values()))
    lane_array = np.tile(lane_list, (time_steps+1, 1))
    outflow_opt = v_optimal * p_optimal * lane_array
    outflow_nocontrol = v_nocontrol * p_nocontrol * lane_array

    plt.plot(outflow_opt[:, seg], label='Optimal')
    plt.plot(outflow_nocontrol[:, seg], label='No Control')
    plt.xlabel('Time (min)')
    plt.ylabel('Flow (veh/hr)')
    plt.xticks(np.arange(0, time_steps, 120), (np.arange(0, time_steps * T * 60, 120 * T * 60)).astype(int))
    plt.title(f'Flow out of segment {seg}')
    plt.axhline(y=2200 * lane_map[seg+1], color='r', linestyle='--', label='Capacity of next segment')
    
    plt.legend()
    plt.show()

def plot_nocontrol_control(traffic_demand, downstream_density, vsl_control, lane_map, v_free, T=10/3600, l=500/1000, plot_v=True, plot_p=False, plot_q=False, params=None):
    plot_var = np.array([plot_v, plot_p, plot_q])
    h = plot_var.sum()
    w = 3 if plot_v else 2
    widths = [2.4, 3, 3] if plot_v else [2.4, 3]
    time_steps, num_segm = vsl_control.shape
    print(time_steps, num_segm)

    start_state =  (np.full(num_segm, 0), np.full(num_segm, 0), traffic_demand[0], 0)
    
    if params is not None:
        p_nocontrol, v_nocontrol, queue_nocontrol, tts_nocontrol= metanet_sim_params(T, l, start_state, np.full((time_steps, num_segm), v_free), traffic_demand, downstream_density, params, real_data=False, plotting=True, lanes=lane_map)
        p_optimal, v_optimal,  queue_optimal, tts_optimal = metanet_sim_params(T, l, start_state, vsl_control, traffic_demand, downstream_density, params, real_data=False, plotting=True, lanes=lane_map)
    else:
        p_nocontrol, v_nocontrol,  queue_nocontrol, tts_nocontrol= metanet_sim(T, l, start_state, np.full((time_steps, num_segm), v_free), traffic_demand, downstream_density, plotting=True, lanes=lane_map)
        p_optimal, v_optimal,   queue_optimal, tts_optimal, = metanet_sim(T, l, start_state, vsl_control, traffic_demand, downstream_density, plotting=True, lanes=lane_map)

    total_inflow = np.sum(np.array(traffic_demand[0:time_steps]) * T)
    freeflow_tts = total_inflow * (num_segm * l) / v_free

    improvement_tts = np.round(((tts_nocontrol - freeflow_tts) - (tts_optimal - freeflow_tts))/(tts_nocontrol - freeflow_tts) * 100, 1)
    delay_nocontrol = tts_nocontrol - freeflow_tts
    delay_optimal = tts_optimal - freeflow_tts
    lane_list = np.array(list(lane_map.values()))
    lane_array = np.tile(lane_list, (time_steps+1, 1))
    
    #plot velocity, density, amd flow in order of plot_var
    fig, all_axs = plt.subplots(h, w, figsize=(30, 5*h), width_ratios=widths)
    axs = all_axs.flatten()
    i = 0
    #set title for first and second column

    if plot_v:
        axs[i].imshow(v_nocontrol.T, cmap='RdYlGn', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=v_free, interpolation='None')
        axs[i].set_title('Delay: {:.2f} veh-hr\nVelocity without control'.format(delay_nocontrol), fontsize=30, fontname='Times New Roman')
        im_v = axs[i+1].imshow(v_optimal.T, cmap='RdYlGn', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=v_free, interpolation='None')
        axs[i+1].set_title(f'Delay: {delay_optimal:.2f} veh-hr\nVelocity with control', fontsize=30, fontname='Times New Roman')
        cbar = fig.colorbar(im_v, label='Velocity (km/hr)', ax=axs[i+1])
        #increase font size of colorbar
        cbar.set_label('Velocity (km/hr)', fontsize=25, fontname='Times New Roman')
        cbar.ax.tick_params(labelsize=20)

        im_cc = axs[i+2].imshow(v_optimal.T - v_nocontrol.T, cmap='RdYlBu', extent=(0, time_steps+1, num_segm, 0), aspect='auto', vmin=-100, vmax=100, interpolation='None')
        axs[i+2].set_title(f'{improvement_tts}% delay reduction\nVelocity Improvement', fontsize=18)
        fig.colorbar(im_cc, label='Velocity Difference (km/hr)', ax=axs[i+2], extend='both')
        i += 3
    if plot_p:
        axs[i].imshow(p_nocontrol.T, cmap='RdYlGn_r', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=180, interpolation='None')
        axs[i].set_title('Density without control')
        im_p = axs[i+1].imshow(p_optimal.T, cmap='RdYlGn_r', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=180)
        axs[i+1].set_title('Density with control')
        fig.colorbar(im_p, label='Density (veh/km)', ax=axs[i+1])

        if plot_v:
            fig.delaxes(all_axs[1, 2])
        i += 3 if plot_v else 2
    if plot_q:
        max_val = max(np.max(v_nocontrol*p_nocontrol* lane_array), np.max(v_optimal*p_optimal*lane_array))
        axs[i].imshow(v_nocontrol.T * p_nocontrol.T * lane_array.T, cmap='viridis', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=max_val, interpolation='None')
        axs[i].set_title('Flow without control')
        im_q = axs[i+1].imshow(v_optimal.T * p_optimal.T * lane_array.T, cmap='viridis', aspect='auto', extent=(0, time_steps+1, num_segm, 0), vmin=0, vmax=max_val)
        axs[i+1].set_title('Flow with control')
        fig.colorbar(im_q, label='Flow (veh/hr)', ax=axs[i+1])

        if plot_v:
            fig.delaxes(all_axs[i//3,2])
    
    # reverse y axis for all subplots
    tick_freq = int(num_segm / 5)
    for ax in axs:
        ax.invert_yaxis()
        ax.set_yticks(np.arange(0, num_segm+1, tick_freq), np.round(np.arange(0, num_segm*l + 1, tick_freq*l), 2))
        ax.set_xticks(np.arange(0, time_steps, 120), (np.arange(0, time_steps * T * 60, 120 * T * 60)).astype(int))
        ax.set_xlabel('Time (min)', fontsize=25, fontname='Times New Roman')
        ax.set_ylabel('Distance (km)', fontsize=25, fontname='Times New Roman')
        ax.tick_params(labelsize=18)
    
    # Plot a horizontal line at 2.5 km (segment 5)
    for ax in axs:
        ax.axhline(10, color='black', linestyle='--')

    fig.subplots_adjust(hspace=0.4)
    plt.show()

    



