import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

#makedirs('png')

def plot(fig1, font_size = 11, fr_thk =  2, xy_tick_thk = 2, is_xlabel = True, xlabel = 'itrations(/100)', is_ylabel = True, ylabel = 'MSE', 
         is_title = False, title = 'input',is_lim = False, xlim = [0,60], ylim = [0,2], is_legend = False, legend_size = 12, leg_loc = 'upper right', bitx = 1,bity = 1, alpha=0.8):

    fig1.spines['bottom'].set_linewidth(fr_thk)
    fig1.spines['left'].set_linewidth(fr_thk)
    fig1.spines['top'].set_linewidth(fr_thk)
    fig1.spines['right'].set_linewidth(fr_thk)
    fig1.xaxis.set_tick_params(width=xy_tick_thk)
    fig1.yaxis.set_tick_params(width=xy_tick_thk)

    #fig1.legend(('NODE $k$ = 1', 'NODE $k$ = 2'))
    if is_xlabel:
        fig1.set_xlabel(xlabel, fontsize=font_size)
    if is_ylabel:
        fig1.set_ylabel(ylabel, fontsize=font_size)
    if is_title:
        fig1.set_title(title, fontsize=font_size)
        
    fig1.tick_params(axis= 'both', direction = 'in', width=xy_tick_thk, labelsize=font_size) 
    fig1.tick_params(axis='x', labelsize= font_size)
    #fig1.tick_params(axis='y', labelsize=font_size)
    matplotlib.pyplot.xticks(fontsize=font_size)
    matplotlib.pyplot.yticks(fontsize=font_size)
    
    if is_lim:
        fig1.set_xlim(xlim)
        fig1.set_ylim(ylim)
   
    fig1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.'+str(bity)+'f'))
    fig1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.'+str(bitx)+'f'))
    if is_legend: 
        legend = fig1.legend(loc=leg_loc,frameon=True, shadow=False, fontsize=legend_size, facecolor='white', framealpha  = alpha)
        legend.get_frame().set_linewidth(0.0)
    #plt.subplots_adjust(hspace=hspace)
    plt.tight_layout()        
    
def visualize(t, ext, true_y, pred_y, start, stop,legend_loc = 'best', hspace = 1,legend_size = 12, alpha = 0.8, plt_name = 'figure'):
    
    fig = plt.figure(figsize=(15, 4), facecolor='white')
    
    fig1 = fig.add_subplot(211, frameon=True)
    fig2 = fig.add_subplot(212, frameon=True)
    
    for i in range(0, ext.size()[2]):
        fig1.plot(t.cpu().numpy(), ext.cpu().numpy()[:, 0, i])
    
    fig1.set_title('input')
    fig1.set_xlim(start, stop)
    
    legend = fig1.legend(('Ku','DMI'), loc=legend_loc,frameon=True, shadow=False, fontsize=legend_size, facecolor='white', framealpha  = alpha)
    legend.get_frame().set_linewidth(0.0)
    
    
    for i in range(0, true_y.size()[2]):
        fig2.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, i], 'b-')
        fig2.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, i], '--', color='orange')
        
    fig2.set_title('output')
    fig2.set_xlim(start, stop)
    
    legend = fig2.legend(('Mumax','NODE'), loc=legend_loc,frameon=True, shadow=False, fontsize=legend_size, facecolor='white', framealpha  = alpha)
    legend.get_frame().set_linewidth(0.0)
    
    fig1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    fig2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    plt.tight_layout()
    #plt.savefig('png/{:03d}'.format(itr))
    plt.savefig(plt_name)
    plt.draw()
    #plt.pause(0.001)
    plt.subplots_adjust(hspace=hspace)
    #plt.show(block=False)
    
def visualize_21(t,  y1, y2, start, stop):
    fig = plt.figure(figsize=(15, 4), facecolor='white')
    
    fig1 = fig.add_subplot(211, frameon=True)
    fig2 = fig.add_subplot(212, frameon=True)
    
    fig1.set_title('input')
    #fig1.set_xlabel('t /(5 ps) ')
    fig1.set_xlabel('steps')
    fig1.set_ylabel('u.n.')
    fig1.set_xlim(start, stop)
    
    for i in range(0, y1.size()[2]):
        fig1.plot(t.cpu().numpy(), y1.cpu().numpy()[:, 0, i])
    
    fig2.set_title('output')
    #fig2.set_xlabel('t /(5 ps)')
    fig2.set_xlabel('steps')
    fig2.set_ylabel('$m_{z}$ (×10)')
    fig2.set_xlim(start, stop)
    
    for i in range(0, y2.size()[2]):
        fig2.plot(t.cpu().numpy(), y2.cpu().numpy()[:, 0, i])
    #fig2.tight_layout()
    #fig2.legend(('Xc(t)','Yc(t)','Xc^(t)','Yc^(t)'), loc="best")
    plt.subplots_adjust(hspace=1) 