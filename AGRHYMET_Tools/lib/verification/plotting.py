import logging
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import seaborn as sns
sns.set(style="whitegrid")

import lib.tools


def plot_map_global(da, vrange=None, cmap_label=None, cmap='jet', out_cfg=None, title=None, fntag=None, pois=False):

    if np.isnan(da.values).all():
        print('Plot contains only NaNs')
        return

    # Reformat grid to be globally representable by plotting lib
    da_plot = da.roll(lon=720, roll_coords=False).assign_coords(lon=da.roll(lon=720, roll_coords=False).lon - 180)
    # da_plot = da.roll(lon=1800).assign_coords(lon=da.roll(lon=1800).lon - 360)
    # da_plot = da.roll(lon=1800, roll_coords=False).assign_coords(lon=da.roll(lon=1800, roll_coords=False).lon - 360)

    # Create base plot
    ext = [-180, 180, -90, 90]
    proj = ccrs.PlateCarree(central_longitude=0)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj))
    myplot = da_plot.plot.imshow(transform=proj, add_colorbar=False, cmap=cmap, extent=ext)
    # ax.coastlines('50m', linewidth=0.2)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2)

    # Set color range
    if vrange is not None:
        myplot.set_clim(vmin=vrange[0], vmax=vrange[1])

    # Create colorbar below plot
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(size="5%", pad=0.1, axes_class=plt.Axes, pack_start=True)
    fig.add_axes(ax_cb)
    cb = plt.colorbar(myplot, orientation="horizontal", cax=ax_cb, extend='both')
    if cmap_label:
        cb.set_label(cmap_label)

    # Set title
    ax.set_title(da.name)
    if title:
        ax.set_title(title)

    # Draw POIS as circle
    if pois:
        for poi, name in pois.items():
            logging.info('......... Add POI: {} / {}'.format(name, poi))
            c = Circle(poi, 1, edgecolor='red', facecolor='none', transform=ccrs.Geodetic())
            ax.add_patch(c)

    # Export plot
    fig.set_size_inches(20, 20, forward=True)
    if out_cfg:
        dirtag = 'global'
        if fntag is None:
            fntag = da.name
            dirtag = 'model'
        fn = '{}/plot_{}_{}_{}_{}.png'.format(out_cfg['dir_out'], out_cfg['param_name'], out_cfg['model'], dirtag, fntag)
        fig.savefig(fn, dpi=72, bbox_inches='tight')
        plt.close('all')
        logging.info('...... created plot {}/{}'.format(dirtag, fntag))


def plot_map_europe(da, vmin=None, vmax=None, cmap_label=None, cmap='jet', out_cfg=None, title=None, fntag=None):
    if np.isnan(da.values).all():
        print('Plot contains only NaNs')
        return

    # Create base plot
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj))
    # myplot = da.plot.pcolormesh(transform=proj, add_colorbar=False, cmap=cmap)
    myplot = da.plot.imshow(transform=proj, add_colorbar=False, cmap=cmap)
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-17, 67, 25, 72], crs=proj)

    # Create gridlines/labels
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Set color range
    if vmin is not None and vmax is not None:
        myplot.set_clim(vmin=vmin, vmax=vmax)

    # Create colorbar below plot
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(size="5%", pad=0.4, axes_class=plt.Axes, pack_start=True)
    fig.add_axes(ax_cb)
    cb = plt.colorbar(myplot, orientation="horizontal", cax=ax_cb, extend='both')
    if cmap_label:
        cb.set_label(cmap_label)

    # Set title
    ax.set_title(da.name)
    if title:
        ax.set_title(title)

    fig.set_size_inches(10, 10, forward=True)
    # Export plot
    if out_cfg:
        if fntag is None:
            fntag = da.name
        fn = '{}/plot_{}_{}_europe_{}.png'.format(out_cfg['dir_out'], out_cfg['param_name'], out_cfg['model'], fntag)
        fig.savefig(fn, dpi=100, bbox_inches='tight')
        plt.close('all')


def plot_scatter_timeseries(gridpoint, gp_name, x0_gp, y_gp, yhat_gp, resid, resid_before, correction, out_cfg, eq_txt, model_eval_gp):

    # Prepare plot data
    features_seas = lib.tools.calc_seasonal_features(x0_gp.time.data)
    count = [x for x in range(0, x0_gp.time.data.size)]
    plotdata = pd.DataFrame({'x0': x0_gp.data,
                             'y': y_gp.data,
                             'y_hat': yhat_gp.data,
                             'resid': resid.data,
                             'resid_before': resid_before.data,
                             'T1': features_seas['T1'],
                             'T2': features_seas['T2'],
                             'T3': features_seas['T3'],
                             'T4': features_seas['T4'],
                             'count': count}, index=x0_gp.time.data)

    # Prepare figure
    fig = plt.figure(figsize=(27, 15))
    # fig.suptitle('{} ({}/{})'.format(gp_name, gridpoint[0], gridpoint[1]), fontsize=14)
    gs = gridspec.GridSpec(4, 3, width_ratios=[2, 2, 1], height_ratios=[2.9, 0.2, 2.9, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[2, :])
    ax5 = plt.subplot(gs[3, :], sharex=ax4)
    axdummy = plt.subplot(gs[1, :])
    axdummy.grid(False)
    axdummy.set_axis_off()

    # def scatter_yyhat_X(x, title, fn_tag):
    #     mpl.rcParams['figure.figsize'] = (15, 8)
    #     plotdata.plot(kind='scatter', x=x, y='y', title=title, c=count, cmap='nipy_spectral', alpha=0.5)
    #     plt.scatter(plotdata[x], plotdata.y_hat, c=count, s=20, cmap='nipy_spectral', alpha=0.5, marker='s')
    #     fn = '{}/plot_{}_{}_scatter{}.png'.format(out_cfg['dir_out'], out_cfg['param_name'], out_cfg['model'], fn_tag)
    #     plt.savefig(fn, dpi=100, bbox_inches='tight')
    #     plt.close('all')
    # # scatter_yyhat_X('x0', 'X0 vs y/yhat (count in colors)', 2)
    # # scatter_yyhat_X('T1', 'T1 vs y/yhat (count in colors)', 3)
    # # scatter_yyhat_X('T2', 'T2 vs y/yhat (count in colors)', 4)
    # # scatter_yyhat_X('T3', 'T3 vs y/yhat (count in colors)', 5)
    # # scatter_yyhat_X('T4', 'T4 vs y/yhat (count in colors)', 6)

    # Draw first scatter plot
    vmin = plotdata[['x0', 'y']].min().min()
    vmax = plotdata[['x0', 'y']].max().max()
    ax1.plot([vmin, vmax], [vmin, vmax], color='gray', lw=1.5, linestyle='-')
    ax1.scatter(plotdata.x0, plotdata.y, color='black', s=20, marker='o', label='ERA5 vs HRES')
    ax1.scatter(plotdata.x0, plotdata.y_hat, color='g', s=20, marker='o', label='ERA5 vs ERA5-corrected')
    ax1.set_title('Scatter HRES vs ERA5-corrected')
    ax1.set_xlabel('ERA5')
    ax1.set_ylabel('HRES/ERA5-corrected')
    ax1.legend()

    # Draw second scatter plot
    vmin = plotdata[['y_hat', 'y_hat']].min().min()
    vmax = plotdata[['y_hat', 'y_hat']].max().max()
    ax2.scatter(plotdata.y_hat, plotdata.resid, c=plotdata.T3, cmap='RdYlBu_r', s=20, marker='o')
    ax2.plot([vmin, vmax], [0, 0], color='gray', lw=1.5, linestyle='-')
    ax2.set_title('Residual ERA5-corrected')
    ax2.set_xlabel('ERA5-corrected')
    ax2.set_ylabel('Residual')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='NH-Summer', markerfacecolor='r'),
                       Line2D([0], [0], marker='o', color='w', label='NH-Winter', markerfacecolor='b')]
    ax2.legend(handles=legend_elements)

    # Draw table
    ax3.set_title('Summary / Metrics')
    ax3.grid(False)
    ax3.set_axis_off()
    tbl = []
    tbl.append(['Parameter:', out_cfg['param_name']])
    tbl.append(['Location:', '{} ({}x{})'.format(gp_name, gridpoint[0], gridpoint[1])])
    tbl.append(['Equation:', '{}\n      {}'.format(eq_txt[0], eq_txt[1])])
    for varname, da in model_eval_gp.data_vars.items():
        tbl.append([varname + ':', '{:.2f}'.format(float(da.data))])
    tbl = pd.DataFrame(tbl)
    ax3.table(cellText=tbl.values, bbox = [-0.1, 0, 1.1, 1], colWidths=[0.25, 0.75])
    ax3.tables[0].auto_set_font_size(False)
    for key, cell in ax3.tables[0].get_celld().items():
        cell.set_linewidth(0.2)
        cell._loc = 'left'
        cell.set_fontsize(11.5)
        if (2, 1) == key:
            cell.set_fontsize(10)

    # Draw both timeseries
    ax4.set_title('Time series')
    plotdata[['y']].plot(style='k-', linewidth=1, ax=ax4)
    plotdata[['x0']].plot(style=['r:'], linewidth=2, ax=ax4)
    plotdata[['y_hat']].plot(style=['g-'], linewidth=1.5, ax=ax4)
    ax4.legend(['HRES', 'ERA5', 'ERA5-corrected'])

    ax5.set_title('Time series residuals')
    plotdata[['resid_before']].plot(style=['r:'], linewidth=2, ax=ax5)
    plotdata[['resid']].plot(style=['g-'], linewidth=1.5, ax=ax5)
    ax5.axhline(y=0, color='gray', lw=1.5, linestyle='-')
    ax5.legend(['residual ERA5', 'residual ERA5-corrected'])

    # Save figure
    fn = '{}/plot_{}_{}_gridpoint_{}.png'.format(out_cfg['dir_out'], out_cfg['param_name'], out_cfg['model'], gp_name)
    fig.savefig(fn, dpi=100, bbox_inches='tight')
    plt.close('all')
