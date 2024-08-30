# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:06:15 2022

@author: poi
"""
###########
##第48-51行是从站点csv数据中提取站点的日期和降水量（单位是0.1mm，需要除以10）的程序段落
##后面是计算指标的函数，以及绘图的代码。
###########

# # ###################################################################################################################################
# # ######################################################地面数据按照日期解译#########################################
# # ###################################################################################################################################

####GHCH-Daily验证数据-----中国0.01度网格
#每个站点 文件名 3600——1800内容：2007-2021年 15*365=5475
#先用公式算天数 365*年 距离1958年1月1日的天数，年份
# # ##纬度：15-55 近似：14.95--55.05  编号：1049-1450 *数组提取写法：1049-1451    14.95-55.05    4011
# # ##经度：70-140 近似：69.95--140.05  编号：2499-3200 *数组提取写法：2499-3201   69.95-140.05  7011
##对于一个站点， 判断如果在经纬度范围内，则逐条计算日期，存储
# import pandas as pd
# import os
# import numpy as np
# import time
# from datetime import *
# import scipy.io

# station_loc=np.zeros((123741,2))
# station_loc_used=0
# folder_path='./GHCH-中国站点/'
# files=os.listdir(folder_path)
# gmap=np.zeros((402,702))

# ##构建闰年查找表
# mark=['2015-01-01','2016-02-29', '2019-12-31']
# mark_days=np.zeros(len(mark))
# for mark_i in range(len(mark)):
#     obs_date=datetime.strptime(mark[mark_i],'%Y-%m-%d')
#     begin_date=datetime.strptime('2015-01-01','%Y-%m-%d')
#     delta=obs_date-begin_date
#     mark_days[mark_i]=delta.days
# print(mark_days)

# ##构建经纬度网格
# lon_net=np.arange(69.95,140.15,0.1)
# lat_net=np.arange(14.95,55.15,0.1)

# for i in range(len(files)):
#     save_flag=0
#     matrix=np.zeros(1825)-1
#     table=pd.read_csv(folder_path+files[i])
#     table=table.loc[:, ['DATE', 'LATITUDE', 'LONGITUDE', 'PRCP']]
#     table=table[table.PRCP>0]
#     mat=table.to_numpy()
    
#     if table.shape[0]>0:
#         if mat[0,1]>=14.95 and mat[0,1]<=55.15 and mat[0,2]>=69.95 and mat[0,2]<=140.15:
#             station_loc[i,:]=mat[0,1:3]
            
#             ##遍历每个观测日期，赋值给长序列
#             for j in range(np.shape(mat)[0]):
#                 obs_date=datetime.strptime(mat[j,0],'%Y-%m-%d')
#                 begin_date=datetime.strptime('2015-01-01','%Y-%m-%d')
#                 end_date=datetime.strptime('2019-12-31','%Y-%m-%d')
#                 delta=obs_date-begin_date
#                 days=delta.days       
#                 if days<0:
#                     continue
#                 for m in range(len(mark_days)-1):
#                     if days>mark_days[m] and days<mark_days[m+1]:
#                         print(mark_days[m])
#                         matrix[days-m]= mat[j,3]/10
#                 save_flag=1
#             ##确定行列号
#             lat_id=np.argmin(np.abs(lat_net-mat[0,1]))
#             lon_id=np.argmin(np.abs(lon_net-mat[0,2]))
#             if save_flag==1:
#                 gmap[lat_id,lon_id]=gmap[lat_id,lon_id]+1
#                 scipy.io.savemat('./GHCN-Daily-CR-001/record/'+str(lon_id)+'_'+str(lat_id)+'_'+str(mat[0,2])+'_'+str(mat[0,1])+'.mat', {'data': matrix})


# np.save('./GHCN-Daily-CR-001/gmap',gmap)
# np.save('./GHCN-Daily-CR-001/station_loc',station_loc)
 
# g = np.load('./GHCN-Daily-CR-001/gmap.npy')
# s = np.load('./GHCN-Daily-CR-001/station_loc.npy')
 
# scipy.io.savemat('gmap.mat', {'gmap': g})
# scipy.io.savemat('station_loc.mat', {'station_loc': s})
 



# ##################################################################################################################################
# #####################################################地面数据与输入、输出数据匹配#########################################
# ##################################################################################################################################

##############地面验证   
###0.01度网格    ####保存的是真实经纬度！！
#  对于每个站点记录，获取对应网格、对应时间的降水数据：  
#  站点编号、经度网格、纬度网格、日期序号、地面站点数据、
#  IMERG-Early、SM2RAIN、0.1度融合降尺度结果、0.01度融合结果、IMERG-Final


import os
import numpy as np
# import time
# from datetime import *
import scipy.io
import matplotlib.colors as mcolors
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
files=os.listdir('./GHCN-Daily-CR-001/record_402_702/')

result=np.zeros((1,11),'float32')

for i in range(len(files)):


    lon=files[i].split('.mat')[0].split('_')[0]
    lat=files[i].split('.mat')[0].split('_')[1]
    true_lon=files[i].split('.mat')[0].split('_')[2]
    true_lat=files[i].split('.mat')[0].split('_')[3]
    

    ###地面站点数据
    load_mat = scipy.io.loadmat('./GHCN-Daily-CR-001/record_402_702/'+files[i])
    data_insitu = load_mat['data']
    data_insitu=data_insitu[0,:]
    print(data_insitu)
    
    ###IMERG-Early
    load_mat = scipy.io.loadmat('I:/水文+SR/新建文件夹/output/IMERG-Early_output/'+files[i])
    data_imerg_early = load_mat['sequence']
    data_imerg_early=data_imerg_early[0,:]
    data_imerg_early[data_imerg_early<-1]=-1
    
    ###SM2RAIN
    load_mat = scipy.io.loadmat('I:/水文+SR/新建文件夹/output/SM2RAIN_output/'+files[i])
    data_sm2rain = load_mat['sequence']
    data_sm2rain=data_sm2rain[0,:]
    data_sm2rain[data_sm2rain<-1]=-1
    
    ###IMERG-Final
    load_mat = scipy.io.loadmat('I:/水文+SR/新建文件夹/output/IMERG-Final_output/'+files[i])
    data_imerg_final = load_mat['sequence']
    data_imerg_final=data_imerg_final[0,:]
    data_imerg_final[data_imerg_final<-1]=-1
    
    ###CNN
    load_mat = scipy.io.loadmat('I:/水文+SR/新建文件夹/output/fusion_result_small_ablation_output/'+files[i])
    data_CNN_fusion = load_mat['sequence']
    data_CNN_fusion=data_CNN_fusion[0,:]
    data_CNN_fusion[data_CNN_fusion<-1]=-1
    
    ###DIVF
    load_mat = scipy.io.loadmat('I:/水文+SR/新建文件夹/output/inter_fusion_01/'+files[i])
    data_DIVF_fusion = load_mat['sequence']
    data_DIVF_fusion=data_DIVF_fusion[0,:]
    data_DIVF_fusion[data_DIVF_fusion<-1]=-1
    
    ###Ours
    load_mat = scipy.io.loadmat('I:/水文+SR/新建文件夹/output/fusion_result_small_output/'+files[i])
    data_Ours_fusion = load_mat['sequence']
    data_Ours_fusion=data_Ours_fusion[0,:]
    data_Ours_fusion[data_Ours_fusion<-1]=-1
    
    #######分析
    matrix=np.zeros((1825,9),float)-1
    matrix[:,0]=data_insitu[:]
    matrix[:,1]=data_imerg_early[:]
    matrix[:,2]=data_sm2rain[:]
    matrix[:,3]=data_imerg_final[:]
    matrix[:,4]=data_CNN_fusion[:]
    matrix[:,5]=data_DIVF_fusion[:]
    matrix[:,6]=data_Ours_fusion[:]
    
    if np.all((matrix[:,0]>0)==False):
        continue
        
    a=matrix[matrix[:,0]>=0]

    days=np.where(matrix[:,0]>=0)[0]
        
    this_result=np.zeros((a.shape[0],11),'float32')
    this_result[:,0]=i
    this_result[:,1]=true_lon
    this_result[:,2]=true_lat
    this_result[:,3]=days
    this_result[:,4]=a[:,0]
    this_result[:,5]=a[:,1]
    this_result[:,6]=a[:,2]
    this_result[:,7]=a[:,3]
    this_result[:,8]=a[:,4]
    this_result[:,9]=a[:,5]
    this_result[:,10]=a[:,6]


    
    result=np.row_stack((result,this_result))
    print(result.shape)

np.save('./result/result',result[1:,:])
scipy.io.savemat('./result/result.mat', {'result': result})



###############################################################################################################################
##################################################数据分析#########################################
###############################################################################################################################


#######################################0.01  验证数据分析
#       0        1        2           3           4          5          6            7            8            9
#### 站点编号  经度网格、纬度网格、日期序号、地面站点数据、IMERG-Early、SM2RAIN、0.01融合结果、0.1融合降尺度结果、IMERG-Final
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib as mpl
import math
import geopandas as gpd
#['2007-01-01','2008-02-29','2012-02-29','2016-02-29','2020-02-29']
#        #2007  2008  2009    2010   2011   2012   2013   2014   2015   2016   2017   2018   2019   2020   2021  end
#year_flag=[0 ,  365  ,   731 ,  1096 , 1461 , 1826 ,    2192 , 2557  ,2922 , 3287,    3653 , 4018 , 4383 , 4748 ,   5114 ,10000]  
year_flag=[0, 364, 730, 1085, 1450, 5000]
title=['IMERG-Early_0.1','SM2RAIN_0.1', 'IMERG-Final_0.1', 'CNN_fusion_0.1', 'DIVF_0.1', 'Ours_fusion_0.1']
year_r_matrix=np.zeros((6,5))

plt.rcParams['font.family'] = 'Times New Roman'  # 设置全局字体为Times New Roman

##逐年份绘图，散点图，站点图       
for year in range(2015,2020):
    result=np.load('./result/result.npy')
    year_start=year_flag[year-2015]
    year_end=year_flag[year-2015+1]
    index=np.where((result[:,3]>=year_start)&(result[:,3]<year_end))[0]
    result=result[index,:].copy()
    index=np.where((result[:,4]>0)&(result[:,5]>0)&(result[:,6]>0)&(result[:,7]>0)&(result[:,8]>0))[0]

    result=result[index,:]
    r_imerg_early=np.corrcoef(result[:,4],result[:,5])[0,1]
    r_sm2rain=np.corrcoef(result[:,4],result[:,6])[0,1]
    r_imerg_final=np.corrcoef(result[:,4],result[:,7])[0,1]
    r_CNN_fusion=np.corrcoef(result[:,4],result[:,8])[0,1]
    r_DIVF_fusion=np.corrcoef(result[:,4],result[:,9])[0,1]
    r_Ours_fusion=np.corrcoef(result[:,4],result[:,10])[0,1]
    
    r=[r_imerg_early, r_sm2rain, r_imerg_final, r_CNN_fusion, r_DIVF_fusion, r_Ours_fusion]
    year_r_matrix[:,year-2015]=r

    for v in range(5,11):
        plt.scatter(result[:,4],result[:,v],s=0.5)   
        plt.plot([0, 400],[0, 400],ls='--')
        plt.axis([0, 400, 0, 400])
        
        plt.title(title[v-5]+' '+str(year)+' CC='+str(np.round(r[v-5],3)))
        plt.savefig('./figure/R_scatterplot/'+title[v-5]+'_'+str(year)+'.png')
        plt.close()
        


                              
   
        
###2007-2019年
result=np.load('./result/result.npy')
year_start=year_flag[2015-2015]
year_end=year_flag[2019-2015+1]
index=np.where((result[:,3]>=year_start)&(result[:,3]<year_end))[0]
result=result[index,:].copy()
index=np.where((result[:,4]>0)&(result[:,5]>0))[0]

result=result[index,:]
r_imerg_early=np.corrcoef(result[:,4],result[:,5])[0,1]
r_sm2rain=np.corrcoef(result[:,4], result[:,6])[0,1]
r_imerg_final=np.corrcoef(result[:,4],result[:,7])[0,1]
r_CNN_fusion=np.corrcoef(result[:,4],result[:,8])[0,1]
r_DIVF_fusion=np.corrcoef(result[:,4],result[:,9])[0,1]
r_Ours_fusion=np.corrcoef(result[:,4],result[:,10])[0,1]

r=[r_imerg_early, r_sm2rain, r_imerg_final, r_CNN_fusion, r_DIVF_fusion, r_Ours_fusion]
print(r)
year_r_matrix[:,-1]=r

x_positions = [80, 100, 120, 140]  # 想要显示的定位器位置
x_labels = ['80°E', '100°E', '120°E', '140°E']  # 与定位器位置相对应的标签
y_positions = [20, 30, 40, 50]
y_labels = ['20°N', '30°N', '40°N', '50°N']

for v in range(5,11):
    plt.scatter(result[:,4],result[:,v],s=0.5)   
    plt.plot([0,400],[0,400],ls='--')
    plt.axis([0, 400, 0, 400])
    rmse = math.sqrt(np.mean(np.square(result[:,v]-result[:,4])))
    Rbias = np.sum(result[:,v]-result[:,4], axis=0)/ np.sum(result[:, 4]) 


    plt.title(title[v-5]+' CC='+str(np.round(r[v-5],3))+' RBias='+str(np.round(Rbias,3))+' RMSE='+str(np.round(rmse,3)))
    plt.savefig('./figure/R_scatterplot/'+title[v-5]+'.png')
    plt.close()
    
    

stations=np.unique(result[:,0])
r_station=np.zeros((len(stations),16))
for s in range(len(stations)):

    s_index=np.where(result[:,0]==stations[s])[0]
    r_station[s,0]=result[s_index[0],0]
    r_station[s,1]=result[s_index[0],1]
    r_station[s,2]=result[s_index[0],2]
    r_station[s,3]=len(s_index)
    
    
    r_station[s,4]=np.corrcoef(result[s_index,4],result[s_index,5])[0,1]
    r_station[s,5]=np.corrcoef(result[s_index,4],result[s_index,6])[0,1]
    r_station[s,6]=np.corrcoef(result[s_index,4],result[s_index,7])[0,1]
    r_station[s,7]=np.corrcoef(result[s_index,4],result[s_index,8])[0,1]
    r_station[s,8]=np.corrcoef(result[s_index,4],result[s_index,9])[0,1]
    r_station[s,9]=np.corrcoef(result[s_index,4],result[s_index,10])[0,1]

    
    r_station[s,10]=np.sqrt(np.sum((result[s_index,4]-result[s_index,5])**2)/len(result[s_index,4]))
    r_station[s,11]=np.sqrt(np.sum((result[s_index,4]-result[s_index,6])**2)/len(result[s_index,4]))
    r_station[s,12]=np.sqrt(np.sum((result[s_index,4]-result[s_index,7])**2)/len(result[s_index,4]))
    r_station[s,13]=np.sqrt(np.sum((result[s_index,4]-result[s_index,8])**2)/len(result[s_index,4]))
    r_station[s,14]=np.sqrt(np.sum((result[s_index,4]-result[s_index,9])**2)/len(result[s_index,4]))
    r_station[s,15]=np.sqrt(np.sum((result[s_index,4]-result[s_index,10])**2)/len(result[s_index,4]))

    
r_station=r_station[np.where(np.isnan(r_station[:,4]) !=1)[0],:]
r_station=r_station[np.where(np.isnan(r_station[:,5]) !=1)[0],:]
r_station=r_station[np.where(np.isnan(r_station[:,6]) !=1)[0],:]
r_station=r_station[np.where(np.isnan(r_station[:,7]) !=1)[0],:]  
r_station=r_station[np.where(np.isnan(r_station[:,8]) !=1)[0],:]
r_station=r_station[np.where(np.isnan(r_station[:,9]) !=1)[0],:]  


for v in range(4, 10):
    plt.figure(figsize=(12, 4))
    china_map = gpd.read_file('./中华人民共和国.shp')

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)  # 调整子图的间距
    ax.set_aspect('equal')
    china_map.plot(ax=ax, color='none', edgecolor='black')

    # Create custom colormap
    cmap = mcolors.ListedColormap(['blue', 'cyan', 'yellow', 'orange', 'red'])

    # Normalize the color scale
    norm = mcolors.BoundaryNorm(boundaries=np.linspace(0, 1, 6), ncolors=5)

    # Scatter plot with custom colormap
    plt.scatter(r_station[:, 1], r_station[:, 2], c=r_station[:, v], alpha=1, cmap=cmap, edgecolor='black', linewidth=1, s=100, norm=norm)

    # Add colorbar with distinct colors
    # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(0, 1, 6))
    # cbar.set_label('Label', fontsize=14)  # Add label to colorbar
    # cbar.ax.tick_params(labelsize=12)  # Adjust tick label size

    plt.title(title[v - 4], fontsize=20, fontname='Times New Roman')
    plt.xticks(x_positions, x_labels, fontsize=20, fontname='Times New Roman')  # 更改x轴标签
    plt.yticks(y_positions, y_labels, fontsize=20, fontname='Times New Roman')  # 更改y轴标签
    plt.xlim(70, 140)  # 根据实际经度范围进行调整
    plt.ylim(15, 55)  # 根据实际纬度范围进行调整
    plt.savefig('./figure/R_station/'+title[v-4]+'.png')
    plt.close()
    

for v in range(10, 16):
    plt.figure(figsize=(12, 4))
    china_map = gpd.read_file('./中华人民共和国.shp')

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)  # 调整子图的间距
    ax.set_aspect('equal')
    china_map.plot(ax=ax, color='none', edgecolor='black')
    
    cmap = mcolors.ListedColormap(['blue', 'cyan', 'yellow', 'orange', 'red'])
    
    norm = mcolors.BoundaryNorm(boundaries=np.linspace(0, 15, 6), ncolors=5)

    plt.scatter(r_station[:, 1], r_station[:, 2], c=r_station[:, v], alpha=1, cmap=cmap, edgecolor='black', linewidth=1, s=100, norm=norm)
    
    # Add colorbar
    # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(0, 1, 6))
    # cbar.set_label('Label', fontsize=14)  # Add label to colorbar
    # cbar.ax.tick_params(labelsize=12)  # Adjust tick label size

    plt.title(title[v - 10], fontsize=20, fontname='Times New Roman')
    plt.xticks(x_positions, x_labels, fontsize=20, fontname='Times New Roman')  # 更改x轴标签
    plt.yticks(y_positions, y_labels, fontsize=20, fontname='Times New Roman')  # 更改y轴标签
    plt.xlim(70, 140)  # 根据实际经度范围进行调整
    plt.ylim(15, 55)   # 根据实际纬度范围进行调整
    
    plt.savefig('./figure/RMSE_station/' + title[v - 10] + '.png')
    plt.close()

     







################################################################################################################################
###################################################绘图#########################################
################################################################################################################################

# ####################绘制散点对称密度图#################################
#0.01
import time
start_time = time.time() 
year_flag=[0, 364, 730, 1085, 1450, 5000]
title=['IMERG-Early_0.1','SM2RAIN_0.1', 'IMERG-Final_0.1', 'CNN_fusion_0.1', 'DIVF_0.1', 'Ours_fusion_0.1']
year_r_matrix=np.zeros((5,11))

result=np.load('./result/result.npy')
year_start=year_flag[2015-2015]
year_end=year_flag[2019-2015+1]
index=np.where((result[:,3]>=year_start)&(result[:,3]<year_end))[0]
result=result[index,:].copy()
index=np.where((result[:,4]>0)&(result[:,5]>0)&(result[:,6]>0)&(result[:,7]>0)&(result[:,8]>0))[0]

result=result[index,:]

#这里是保存散点密度的矩阵，因为点很多，所以计算会很慢
for i in range(5,11):

    # x = np.array(range(0, 10000))
    # y = np.array([i + random.gauss(0, 500) for i in range(10000)])
    select=np.arange(0,len(index),1)
    x=result[select,4]
    y=result[select,i]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    np.save('./散点对称图/z/'+title[i-5],z)
    end_time = time.time()    # 程序结束时间
    run_time = end_time - start_time    # 程序的运行时间，单位为秒
    print(run_time)
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    # "font.size": 80,
#     "mathtext.fontset":'stix',
}
rcParams.update(config)
plt.rcParams['xtick.labelsize'] = 12  # 设置x轴刻度标签的字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置y轴刻度标签的字体大小

for i in range(5,11):

    x=result[:,4]
    y=result[:,i]
    z=np.load('./散点对称图/z/'+title[i-5]+'.npy')    
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(x, y, c=z, cmap='Spectral_r',s=0.5) 
    ax.set_ylabel(title[i-5], fontsize=20, rotation='vertical', va='bottom')  # 设置y轴标签
    ax.set_xlabel('')  # 设置x轴标签为空
    plt.plot([0,100],[0,100],ls='--',c='k',linewidth =1)
    plt.axis([0, 100, 0, 100])
    ax.set_aspect('equal', adjustable='box')
    ax.text(50, -10, 'In-situ measurement', ha='center', fontsize=20)  # 在底部添加自定义文本
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.savefig('./figure/scatter density/'+title[i-5]+'.png')
    
    plt.show()



