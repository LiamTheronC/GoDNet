

result = dict()

result['xyvp_1f'] = 'loss:1.61 --- fde:2.63 --- ade:1.08'
result['xyp_1f'] ='loss:1.68 --- fde:2.79 --- ade:1.15'
result['vp_1f'] = ['loss:1.54 --- fde:2.58 --- ade:1.06',
                   'loss:1.51 -- fde:2.47 -- ade:1.04 -- Tfde:9.68 -- Tade:3.88'
                   'loss:1.45 -- fde:2.45 -- ade:1.03 -- Tfde:9.55 -- Tade:3.89(sparse)']

result['xy_1f'] = 'loss:1.80 -- fde:3.07 -- ade:1.29 -- Tfde:10.74 -- Tade:4.28' #3m
-----------------------------------------------------------------------------------------------------------------------


result['vp_5f_laneGCN'] = 'loss:1.16 -- fde:1.92 -- ade:0.79 -- Tfde:7.26 -- Tade:2.91' #60
result['vp_5f_sparse'] = 'loss:1.10 -- fde:1.72 -- ade:0.73 -- Tfde:6.56 -- Tade:2.69' #100


result['vp_5f_GANet'] = 'loss:4.30 -- fde:1.63 -- ade:0.71 -- Tfde:6.19 -- Tade:2.60' #110, 48, cls=5
result['vp_5f_GANet'] = 'loss:4.30 -- fde:1.59 -- ade:0.70 -- Tfde:5.98 -- Tade:2.53' #130, 48, cls=2
result['vp_5f_GANet'] = 'loss:3.73 -- fde:1.57 -- ade:0.69 -- Tfde:5.88 -- Tade:2.49' # 90, 246, cls = 2
result['vp_5f_GANet1'] = 'loss:3.51 -- fde:1.55 -- ade:0.67 -- Tfde:5.82 -- Tade:2.43'# 80, 246, cls =2
result['vp_5f_GANet1'] = 'loss:3.78 -- fde:1.82 -- ade:0.78 -- Tfde:6.68 -- Tade:2.70'# 40, 246, cls=2, M2A,A2A
result['vp_5f_GANet1'] = 'loss:4.42 -- fde:2.23 -- ade:0.93 -- Tfde:7.95 -- Tade:3.15'# 30,246, A2M, M2A, A2A

result['vp_5f_Order]' = 'loss:1.05 -- fde:1.67 -- ade:0.71 -- Tfde:6.39 -- Tade:2.62'# 80 M2A, A2A


-----------------------------------------------------------------------------------------------------------------------

result['xy_10f_lanGCN'] = 'loss:0.97 -- fde:1.51 -- ade:0.65 -- Tfde:5.71 -- Tade:2.36'
result['xy_10f_laneGCN_focal'] = 'loss:3.24 -- fde:2.14 -- ade:0.92 -- Tfde:6.74 -- Tade:2.73'
result['vpt_10f_laneGCN1'] = 'loss:1.02 -- fde:1.62 -- ade:0.68 -- Tfde:6.27 -- Tade:2.54'
result['xy_10f_GANet'] = 'loss:2.43 -- fde:1.53 -- ade:0.68 -- Tfde:5.73 -- Tade:2.44'
result['xy_10f_GANet_midgoal'] = 'loss:1.66 -- fde:1.45 -- ade:0.63 -- Tfde:5.51 -- Tade:2.28'


-------------------------------------final-----------------------------------------------------------------------------
result['vp_5f_GANet'] = 'loss:3.60 -- fde:1.53 -- ade:0.68 -- Tfde:5.71 -- Tade:2.42' # 70, 246, A2M,M2M, (M2A,A2A)
result['vp_5f_GANet'] = 'loss:3.06 -- fde:1.32 -- ade:0.60 -- Tfde:4.95 -- Tade:2.16'#246,(M2A,A2A)
#246,(A2M,M2M,M2A,A2A)

result['vp_100f_GANet'] = 'loss:2.25 -- fde:0.95 -- ade:0.44 -- Tfde:3.58 -- Tade:1.59' # 246, (M2A,A2A)
result['laneGCN'] = 'loss:0.77 -- fde:1.17 -- ade:0.51 -- Tfde:4.55 -- Tade:1.89'






def main():

    for key in result.keys():
        msg = key + ':' + result[key]
        print(msg)

if __name__ == "__main__":
    main()
    

