

result = dict()

result['xyvp_1f'] = 'loss:1.61 --- fde:2.63 --- ade:1.08'
result['xyp_1f'] ='loss:1.68 --- fde:2.79 --- ade:1.15'
result['vp_1f'] = 'loss:1.54 --- fde:2.58 --- ade:1.06'
result['xy_1f'] = 'loss:1.80 -- fde:3.07 -- ade:1.29 -- Tfde:10.74 -- Tade:4.28' #30



def main():

    for key in result.keys():
        msg = key + ':' + result[key]
        print(msg)

if __name__ == "__main__":
    main()
    

