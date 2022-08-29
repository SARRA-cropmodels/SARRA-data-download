#!/usr/bin/env python

from ecmwfapi import ECMWFService
server = ECMWFService('mars')

datasets = {'inst':    {'request': {'param': '33.128/164.128/165.128/166.128/167.128/168.128/141.128',
                                    'step': '0/3/6/9'}},
            'accmnmx': {'request': {'param': '169.128/228.128/26.228/27.228',
                                    'step': '3/6/9/12'}},
            # 'landsea2': {'request': {'param': '172.128/129.128',
            #                         'step': '0'}},
            }

std = {'class': 'od',
       'expver': '1',
       'stream': 'oper',
       'type': 'fc',
       'levtype': 'sfc',
       'time': '00:00:00/12:00:00'}

dates   = [
        #    '20160331/to/20160331',
        #    '20160401/to/20160430',
        #    '20160501/to/20160531',
        #    '20160601/to/20160630',
        #    '20160701/to/20160731',
        #    '20160801/to/20160831',
        #    '20160901/to/20160930',
        #    '20161001/to/20161031',
        #    '20161101/to/20161130',
        #    '20161201/to/20161231',
        #    '20170101/to/20170131',
        #    '20170201/to/20170228',
        #    '20170301/to/20170331',
        #    '20170401/to/20170430',
        #    '20170501/to/20170531',
        #    '20170601/to/20170630',
        #    '20170701/to/20170731',
        #    '20170801/to/20170831',
        #    '20170901/to/20170930',
        #    '20171001/to/20171031',
        #    '20171101/to/20171130',
        #    '20171201/to/20171231',
        #    '20180101/to/20180131',
        #    '20180201/to/20180228',
        #    '20180301/to/20180331',
        #    '20180401/to/20180430',
        #    '20180501/to/20180531',
        #    '20180601/to/20180630',
        #    '20180701/to/20180731',
        #    '20180801/to/20180831',
        #    '20180901/to/20180930',
        #    '20181001/to/20181031',
        #    '20181101/to/20181130',
        #    '20181201/to/20181231',
        #    '20190101/to/20190131',
        #    '20190201/to/20190228',
        #    '20190301/to/20190331',
        #    '20190401/to/20190430',
        #    '20190501/to/20190531',
        #    '20190601/to/20190630',
        #    '20190701/to/20190731',
        #    '20190801/to/20190831',
        #    '20190901/to/20190930',
        #    '20191001/to/20191031',
        #    '20191101/to/20191130',
           '20191201/to/20191231',
           '20200101/to/20200101',
           ]

# formats = ['grib1', 'grib2', 'netcdf']
formats = ['netcdf']

# regrids = ['emoslib', 'mir']
regrids = ['mir']

grids   = ['0.25/0.25']

for format in formats:
    for regrid in regrids:
        for grid in grids:
            for date in dates:
                for key, val in datasets.items():

                    while True:
                        print('=========================================================================================================================================================================================')
                        print('{}/{}/{}'.format(format, regrid, key))

                        try:
                            # Standard fixed request keys
                            request = std

                            # Add changing keys
                            request['date'] = date
                            request['ppengine'] = regrid
                            request['format'] = format
                            request['grid'] = grid

                            # Add dataset specific keys
                            request['param'] = val['request']['param']
                            request['step']  = val['request']['step']

                            target = './data/EChres_{}_{}_{}.{}'.format(date.replace('/', '')[:6], key, regrid, format)

                            print(target)
                            print(request)
                            server.execute(request, target)

                        except:
                            print('{}/{}/{}/{}: execute did not work, RETRY...'.format(format, regrid, key, grid))
                            break
                        break
