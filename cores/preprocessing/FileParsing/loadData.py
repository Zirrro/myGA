# -*- coding: utf-8 -*-
# @Time    : 2023/6/30 14:03
# @Author  : Salieri
# @FileName: transfer.py
# @Software: PyCharm
# @Comment : read data

import pandas as pd
import numpy as np
from nptdms import TdmsFile
import datetime


class ReadTDMSFile(object):
    """
    class:读取TDMS文件
    args:
    
    """

    def __init__(self, file):
        self.file = file

    def QuickRead(self):
        """快读提取内容"""
        with TdmsFile.open(self.file) as tdms_file:
            data_t = tdms_file.as_dataframe(time_index=False, absolute_time=True)
        return data_t

    def NormalRead(self):
        """提取完整内容"""
        dataDict = {}
        with TdmsFile.open(self.file) as tdms_file:
            all_groups = tdms_file.groups()
            for groupname in all_groups:
                groupnamecut = ReadTDMSFile.ExtractGroupNameGroup(groupname)
                group = tdms_file[groupnamecut]
                all_group_channels = group.channels()
                for channelname in all_group_channels:
                    channelnamecut = ReadTDMSFile.ExtractGroupNameChannel(channelname)[1:]
                    channel = group[channelnamecut]
                    all_channel_data = channel[:]
                    self.sampleRate = ReadTDMSFile.QueryProperties(channel.properties, 'channel')
                    if (len(all_channel_data) != 0):
                        key = groupnamecut + '/' + channelnamecut
                        dataDict[key] = all_channel_data
        data = pd.DataFrame(dataDict)
        return data

    @staticmethod
    def ExtractGroupNameGroup(groupname):
        """提取groupname"""
        groupname = str(groupname)
        start_quote = groupname.find("'")
        end_quote = groupname.rfind("'")
        return groupname[start_quote + 1:end_quote]

    @staticmethod
    def ExtractGroupNameChannel(channelname):
        """提取groupname"""
        channelname = str(channelname)
        start_quote = channelname.rfind("/'")
        end_quote = channelname.rfind("'")
        return channelname[start_quote + 1:end_quote]

    @staticmethod
    def QueryProperties(properties, source):
        """提取频率信息"""
        if "wf_increment" in properties:
            timeincrement = properties['wf_increment']
            print('fs', 1 / timeincrement)
            return 1 / timeincrement


class OPMDataManger(object):
    """
    class:光泵数据管理器
    """

    def __init__(self):
        self.data = []

    def addData(self, file):
        print(">>添加%s文件数据" % file)
        data_t = self.readOPMData(file)
        if (len(self.data) == 0):
            self.data = data_t
        else:
            self.data = pd.concat([self.data, data_t])

    def readOPMData(self, file):
        data_t = pd.read_csv(file, delim_whitespace=True)
        data_t.columns = ['hr', 'mm', 'ss', 'C1', 'C2', 'X', 'Y', 'Z', '经度', '纬度', '高度']
        data_t['hr'] = data_t['hr'] + 8
        return data_t
