#encoding: utf-8

from openpyxl import load_workbook
from xlrd import open_workbook
import sys
import re

def strQ2B(ustring):
    """translate full-width character to half-width character"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        if inside_code >= 65 and inside_code <= 90:
            inside_code += 32
        rstring += chr(inside_code)
    return rstring

def clean(s):
    s = strQ2B(s)
    s = re.sub(r'<br>', "", s)
    s = re.sub(r'\s+', " ", s)
    return s

def xlsx2txt(source_file, target_file)
    wb = load_workbook(source_file)
    ws = wb['Sheet1']
    wf = open(target_file, 'w')
    count = 0
    for row in ws.rows:
        data_list = []
        if count == 0:
            continue
        for cell in row:
            data_list.append(cell.value)
        count += 1
        wf.write("\t".join([str(data_list[0]), str(data_list[1])]) + "\n")

def xld2txt(source_file, target_file):
    wb = open_workbook(source_file)
    for i in range(2, 9):
        ws = wb.sheets()[i]
        wf = open(target_file, 'a')
        count = 0
        for row in range(ws.nrows):
            data_list = []
            for cell in ws.row_values(row):
                data_list.append(cell)
            count += 1;
            if count == 1:
                continue
            wf.write("\t".join([str(data_list[0]), str(data_list[1])]) + "\n")
