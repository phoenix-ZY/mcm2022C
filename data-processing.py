import csv
csvFile = open("LBMA-GOLD.csv", "r")
reader = csv.reader(csvFile)
golddata={}
for item in reader:
    if reader.line_num==1:#此处为表头，跳过
        continue
    golddata[item[0]]=item[1]
csvFile.close()
print(golddata)

csvFile = open("BCHAIN-MKPRU.csv", "r")
reader = csv.reader(csvFile)
data=[]
for item in reader:
    if reader.line_num==1:
        continue
    if item[0] in golddata.keys():
        item.append(golddata[item[0]])
    else:
        item.append(0)
    data.append(item)
csvFile.close()
print(data)

fileHeader = ["Date", "Value","USD(PM)"]#表头
csvFile=open("GoodLuck.csv","w")
writer=csv.writer(csvFile, lineterminator='\n')#设置lineterminator，否则在excel中打开csv时会多出空行
writer.writerow(fileHeader)
writer.writerows(data)
# for item in data:
#     writer.writerow(item)

csvFile.close()
