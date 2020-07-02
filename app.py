import csv
import operator


with open('table.csv', 'r') as file:
    reader = csv.reader(file)
    sortedlist = sorted(reader, key=operator.itemgetter(2), reverse=True)
    print(sortedlist)