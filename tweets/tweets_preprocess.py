import string
f=open('tweets_data.txt','r')
label=[]
accepted=string.letters+string.digits+' '+'\n'
new=open('input.txt','w')
label_file=open('label.txt','w')

line_array=[]
label_array=[]

for line in f:
	phrase,label=line.split(' ',1)[1].rsplit(' ',1)
	processed=filter(lambda c: c in accepted,phrase)
	line_array.append(str(processed+'\n'))
	label_array.append(label)

final_list=list(set(line_array))
final_label_list=[]
for value in final_list:
	final_label_list.append(label_array[line_array.index(value)])

for index,row in enumerate(final_list):
	new.write(row)
	label_file.write(final_label_list[index])
