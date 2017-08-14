import sys
import json
input_dir = sys.argv[1]
output_file = sys.argv[2]
fh = open(output_file,'w')
score_list = []
for line in file(input_dir+"/log.txt"):
    json_str = line.split("json_stats:")[1].strip()
    item = json.loads(json_str)
    score_list.append(float(item["test_acc"]))
score_tuple = zip(range(1,len(score_list)+1), score_list)
max_epoch,max_score = sorted(score_tuple,key=lambda x:x[1],reverse=True)[0]
fh.write("max epoch:"+str(max_epoch)+"\t"+str(max_score)+"\n")
fh.write("\n".join(map(str,score_list)))
