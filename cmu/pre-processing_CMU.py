#import packages
import re
import random
import string

#read in the CMU dictionary
text_file_cmu = open("cmudict-0.7b", "r", encoding='latin-1')
cmu_lines = text_file_cmu.readlines()

#get rid of non-words, e.g., '('
cmu_lines = [x.replace('\n', '') for x in cmu_lines if x[0] != ';']
cmu_lines = [x for x in cmu_lines if x[0] in string.ascii_uppercase]
cmu_lines = [x for x in cmu_lines if '(' not in x]

#get rid of numbers from the input (as Rao et al 2015)
pairs = [[s for s in l.split('  ')] for l in cmu_lines]
regex = re.compile('[^a-zA-Z\']')
pairs = [[regex.sub('1', pair[0]), pair[1]] for pair in pairs]
pairs = [pair for pair in pairs if '1' not in pair[0]]

#shuffle and create train/dev/test sets, with and without stress
#numbers for split from Rao et al 2015

random.shuffle(pairs)

lines = [pair[0]+'  '+pair[1] for pair in pairs]

test_set_stress = lines[:12000]
dev_set_stress = lines[12000:12000+2760]
training_set_stress = lines[12000+2760:106837+2760+12000]

pairs = [[pair[0], re.sub('[0-9]', '', pair[1])] for pair in pairs ]
lines = [pair[0]+'  '+pair[1] for pair in pairs]

test_set = lines[:12000]
dev_set = lines[12000:12000+2760]
training_set = lines[12000+2760:106837+2760+12000]


#save

thefile = open('letter_only_with_stress/test_set.txt', 'w', encoding='latin-1')
for item in test_set_stress:
    thefile.write("%s\n" % item)
thefile.close()

thefile = open('letter_only_with_stress/dev_set.txt', 'w', encoding='latin-1')
for item in dev_set_stress:
    thefile.write("%s\n" % item)
thefile.close()

thefile = open('letter_only_with_stress/train_set.txt', 'w', encoding='latin-1')
for item in training_set_stress:
    thefile.write("%s\n" % item)
thefile.close()

thefile = open('letter_only_no_stress/test_set.txt', 'w', encoding='latin-1')
for item in test_set:
    thefile.write("%s\n" % item)
thefile.close()

thefile = open('letter_only_no_stress/dev_set.txt', 'w', encoding='latin-1')
for item in dev_set:
    thefile.write("%s\n" % item)
thefile.close()

thefile = open('letter_only_no_stress/train_set.txt', 'w', encoding='latin-1')
for item in training_set:
    thefile.write("%s\n" % item)
thefile.close()



