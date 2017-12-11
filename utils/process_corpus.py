START = "<s>"
STOP = "</s>"

files = [
    "../cmu/new_no_stress/dev_set.txt",
    "../cmu/new_no_stress/test_set.txt",
    "../cmu/new_no_stress/train_set.txt",
    "../cmu/new_with_stress/dev_set.txt",
    "../cmu/new_with_stress/test_set.txt",
    "../cmu/new_with_stress/train_set.txt"]

out_files = [
    "../cmu/no_stress/dev_set.txt",
    "../cmu/no_stress/test_set.txt",
    "../cmu/no_stress/train_set.txt",
    "../cmu/stress/dev_set.txt",
    "../cmu/stress/test_set.txt",
    "../cmu/stress/train_set.txt"]

def crop(in_path, out_path):
    in_file = open(in_path)
    out_file = open(out_path, "w")
    for line in in_file:
        l = line.split()
        if len(l) > 19:
            continue
        first = l[0]
        del l[0]
        l.insert(0, START)
        for _ in range(20):
            l.append(STOP)
        l = l[0:20]
        line = reduce(lambda s1, s2: s1 + " " + s2, l, "")
        out_file.write(first + "\t" + line + "\n")
    out_file.close()


def crop2(in_path, out_path):
    in_file = open(in_path)
    out_file = open(out_path, "w")
    for line in in_file:
        l = line.split("\t")
        second = l[1]
        del l[1]
        chars = list(l[0])
        chars.insert(0, START)
        for _ in range(20):
            chars.append(STOP)
        chars = chars[0:22]
        word = reduce(lambda s1, s2: s1 + " " + s2, chars, "")
        out_file.write(word + "\t" + second)
    out_file.close()



def crop3(in_path, out_path):
    in_file = open(in_path)
    out_file = open(out_path, "w")
    for line in in_file:
        l = line.split()
        if len(l) > 19:
            continue
        first = l[0]
        del l[0]
        l.insert(0, START)
        for _ in range(20):
            l.append(STOP)
        l = l[0:20]
        second = reduce(lambda s1, s2: s1 + " " + s2, l, "")
        second = second.strip()

        chars = list(first)
        chars.insert(0, START)
        for _ in range(20):
            chars.append(STOP)
        chars = chars[0:22]
        first = reduce(lambda s1, s2: s1 + " " + s2, chars, "")
        first = first.strip()
        out_file.write(first + "\t" + second + "\n")
    out_file.close()

for i, o in zip(files, out_files):
    crop3(i, o)

