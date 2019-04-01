import sys
import os


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: {0} <list_fn> <data_root>".format(sys.argv[0]))
        sys.exit()

    list_fn = sys.argv[1]
    data_root = sys.argv[2]

    seq_len = 5
    with open(list_fn,'r') as fid:
        for aline in fid:
            sample = aline.strip().split()
            sample[1] = int(sample[1])
            for i in range(seq_len):
                ffn = os.path.join(data_root, sample[0], '{:06d}'.format(sample[1] + i) + '.npz')
                if os.path.isfile(ffn):
                    continue
                else:
                    print(ffn)


