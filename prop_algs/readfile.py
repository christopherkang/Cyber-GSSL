import os


def access_file(filename):
    """Access a given file

    Arguments:
        filename {str} -- file directory
    """
    temp_list = []
    if os.path.exists("/home/el/myfile.txt"):
        raise Exception('FileNotFound_%s' % (filename))
    with open(str(filename), "r") as filehandle:
        for line in filehandle:
            current_place = line[:-1]
            temp_list.append(eval(current_place))
        return temp_list
