from mnist import MNIST

mndata = MNIST('samples')


def get_training():
    images, labels = mndata.load_training()

    allImages = mndata.process_images_to_lists(images)
    resultList = [0] * 9

    return [([v/255 for v in x], [1 if ri==labels[i] else 0 for ri,r in enumerate(resultList)]) for i,x in enumerate(allImages)]

def get_testing():
    images, labels = mndata.load_testing()

    allImages = mndata.process_images_to_lists(images)
    resultList = [0] * 9

    return [([v/255 for v in x], [1 if ri==labels[i] else 0 for ri,r in enumerate(resultList)]) for i,x in enumerate(allImages)]
