# 从../resources/train-images-idx3-ubyte 中读取数据
# 前四个字节为magic 必须是2049
# 之后四个字节为图片数量
# 之后四个字节为图片像素的行数
# 之后四个字节为图片像素的列数
# 每个像素用一个字节表示

def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic = f.read(4)
        magic = int.from_bytes(magic, "big")
        if magic != 2051:
            raise ValueError("Magic number does not match 2051")
        n = f.read(4)
        n = int.from_bytes(n, "big")
        print ("number of images: ", n)
        rows = f.read(4)
        rows = int.from_bytes(rows, "big")
        print ("rows: ", rows)
        cols = f.read(4)
        cols = int.from_bytes(cols, "big")
        print ("cols: ", cols)
        
        images = []
        for i in range(n):
            image = [0] * rows * cols
            for j in range(rows * cols):
                image[j] = int.from_bytes(f.read(1), "big")
            images.append(image)
    return images

def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic = f.read(4)
        magic = int.from_bytes(magic, "big")
        if magic != 2049:
            raise ValueError("Magic number does not match 2049")
        n = f.read(4)
        n = int.from_bytes(n, "big")
        print ("number of labels: ", n)
        
        labels = []
        for i in range(n):
            labels.append(int.from_bytes(f.read(1), "big"))
    return labels

if __name__ == "__main__":
    images = load_mnist_images("../resources/train-images-idx3-ubyte")
    labels = load_mnist_labels("../resources/train-labels-idx1-ubyte")