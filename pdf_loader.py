import pdftotext
from pdf2image import convert_from_path, convert_from_bytes
import numpy as np


def pdf2txt(input_path, output_path, singlefile = True):
    with open(input_path, "rb") as f_pdf:
        pdf = pdftotext.PDF(f_pdf)

    if singlefile:
        f_txt = open(output_path, 'w+')
        for page in pdf:
            f_txt.write(page)
    else:
        i = 1
        for page in pdf:
            f_txt = open(output_path + '/Page' + str(i), 'w+')
            f_txt.write(page)


def pdf2ppm(input_path, output_path, greyscale = True, size = (100, 100)):
    images = convert_from_path(input_path, output_folder = output_path, grayscale = greyscale, size = size)


def pdf2pix(input_path, page, greyscale = True, size = (100, 100)):
    images = convert_from_path(input_path, grayscale = greyscale, size = size)
    img = images[page - 1]
    if greyscale:
        return np.reshape(np.array(list(img.getdata())), img.size)
    else:
        return np.reshape(np.array(list(img.getdata())), img.size + (3,))

def test():
    pdf2txt("pdfLoaderTests/2019-10_LS-ANS2019_TIHellemmes.pdf", "pdfLoaderTests/test_pdf2txt.txt")
    pdf2ppm("pdfLoaderTests/2019-10_LS-ANS2019_TIHellemmes.pdf", "pdfLoaderTests/test_pdf2ppm")


if __name__=='__main__':
    test()