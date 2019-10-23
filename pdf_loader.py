import pdftotext
from pdf2image import convert_from_path, convert_from_bytes
import numpy as np


def pdf2txt(input_path, output_path, singlefile = True):
    """
    Converts a pdf file into a txt file.

    Keyword arguments:
    input_path -- the path to the pdf file to convert
    output_path -- the path to either a txt file or a folder to create the txt file in
    singlefile -- is true to write the whole file into a single one, is false to write each pdf page in a separate txt file (default True)
    """

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
    """
    Converts a pdf file into ppm files (image).
    1 ppm file corresponds to 1 pdf page.

    Keyword arguments:
    input_path -- the path to the pdf file to convert
    output_path -- the path to the folder to create the ppm files in
    greyscale -- is false if the image must be in colors, is true otherwise (default True)
    size -- (height, width) : size of the resulting pic in pixels (default (100, 100))
    """

    images = convert_from_path(input_path, output_folder = output_path, grayscale = greyscale, size = size)


def pdf2pix(input_path, page, greyscale = True, size = (100, 100)):
    """
    Converts a pdf into an array of pixel values.
    If greyscale is true, returns a numpy array height x width ; else returns a numpy array height x width x 3 (using rgb format).

    Keyword arguments:
    input_path -- the path to the pdf file to convert
    page -- the index of the page to convert (starts from 1)
    greyscale -- is false if the image must be in colors, is true otherwise (default True)
    size -- (height, width) : size of the resulting pic in pixels (default (100, 100))
    """
    images = convert_from_path(input_path, grayscale = greyscale, size = size)
    img = images[page - 1]
    if greyscale:
        return np.reshape(np.array(list(img.getdata())), img.size)
    else:
        return np.reshape(np.array(list(img.getdata())), img.size + (3,))

def test():
    """
    Tests pdf2txt and pdf2ppm thanks to the folder pdfLoaderTests.
    """
    pdf2txt("pdfLoaderTests/2019-10_LS-ANS2019_TIHellemmes.pdf", "pdfLoaderTests/test_pdf2txt.txt")
    pdf2ppm("pdfLoaderTests/2019-10_LS-ANS2019_TIHellemmes.pdf", "pdfLoaderTests/test_pdf2ppm")


if __name__=='__main__':
    test()