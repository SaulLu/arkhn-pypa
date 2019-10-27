import pdftotext
from pdf2image import convert_from_path, convert_from_bytes
import numpy as np
import pdftotree
from path import Path


def pdf2txt(input_path, output_path, singlefile=True):
    """Converts a pdf file into a txt file.

    Args:
        input_path (str): the path to the pdf file to convert
        output_path (str): the path to the folder to create the txt file in
        singlefile (bool): is true to write the whole file into a single one, is false to write each pdf page in a separate txt file. Defaults to True.
    """

    with open(input_path, "rb") as f_pdf:
        pdf = pdftotext.PDF(f_pdf)

    if singlefile:
        with open(output_path, "w+") as f_txt:
            for page in pdf:
                f_txt.write(page)
    else:
        i = 1
        for page in pdf:
            with open(output_path + Path("/Page") + str(i), "w+") as f_txt:
                f_txt.write(page)


def pdf2ppm(input_path, output_path, greyscale=True, dpi=20):
    """Converts a pdf file into ppm files (image).
    1 ppm file corresponds to 1 pdf page.

    Args:
        input_path (str): the path to the pdf file to convert
        output_path (str): the path to the folder to create the ppm/pgm file in
        greyscale (bool): is false if the image must be in colors, is true otherwise. Defaults to True.
        dpi (int): output resolution in dots per inch. Defaults to 20.
    """

    images = convert_from_path(
        input_path, output_folder=output_path, grayscale=greyscale, dpi=dpi
    )


def pdf2pix(input_path, page=1, greyscale=True, dpi=20):
    """
    Converts a pdf into an array of pixel values.
    If greyscale is true, returns a numpy array height x width ; else returns a numpy array height x width x 3 (using rgb format).

    Keyword arguments:
    input_path -- the path to the pdf file to convert
    page -- the index of the page to convert (starts from 1)
    greyscale -- is false if the image must be in colors, is true otherwise (default True)
    dpi -- resolution in dots per inch (default 20)
    """
    """Converts a pdf into an array of pixel values.

    Args:
        input_path (str): the path to the pdf file to convert
        page (int): the index of the page to convert (starts from 1). Defaults to 1.
        greyscale (bool): is false if the image must be in colors, is true otherwise. Defaults to True.
        dpi (int): output resolution in dots per inch. Defaults to 20.
    
    Returns:
        array of pixel values : numpy array of dimensions height x width if greyscale is True or height x width x 3 if greyscale is False (using RGB format).
    """
    images = convert_from_path(input_path, grayscale=greyscale, dpi=dpi)
    img = images[page - 1]
    if greyscale:
        return np.reshape(np.array(list(img.getdata())), img.size)
    else:
        return np.reshape(np.array(list(img.getdata())), img.size + (3,))


def pdf2html(input_path, output_path):
    """
    Converts a pdf into a html file.
    
    Keyword arguments:
    input_path -- the path to the pdf file to convert
    output_path -- the path to the folder to create the html file in
    """
    pdftotree.parse(input_path, html_path=output_path)


def test():
    """
    Tests pdf2txt and pdf2ppm thanks to the folder pdfLoaderTests.
    """
    pdf2txt(
        "pdfLoaderTests/2019-10_LS-ANS2019_TIHellemmes.pdf",
        "pdfLoaderTests/test_pdf2txt.txt",
    )
    pdf2ppm(
        "pdfLoaderTests/2019-10_LS-ANS2019_TIHellemmes.pdf",
        "pdfLoaderTests/test_pdf2ppm",
    )
    pdf2html(
        "pdfLoaderTests/2019-10_LS-ANS2019_TIHellemmes.pdf",
        "pdfLoaderTests/test_pdf2html/",
    )


if __name__ == "__main__":
    test()
