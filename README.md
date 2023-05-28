# openpifpaf

# Introduction

The goal of our project was to do 2D object detection using the OpenPifPaf model provided here. Indeed, OpenPifPaf already has a module called 'cifdet' that detects the 2D bounding boxes of the objects. However, this is not using the ??? JSP CE QUI VA PAS.

Thus, we decided to implement 2D object detection using the 'cifcaf' module normally used for keypoints detection. 

# Methodology

# What is not implemented yet

# Conclusion 

# How to use it?

## [Guide](https://vita-epfl.github.io/openpifpaf/intro.html)

The detailed documentation concerning the OpenPifPaf model can be founf in the __[OpenPifPaf Guide](https://vita-epfl.github.io/openpifpaf/intro.html)__.
For developers, there is also the
__[DEV Guide](https://vita-epfl.github.io/openpifpaf/dev/intro.html)__
which is the same guide but based on the latest code in the `main` branch.

We based our work on this guide, please refer to it if the information provided below are not sufficient.


## Installation

This version of OpenPifPaf (`openpifpaf-vita`) cannot co-exist with the original one ([`openpifpaf`](https://github.com/openpifpaf/openpifpaf)) in the same environment.
If you have previously installed the package `openpifpaf`, remove it before installation to avoid conflicts.

This project was forked from [OpenPifPaf v0.13.1](https://github.com/openpifpaf/openpifpaf/releases/tag/v0.13.1) and developed separately from version v0.14.0 on.

Do not clone this repository.
Make sure there is no folder named `openpifpaf-vita` in your current directory, and run:
```sh
pip3 install openpifpaf-vita
```

You need to install `matplotlib` to produce visual outputs:
```sh
pip3 install matplotlib
```

To modify OpenPifPaf itself, please follow [Modify Code](https://vita-epfl.github.io/openpifpaf/dev.html#modify-code).


## Examples

![example image with overlaid pose predictions](https://github.com/vita-epfl/openpifpaf/raw/main/docs/coco/000000081988.jpg.predictions.jpeg)

Image credit: "[Learning to surf](https://www.flickr.com/photos/fotologic/6038911779/in/photostream/)" by fotologic which is licensed under [CC-BY-2.0].<br />
Created with:
```sh
python3 -m openpifpaf.predict docs/coco/000000081988.jpg --image-output
```

---

Here is the [tutorial for body, foot, face and hand keypoints](https://vita-epfl.github.io/openpifpaf/plugins_wholebody.html). Example:
![example image with overlaid wholebody pose predictions](https://raw.githubusercontent.com/vita-epfl/openpifpaf/main/docs/soccer.jpeg.predictions.jpeg)

Image credit: [Photo](https://de.wikipedia.org/wiki/Kamil_Vacek#/media/Datei:Kamil_Vacek_20200627.jpg) by [Lokomotive74](https://commons.wikimedia.org/wiki/User:Lokomotive74) which is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).<br />
Created with:
```sh
python -m openpifpaf.predict guide/wholebody/soccer.jpeg \
  --checkpoint=shufflenetv2k30-wholebody --line-width=2 --image-output
```

---

Here is the [tutorial for car keypoints](https://vita-epfl.github.io/openpifpaf/plugins_apollocar3d.html). Example:
![example image cars](https://raw.githubusercontent.com/vita-epfl/openpifpaf/main/docs/peterbourg.jpg.predictions.jpeg)

Image credit: [Photo](https://commons.wikimedia.org/wiki/File:Streets_of_Saint_Petersburg,_Russia.jpg) by [Ninaras](https://commons.wikimedia.org/wiki/User:Ninaras) which is licensed under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Created with:
```sh
python -m openpifpaf.predict guide/images/peterbourg.jpg \
  --checkpoint shufflenetv2k16-apollo-24 -o images \
  --instance-threshold 0.05 --seed-threshold 0.05 \
  --line-width 4 --font-size 0
```

---

Here is the [tutorial for animal keypoints (dogs, cats, sheep, horses and cows)](https://vita-epfl.github.io/openpifpaf/plugins_animalpose.html). Example:
![example image cars](https://raw.githubusercontent.com/vita-epfl/openpifpaf/main/docs/tappo_loomo.jpg.predictions.jpeg)


```sh
python -m openpifpaf.predict guide/images tappo_loomo.jpg \
  --checkpoint=shufflenetv2k30-animalpose \
  --line-width=6 --font-size=6 --white-overlay=0.3 \
  --long-edge=500
```


# Commercial License

The open source license is in the [LICENSE](https://github.com/vita-epfl/openpifpaf/blob/main/LICENSE) file.
This software is also available for licensing via the EPFL Technology Transfer
Office (https://tto.epfl.ch/, info.tto@epfl.ch).


[CC-BY-2.0]: https://creativecommons.org/licenses/by/2.0/
