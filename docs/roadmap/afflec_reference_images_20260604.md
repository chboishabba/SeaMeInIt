# Afflec Reference Images - 2026-06-04

The Afflec calibration reference images are local binary assets staged under:

`assets/reference_images/afflec/`

Project-owner exception: these Afflec reference images are allowed as a manual
binary commit for the calibration lane when the project owner explicitly chooses
to force-add them. Agent policy still prohibits agents from committing binary
files, so agents should keep the tracked manifest current and leave the actual
image commit to the project owner.

## Local Asset Manifest

| File | SHA-256 |
| --- | --- |
| `1365.webp` | `6ab48a11f3c6ed89b1f8246cdea4d1a1684e01e4af0795b1d0ee9d90d2385084` |
| `1_PAY-EXCLUSIVE-Ben-Affleck-Defends-His-Massive-Back-Tattoo-After-Admitting-Sentiment-Ran-Against-It.avif` | `9ebdc2df3463aa3a631164bd0a96b96a0ff5515954b1aa226f69ff0ee47589bc` |
| `Screenshot_20260604_135454.png` | `f1fd8614bddb8812d13386e619b5ec4cdabe8635253615a12c4e13de6ef42461` |
| `ben-affleck-b-2000-9d49258cbae143eb87d2c81462be1e27.jpg` | `b78ba5f7deb1a236dc04399c858ea0965475d9875e120105d20985f3cf211233` |
| `ben-affleck-beach-filming-tattoo.webp` | `493ba775bd11d3495db8f73b238786a9c04d898d50158fbd7492e4741448161e` |
| `gettyimages-1233897170-2048x2048.webp` | `7c9dc7774fe8f93a57da3778b516e5c6ab17b8da6213b2ac9f66f54610a11e52` |
| `gettyimages-2150568038-2048x2048.webp` | `b14b654e168224f947c56ea1c8b6a0c263de205713dc810ff384dc4d0227a162` |
| `gettyimages-2211353811-2048x2048.webp` | `18de1c56cc47c9a620bfdff2d340a568295c3ae93a3ed482810526016bbc7379` |
| `gettyimages-2256160830-2048x2048.webp` | `f7bb09d65bbfcbe87090979c5c2f5c6501762fb4a13afc3eed614013c18df5c2` |
| `images.jpg` | `b33b6b291b7cc6122999ae40055303e36a54161dce483faafa48a0091f2fb5da` |

## Recreate Local Staging

After supplying the source files locally, run:

```bash
mkdir -p assets/reference_images/afflec
cp /home/c/Downloads/gettyimages-2256160830-2048x2048.webp \
  /home/c/Downloads/gettyimages-2150568038-2048x2048.webp \
  /home/c/Downloads/gettyimages-2211353811-2048x2048.webp \
  /home/c/Downloads/gettyimages-1233897170-2048x2048.webp \
  /home/c/Downloads/Screenshot_20260604_135454.png \
  /home/c/Downloads/ben-affleck-b-2000-9d49258cbae143eb87d2c81462be1e27.jpg \
  /home/c/Downloads/ben-affleck-beach-filming-tattoo.webp \
  /home/c/Downloads/1365.webp \
  /home/c/Downloads/images.jpg \
  /home/c/Downloads/1_PAY-EXCLUSIVE-Ben-Affleck-Defends-His-Massive-Back-Tattoo-After-Admitting-Sentiment-Ran-Against-It.avif \
  assets/reference_images/afflec/
sha256sum assets/reference_images/afflec/*
```

## Owner Manual Commit

Because `assets/` is ignored and agents must not commit binary files, the
project owner can commit this approved reference-image exception manually with:

```bash
git add docs/roadmap/afflec_reference_images_20260604.md TODO.md
git add -f assets/reference_images/afflec/1365.webp \
  assets/reference_images/afflec/1_PAY-EXCLUSIVE-Ben-Affleck-Defends-His-Massive-Back-Tattoo-After-Admitting-Sentiment-Ran-Against-It.avif \
  assets/reference_images/afflec/Screenshot_20260604_135454.png \
  assets/reference_images/afflec/ben-affleck-b-2000-9d49258cbae143eb87d2c81462be1e27.jpg \
  assets/reference_images/afflec/ben-affleck-beach-filming-tattoo.webp \
  assets/reference_images/afflec/gettyimages-1233897170-2048x2048.webp \
  assets/reference_images/afflec/gettyimages-2150568038-2048x2048.webp \
  assets/reference_images/afflec/gettyimages-2211353811-2048x2048.webp \
  assets/reference_images/afflec/gettyimages-2256160830-2048x2048.webp \
  assets/reference_images/afflec/images.jpg
git commit -m "Add Afflec calibration reference images"
```
