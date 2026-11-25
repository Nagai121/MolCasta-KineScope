"""GA1265+: augment the readout with matched exchanges of the generated fields."""
from .ga_mixture import main

if __name__=="__main__":
    main(stability=True,competitive=True,orient_capacity=True,field_augmentation=True)
