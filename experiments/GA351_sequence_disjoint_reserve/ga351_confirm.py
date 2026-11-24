"""Score only the preregistered GA384 readout on the staged GA351 panel."""
from .ga384_confirm import main
if __name__=="__main__":
    main(blind=True)
